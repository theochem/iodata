# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Multiwfn MWFN file format."""

from typing import Tuple

import numpy as np

from ..basis import HORTON2_CONVENTIONS, MolecularBasis, Shell
from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..utils import LineIterator, angstrom

__all__ = []

PATTERNS = ['*.mwfn']

# From the MWFN chemrxiv paper
# https://chemrxiv.org/articles/Mwfn_A_Strict_Concise_and_Extensible_Format
# _for_Electronic_Wavefunction_Storage_and_Exchange/11872524
# For cartesian shells
# S shell: S
# P shell: X, Y, Z
# D shell: XX, YY, ZZ, XY, XZ, YZ
# F shell: XXX, YYY, ZZZ, XYY, XXY, XXZ, XZZ, YZZ, YYZ, XYZ
# G shell: ZZZZ, YZZZ, YYZZ, YYYZ, YYYY, XZZZ, XYZZ, XYYZ, XYYY, XXZZ, XXYZ, XXYY, XXXZ, XXXY, XXXX
# H shell: ZZZZZ, YZZZZ, YYZZZ, YYYZZ, YYYYZ, YYYYY, XZZZZ, XYZZZ, XYYZZ, XYYYZ, XYYYY, XXZZZ,
#           XXYZZ, XXYYZ, XXYYY, XXXZZ, XXXYZ, XXXYY, XXXXZ, XXXXY, XXXXX
# For pure shells, the order is
# D shell: D 0, D+1, D-1, D+2, D-2
# F shell: F 0, F+1, F-1, F+2, F-2, F+3, F-3
# G shell: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4

CONVENTIONS = {
    (4, 'p'): HORTON2_CONVENTIONS[(4, 'p')],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'c'): ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy',
               'xxzz', 'xxyz', 'xxyy', 'xxxz', 'xxxy', 'xxxx'],
    (5, 'c'): ['zzzzz', 'yzzzz', 'yyzzz', 'yyyzz', 'yyyyz', 'yyyyy', 'xzzzz', 'xyzzz',
               'xyyzz', 'xyyyz', 'xyyyy', 'xxzzz', 'xxyzz', 'xxyyz', 'xxyyy', 'xxxzz',
               'xxxyz', 'xxxyy', 'xxxxz', 'xxxxy', 'xxxxx'],
}


def _load_helper_opener(lit: LineIterator) -> dict:
    """Read initial variables."""
    keys = {"Wfntype": int, "Charge": float, "Naelec": float, "Nbelec": float, "E_tot": float,
            "VT_ratio": float, "Ncenter": int}
    max_count = len(keys)
    count = 0
    data = {}
    while count < max_count:
        line = next(lit)
        for name, ftype in keys.items():
            if name in line:
                data[name] = ftype(line.split('=')[1].strip())
                count += 1

    # check values parsed
    if data["Ncenter"] <= 0:
        lit.error(f"Ncenter should be a positive integer! Read Ncenter= {data['Ncenter']}")
    # Possible values of Wfntype (wavefunction type):
    #     0: Restricted closed - shell single - determinant wavefunction(e.g.RHF, RKS)
    #     1: Unrestricted open - shell single - determinant wavefunction(e.g.UHF, UKS)
    #     2: Restricted open - shell single - determinant wavefunction(e.g.ROHF, ROKS)
    #     3: Restricted multiconfiguration wavefunction(e.g.RMP2, RCCSD)
    #     4: Unrestricted multiconfiguration wavefunction(e.g.UMP2, UCCSD)
    if data["Wfntype"] in [0, 2, 3]:
        # restricted wavefunction
        data["mo_kind"] = "restricted"
    elif data["Wfntype"] in [1, 4]:
        # unrestricted wavefunction
        data["mo_kind"] = "unrestricted"
    else:
        lit.error(f"Wavefunction type cannot be determined. Read Wfntype= {data['Wfntype']}")

    return data


def _load_helper_basis(lit: LineIterator) -> dict:
    """Read initial variables."""
    # Nprims must be last or else it gets read in with Nprimshell
    keys = ["Nbasis", "Nindbasis", "Nshell", "Nprimshell", "Nprims"]
    count = 0
    data = {}
    next(lit)
    while count < len(keys):
        line = next(lit)
        for name in keys:
            if name in line:
                data[name] = int(line.split('=')[1].strip())
                count += 1
                break
    return data


def _load_helper_atoms(lit: LineIterator, natom: int) -> dict:
    """Read the coordinates of the atoms."""
    data = {"atnums": np.empty(natom, int), "atcorenums": np.empty(natom, float),
            "atcoords": np.empty((natom, 3), float)}

    # skip lines until "$Centers" section is reached
    line = next(lit)
    while '$Centers' not in line and line is not None:
        line = next(lit)

    for atom in range(natom):
        words = next(lit).split()
        data["atnums"][atom] = int(words[2].strip())
        data["atcorenums"][atom] = float(words[3].strip())
        data["atcoords"][atom, :] = words[4:7]
    # coordinates are in angstrom in MWFN, so they are converted to atomic units
    data["atcoords"] *= angstrom

    # check atomic numbers
    if min(data["atnums"]) <= 0:
        lit.error(f"Atomic numbers should be positive integers! Read atnums= {data['atnums']}")

    return data


def _load_helper_shells(lit: LineIterator, nshell: int) -> dict:
    """Read one section of MO information."""
    keys = ["Shell types", "Shell centers", "Shell contraction"]
    data = {}
    line = next(lit)
    while keys[0] not in line and line is not None:
        line = next(lit)
    assert line.startswith('$' + keys[0])
    data["shell_types"] = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + keys[1])
    data["shell_centers"] = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + keys[2])
    data["shell_contraction_degrees"] = _load_helper_section(lit, nshell, ' ', 0, int)
    return data


def _load_helper_prims(lit: LineIterator, nprimshell: int) -> np.ndarray:
    """Read SHELL CENTER, SHELL TYPE, and SHELL CONTRACTION DEGREES sections."""
    next(lit)  # skip line
    # concatenate list of arrays into a single array of length nshell
    array = _load_helper_section(lit, nprimshell, '', 0, float)
    assert len(array) == nprimshell
    return array


def _load_helper_section(lit: LineIterator, nprim: int, start: str, skip: int,
                         dtype: np.dtype) -> np.ndarray:
    """Read SHELL CENTER, SHELL TYPE, and SHELL CONTRACTION DEGREES sections."""
    section = []
    while len(section) < nprim:
        line = next(lit)
        assert line.startswith(start)
        words = line.split()
        section.extend(words[skip:])
    assert len(section) == nprim
    return np.array(section).astype(dtype)


def _load_helper_mo(lit: LineIterator, n_basis: int, n_mo: int) -> dict:
    """Read one section of MO information."""

    data = {
        "mo_numbers": np.empty(n_mo, int),
        "mo_type": np.empty(n_mo, int),
        "mo_energies": np.empty(n_mo, float),
        "mo_occs": np.empty(n_mo, float),
        "mo_sym": np.empty(n_mo, str),
        "mo_coeffs": np.empty([n_basis, n_mo], float),
    }

    for index in range(n_mo):
        line = next(lit)
        while 'Index' not in line:
            line = next(lit)
        assert line.startswith('Index')
        data["mo_numbers"][index] = int(line.split()[1])
        data["mo_type"][index] = int(next(lit).split()[1])
        data["mo_energies"][index] = float(next(lit).split()[1])
        data["mo_occs"][index] = float(next(lit).split()[1])
        data["mo_sym"][index] = str(next(lit).split()[1])
        # skip "$Coeff line
        next(lit)
        data["mo_coeffs"][:, index] = _load_helper_section(lit, n_basis, '', 0, float)

    return data


def _load_mwfn_low(lit: LineIterator) -> dict:
    """Load data from a MWFN file into arrays.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    """
    # Note:
    # ---------
    # mwfn is a fortran program which loads *.mwfn by locating the line with the keyword,
    #  then uses `backspace`, then begins reading. Despite this flexibility, it is stated by
    #  the authors that the order of section, and indeed, entries in general, must be fixed.
    #  With this in mind the input utilized some hardcoding since order should be fixed.
    #
    #  mwfn ignores lines beginning with `#`.
    # read sections of mwfn file
    # This assumes title is on first line which seems to be the standard

    # read title
    data = {"title": next(lit).strip()}

    # load Wfntype, Charge, Naelec, Nbelec, E_tot, VT_ratio, & Ncenter
    data.update(_load_helper_opener(lit))

    # load atnums, atcorenums, & atcoords (in atomic units)
    data.update(_load_helper_atoms(lit, data["Ncenter"]))

    # load Nbasis, Nindbasis, Nprims, Nshell, & Nprimshell
    data.update(_load_helper_basis(lit))

    # load shell_types, shell_centers, & shell_contraction_degrees
    data.update(_load_helper_shells(lit, data["Nshell"]))
    # IOData indices start at 0, so the centers are shifted
    data["shell_centers"] -= 1

    # load primitive exponents & coefficients
    data["exponents"] = _load_helper_prims(lit, data["Nprimshell"])
    data["coeffs"] = _load_helper_prims(lit, data["Nprimshell"])

    # get number of basis & molecular orbitals (MO)
    # Note: MWFN includes virtual orbitals, so num_mo equals number independent basis functions
    num_basis = data["Nindbasis"]
    num_mo = data["Nindbasis"]
    if data["mo_kind"] is "unrestricted":
        num_mo *= 2
    # load MO information
    data.update(_load_helper_mo(lit, num_basis, num_mo))

    # TODO: add density matrix and overlap

    return data


def _build_obasis(shell_map: np.ndarray, shell_types: np.ndarray,
                  exponents: np.ndarray, prim_per_shell: np.ndarray,
                  coeffs: np.ndarray,
                  ) -> Tuple[MolecularBasis]:
    """Construct a basis set using the arrays read from a MWFN file.

    Parameters
    -------------
    shell_map:  np.ndarray (integer)
        Index of what atom the shell is centered on. The mwfn file refers to this section
        as `Shell centers`. Mwfn indices start at 1, this has been modified and starts
        at 0 here. For water (O, H, H) with 6-31G, this would be an array like
        [0, 0, 0, 0, 0, 1, 1, 2, 2]. , `O` in 6-31G has 5 shells and`H` has two shells.
    shell_types: np.ndarray (integer)
        Angular momentum of the shell. Indices start at 0 for 's' orbital, 1 for 'p' etc.
        For 6-31G for a heavy atom this would be [0, 0, 1, 0, 1] corresponding
        to [1s, 2s, 2p, 2s, 2p]
    exponents: np.ndarray (float)
        Gaussian function decay exponents for the primitives in the basis set.
    prim_per_shell: np.ndarray (integer)
        Array denoting the number of primitives per shell. If basis set is 6-31G this will be
        [6, 3, 3, 1, 1] if the atom is a heavy atom. This corresponds to
        [1s, 2s, 2p, 2s, 2p]. If additional atoms are present, the array is extended.
    coeffs: np.ndarray (float)
        Array of same length as `exponents` containing orbital expansion coefficients.
    """
    shells = []
    counter = 0
    # First loop over all shells
    for i, n in enumerate(prim_per_shell):
        shells.append(Shell(
            shell_map[i],
            [abs(shell_types[i])],
            ['p' if shell_types[i] < 0 else 'c'],
            exponents[counter:counter + n],
            coeffs[counter:counter + n][:, np.newaxis]
        ))
        counter += n
    del shell_map
    del shell_types
    del prim_per_shell
    del exponents
    del coeffs

    obasis = MolecularBasis(tuple(shells), CONVENTIONS, 'L2')
    return obasis


@document_load_one("MWFN", ['atcoords', 'atnums', 'atcorenums', 'energy',
                            'mo', 'obasis', 'extra', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    inp = _load_mwfn_low(lit)

    # MWFN contains more information than most formats, so the following dict
    # stores some "extra" stuff.
    extra = {
        'mo_sym': inp['mo_sym'], 'mo_type': inp['mo_type'], 'mo_numbers': inp['mo_numbers'],
        'wfntype': inp['Wfntype'], 'nelec_a': inp['Naelec'], 'nelec_b': inp['Nbelec'],
        'nbasis': inp['Nbasis'], 'nindbasis': inp['Nindbasis'], 'nprims': inp['Nprims'],
        'nshells': inp['Nshell'], 'nprimshells': inp['Nprimshell'],
        'shell_types': inp['shell_types'], 'shell_centers': inp['shell_centers'],
        'shell_contraction_degrees': inp['shell_contraction_degrees'],
        'full_virial_ratio': inp['VT_ratio']}

    # Unlike WFN, MWFN does include orbital expansion coefficients.
    obasis = _build_obasis(inp['shell_centers'],
                           inp['shell_types'],
                           inp['exponents'],
                           inp['shell_contraction_degrees'],
                           inp['coeffs'],
                           )
    # MFWN provides number of alpha and beta electrons, this is a double check
    # mo_type (integer, scalar): Orbital type
    #     0: Alpha + Beta (i.e. spatial orbital)
    #     1: Alpha
    #     2: Beta
    # TODO calculate number of alpha and beta electrons manually.

    # Build the molecular orbitals
    mo = MolecularOrbitals(inp["mo_kind"],
                           inp['Naelec'],
                           inp['Nbelec'],
                           inp['mo_occs'],
                           inp['mo_coeffs'],
                           inp['mo_energies'],
                           None,
                           )

    return {
        'title': inp['title'],
        'atcoords': inp['atcoords'],
        'atnums': inp['atnums'],
        'atcorenums': inp['atcorenums'],
        'obasis': obasis,
        'mo': mo,
        'energy': inp['E_tot'],
        'extra': extra,
    }
