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
    """Read initial variables at the beginning of a MWFN file."""
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
    """Read basis set information typically labelled '# Basis function information'."""
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
    """Read atoms section typically labelled '# Atom information'."""
    data = {
        "atnums": np.empty(natom, int),
        "atcorenums": np.empty(natom, float),
        "atcoords": np.empty((natom, 3), float),
    }

    # skip lines until "$Centers" section is reached
    line = next(lit)
    while '$Centers' not in line and line is not None:
        line = next(lit)

    for atom in range(natom):
        words = next(lit).split()
        data["atnums"][atom] = words[2]
        data["atcorenums"][atom] = words[3]
        data["atcoords"][atom, :] = words[4:7]
    # coordinates are in angstrom in MWFN, so they are converted to atomic units
    data["atcoords"] *= angstrom

    # check atomic numbers
    if min(data["atnums"]) <= 0:
        lit.error(f"Atomic numbers should be positive integers! Read atnums= {data['atnums']}")

    return data


def _load_helper_shells(lit: LineIterator, nshell: int) -> dict:
    """Read shell types, centers, and contraction degrees."""
    sections = ["$Shell types", "$Shell centers", "$Shell contraction"]
    var_name = ["shell_types", "shell_centers", "shell_ncons"]

    # read lines until '$Shell types' is reached
    line = next(lit)
    while sections[0] not in line and line is not None:
        line = next(lit)

    data = {}
    for section, name in zip(sections, var_name):
        if not line.startswith(section):
            lit.error(f"Expected line to start with {section}, but got line={line}.")
        data[name] = _load_helper_section(lit, nshell, ' ', 0, int)
        line = next(lit)
    lit.back(line)
    return data


def _load_helper_section(lit: LineIterator, nprim: int, start: str, skip: int,
                         dtype: np.dtype) -> np.ndarray:
    """Read single or multiple line(s) section."""
    section = []
    while len(section) < nprim:
        line = next(lit)
        if not line.startswith(start):
            lit.error(f"Expected line to start with {start}! Got line={line}")
        section.extend(line.split()[skip:])
    return np.array(section, dtype)


def _load_helper_mo(lit: LineIterator, n_basis: int, n_mo: int) -> dict:
    """Read molecular orbitals section typically labelled '# Orbital information'."""
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
        data["mo_numbers"][index] = line.split()[1]
        data["mo_type"][index] = next(lit).split()[1]
        data["mo_energies"][index] = next(lit).split()[1]
        data["mo_occs"][index] = next(lit).split()[1]
        data["mo_sym"][index] = next(lit).split()[1]
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
    # -----
    # 1) mwfn is a fortran program which loads *.mwfn by locating the line with the keyword,
    #    then uses `backspace`, then begins reading. Despite this flexibility, it is stated by
    #    the authors that the order of section, and indeed, entries in general, must be fixed.
    #    With this in mind the input utilized some hard-coding since order should be fixed.
    # 2) mwfn ignores lines beginning with `#`.

    # read title (assuming title is the 1st line, which seems to be the standard)
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
    if not next(lit).startswith("$Primitive exponents"):
        lit.error("Expected '$Primitive exponents' section!")
    data["exponents"] = _load_helper_section(lit, data["Nprimshell"], '', 0, float)
    if not next(lit).startswith("$Contraction coefficients"):
        lit.error("Expected '$Contraction coefficients' section!")
    data["coeffs"] = _load_helper_section(lit, data["Nprimshell"], '', 0, float)

    # get number of basis & molecular orbitals (MO)
    # Note: MWFN includes virtual orbitals, so num_mo equals number independent basis functions
    num_basis = data["Nindbasis"]
    num_mo = data["Nindbasis"]
    if data["mo_kind"] == "unrestricted":
        num_mo *= 2
    # load MO information
    data.update(_load_helper_mo(lit, num_basis, num_mo))

    return data


@document_load_one("MWFN", ['atcoords', 'atnums', 'atcorenums', 'energy',
                            'mo', 'obasis', 'extra', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    inp = _load_mwfn_low(lit)

    # store certain information loaded from MWFN in extra dictionary
    extra = {'wfntype': inp['Wfntype'],
             'nindbasis': inp['Nindbasis'],
             'mo_sym': inp['mo_sym'],
             'full_virial_ratio': inp['VT_ratio']}

    # Build MolecularBasis instance
    # Unlike WFN, MWFN does include orbital expansion coefficients.
    shells = []
    counter = 0
    for center, stype, ncon in zip(inp['shell_centers'], inp['shell_types'], inp['shell_ncons']):
        shells.append(Shell(
            center,
            [abs(stype)],
            ['p' if stype < 0 else 'c'],
            inp['exponents'][counter:counter + ncon],
            inp['coeffs'][counter:counter + ncon][:, np.newaxis]
        ))
        counter += ncon
    obasis = MolecularBasis(shells, CONVENTIONS, 'L2')
    # check number of basis functions
    if obasis.nbasis != inp["Nbasis"]:
        raise ValueError(f"Number of basis in MolecularBasis is not equal to the 'Nbasis'. "
                         f"{obasis.nbasis} != {inp['Nbasis']}")

    # Determine number of orbitals of each kind.
    if inp["mo_kind"] == "restricted":
        norba = len(inp["mo_type"])
        norbb = norba
    else:
        # Legend
        # 0: restricted doubly occupied
        # 1: alpha
        # 2: beta
        norba = (inp["mo_type"] == 1).sum()
        norbb = (inp["mo_type"] == 2).sum()
        assert (inp["mo_type"] == 0).sum() == 0
    # Build MolecularOrbitals instance
    mo = MolecularOrbitals(
        inp["mo_kind"], norba, norbb, inp['mo_occs'], inp['mo_coeffs'],
        inp['mo_energies'], None
    )
    # check number of electrons
    if mo.nelec != inp['Naelec'] + inp['Nbelec']:
        raise ValueError(f"Number of electrons in MolecularOrbitals is not equal to the sum of "
                         f"'Naelec' and 'Nbelec'. {mo.nelec} != {inp['Naelec']} + {inp['Nbelec']}")
    if mo.occsa.sum() != inp['Naelec']:
        raise ValueError(f"Number of alpha electrons in MolecularOrbitals is not equal to the "
                         f"'Naelec'. {mo.occsa.sum()} != {inp['Naelec']}")
    if mo.occsb.sum() != inp['Nbelec']:
        raise ValueError(f"Number of beta electrons in MolecularOrbitals is not equal to the "
                         f"'Nbelec'. {mo.occsb.sum()} != {inp['Nbelec']}")

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
