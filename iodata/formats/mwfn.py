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
import scipy.constants as spc

from ..basis import HORTON2_CONVENTIONS, MolecularBasis, Shell
from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..utils import LineIterator

__all__ = ['load_mwfn_low']

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
    (9, 'p'): HORTON2_CONVENTIONS[(9, 'p')],
    (8, 'p'): HORTON2_CONVENTIONS[(8, 'p')],
    (7, 'p'): HORTON2_CONVENTIONS[(7, 'p')],
    (6, 'p'): HORTON2_CONVENTIONS[(6, 'p')],
    (5, 'p'): HORTON2_CONVENTIONS[(5, 'p')],
    (4, 'p'): HORTON2_CONVENTIONS[(4, 'p')],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'c'): HORTON2_CONVENTIONS[(4, 'c')][::-1],
    (5, 'c'): HORTON2_CONVENTIONS[(5, 'c')][::-1],
    (6, 'c'): HORTON2_CONVENTIONS[(6, 'c')][::-1],
    (7, 'c'): HORTON2_CONVENTIONS[(7, 'c')][::-1],
    (8, 'c'): HORTON2_CONVENTIONS[(8, 'c')][::-1],
    (9, 'c'): HORTON2_CONVENTIONS[(9, 'c')][::-1],
}


def _load_helper_opener(lit: LineIterator, keys: list) -> Tuple[int, float, float,
                                                                float, float, float, int]:
    """Read initial variables."""
    max_count = len(keys)
    count = 0
    d = {}
    while count < max_count:
        line = next(lit)
        for name in keys:
            if name in line:
                d[name] = line.split('=')[1].strip()
                count += 1
    return int(d['Wfntype']), float(d['Charge']), float(d['Naelec']), float(d['Nbelec']),\
        float(d['E_tot']), float(d['VT_ratio']), int(d['Ncenter'])


def _load_helper_basis(lit: LineIterator) -> Tuple[int, int, int, int, int]:
    """Read initial variables."""
    # Nprims must be last or else it gets read in with Nprimshell
    basis_keywords = ["Nbasis", "Nindbasis", "Nshell", "Nprimshell", "Nprims", ]
    max_count = len(basis_keywords)
    count = 0
    d = {}
    next(lit)
    while count < max_count:
        line = next(lit)
        for name in basis_keywords:
            if name in line:
                d[name] = int(line.split('=')[1].strip())
                count += 1
                break
    return d['Nbasis'], d['Nindbasis'], d['Nprims'], d['Nshell'], d['Nprimshell']


def _load_helper_atoms(lit: LineIterator, num_atoms: int) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray]:
    """Read the coordinates of the atoms."""
    atnums = np.empty(num_atoms, int)
    atcorenums = np.empty(num_atoms, float)
    atcoords = np.empty((num_atoms, 3), float)
    line = next(lit)
    while '$Centers' not in line and line is not None:
        line = next(lit)

    for atom in range(num_atoms):
        line = next(lit)
        atnums[atom] = int(line.split()[2].strip())
        atcorenums[atom] = float(line.split()[3].strip())
        # extract atomic coordinates
        coords = line.split()
        atcoords[atom, :] = [coords[4], coords[5], coords[6]]
        # return but convert angstroms to amu
    return atnums, atcorenums, atcoords / spc.value(u'atomic unit of length') / 1E10


def _load_helper_shells(lit: LineIterator, nshell: int, starts: list) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one section of MO information."""
    line = next(lit)
    while starts[0] not in line and line is not None:
        line = next(lit)
    assert line.startswith('$' + starts[0])
    shell_types = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + starts[1])
    centers = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + starts[2])
    degrees = _load_helper_section(lit, nshell, ' ', 0, int)
    return shell_types, centers, degrees


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


def _load_helper_mo(lit: LineIterator, nbasis: int) -> Tuple[int, float, float,
                                                             np.ndarray, int, str]:
    """Read one section of MO information."""
    line = next(lit)
    while 'Index' not in line:
        line = next(lit)

    assert line.startswith('Index')
    number = int(line.split()[1])
    mo_type = int(next(lit).split()[1])
    energy = float(next(lit).split()[1])
    occ = float(next(lit).split()[1])
    sym = str(next(lit).split()[1])
    next(lit)  # skip line
    coeffs = _load_helper_section(lit, nbasis, '', 0, float)
    return number, occ, energy, coeffs, mo_type, sym


def load_mwfn_low(lit: LineIterator) -> dict:
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
    title = next(lit).strip()
    opener_keywords = ["Wfntype", "Charge", "Naelec", "Nbelec", "E_tot", "VT_ratio", "Ncenter"]
    wfntype, charge, nelec_a, nelec_b, \
        energy, vt_ratio, num_atoms = _load_helper_opener(lit, opener_keywords)
    # coordinates are in Angstrom in MWFN
    atnums, atcorenums, atcoords = _load_helper_atoms(lit, num_atoms)
    nbasis, nindbasis, nprim, nshell, nprimshell = _load_helper_basis(lit)
    keywords = ["Shell types", "Shell centers", "Shell contraction"]
    shell_types, shell_centers, prim_per_shell = _load_helper_shells(lit, nshell, keywords)
    # HORTON indices start at 0 because Pythons do.
    shell_centers -= 1
    assert wfntype < 5
    assert num_atoms > 0
    assert min(atnums) >= 0
    assert len(shell_types) == nshell
    assert len(shell_centers) == nshell
    assert len(prim_per_shell) == nshell
    exponent = _load_helper_prims(lit, nprimshell)
    coeffs = _load_helper_prims(lit, nprimshell)
    # number of MO's should equal number of independent basis functions. MWFN inc. virtual orbitals.
    num_coeffs = nindbasis
    if wfntype in [0, 2, 3]:
        # restricted wave function
        num_mo = nindbasis
    elif wfntype in [1, 4]:
        # unrestricted wavefunction
        num_mo = 2 * nindbasis

    mo_numbers = np.empty(num_mo, int)
    mo_type = np.empty(num_mo, int)
    mo_occs = np.empty(num_mo, float)
    mo_sym = np.empty(num_mo, str)
    mo_energies = np.empty(num_mo, float)
    mo_coeffs = np.empty([num_coeffs, num_mo], float)

    for mo in range(num_mo):
        mo_numbers[mo], mo_occs[mo], mo_energies[mo], mo_coeffs[:, mo], \
            mo_type[mo], mo_sym[mo] = _load_helper_mo(lit, num_coeffs)

    # TODO add density matrix and overlap

    return {'title': title, 'energy': energy, 'wfntype': wfntype,
            'nelec_a': nelec_a, 'nelec_b': nelec_b, 'charge': charge,
            'atnums': atnums, 'atcoords': atcoords, 'atcorenums': atcorenums,
            'nbasis': nbasis, 'nindbasis': nindbasis, 'nprims': nprim,
            'nshells': nshell, 'nprimshells': nprimshell, 'full_virial_ratio': vt_ratio,
            'shell_centers': shell_centers, 'shell_types': shell_types,
            'prim_per_shell': prim_per_shell, 'exponents': exponent, 'coeffs': coeffs,
            'mo_numbers': mo_numbers, 'mo_occs': mo_occs, 'mo_energies': mo_energies,
            'mo_coeffs': mo_coeffs, 'mo_type': mo_type, 'mo_sym': mo_sym}


def build_obasis(shell_map: np.ndarray, shell_types: np.ndarray,
                 exponents: np.ndarray, prim_per_shell: np.ndarray,
                 coeffs: np.ndarray,
                 ) -> Tuple[MolecularBasis]:
    """Based on the fchk modules basis building.

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
    inp = load_mwfn_low(lit)

    # MWFN contains more information than most formats, so the following dict
    # stores some "extra" stuff.
    mwfn_dict = {
        'mo_sym': inp['mo_sym'], 'mo_type': inp['mo_type'], 'mo_numbers': inp['mo_numbers'],
        'wfntype': inp['wfntype'], 'nelec_a': inp['nelec_a'], 'nelec_b': inp['nelec_b'],
        'nbasis': inp['nbasis'], 'nindbasis': inp['nindbasis'], 'nprims': inp['nprims'],
        'nshells': inp['nshells'], 'nprimshells': inp['nprimshells'],
        'shell_types': inp['shell_types'], 'shell_centers': inp['shell_centers'],
        'prim_per_shell': inp['prim_per_shell'], 'full_virial_ratio': inp['full_virial_ratio']}

    # Unlike WFN, MWFN does include orbital expansion coefficients.
    obasis = build_obasis(inp['shell_centers'],
                          inp['shell_types'],
                          inp['exponents'],
                          inp['prim_per_shell'],
                          inp['coeffs'],
                          )
    # wfntype(integer, scalar): Wavefunction type. Possible values:
    #     0: Restricted closed - shell single - determinant wavefunction(e.g.RHF, RKS)
    #     1: Unrestricted open - shell single - determinant wavefunction(e.g.UHF, UKS)
    #     2: Restricted open - shell single - determinant wavefunction(e.g.ROHF, ROKS)
    #     3: Restricted multiconfiguration wavefunction(e.g.RMP2, RCCSD)
    #     4: Unrestricted multiconfiguration wavefunction(e.g.UMP2, UCCSD)
    wfntype = inp['wfntype']
    if wfntype in [0, 2, 3]:
        restrictions = "restricted"
    elif wfntype in [1, 4]:
        restrictions = "unrestricted"
    else:
        raise IOError('Cannot determine if restricted or unrestricted wfntype wave function.')
    # MFWN provides number of alpha and beta electrons, this is a double check
    # mo_type (integer, scalar): Orbital type
    #     0: Alpha + Beta (i.e. spatial orbital)
    #     1: Alpha
    #     2: Beta
    # TODO calculate number of alpha and beta electrons manually.

    # Build the molecular orbitals
    mo = MolecularOrbitals(restrictions,
                           inp['nelec_a'],
                           inp['nelec_b'],
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
        'charge': inp['charge'],
        'obasis': obasis,
        'mo': mo,
        'nelec': inp['nelec_a'] + inp['nelec_b'],
        'energy': inp['energy'],
        'extra': mwfn_dict,
    }
