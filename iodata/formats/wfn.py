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
"""Gaussian/GAMESS-US WFN file format.

Only use this format if the program that generated it does not offer any alternatives that
HORTON can load. The WFN format has the disadvantage that it cannot represent contractions
and therefore expands all orbitals into a decontracted basis. This makes the
post-processing less efficient compared to formats that do support contractions of
Gaussian functions.
"""


from typing import Tuple, List

import numpy as np

from ..basis import MolecularBasis, Shell
from ..docstrings import document_load_one
from ..overlap import gob_cart_normalization
from ..orbitals import MolecularOrbitals
from ..periodic import sym2num
from ..utils import LineIterator


__all__ = []


PATTERNS = ['*.wfn']


# From the AIMALL documentation
# 1 S
# 2 PX
# 3 PY
# 4 PZ
# 5 DXX
# 6 DYY
# 7 DZZ
# 8 DXY
# 9 DXZ
# 10 DYZ
# 11 FXXX
# 12 FYYY
# 13 FZZZ
# 14 FXXY
# 15 FXXZ
# 16 FYYZ
# 17 FXYY
# 18 FXZZ
# 19 FYZZ
# 20 FXYZ
# 21 GXXXX
# 22 GYYYY
# 23 GZZZZ
# 24 GXXXY
# 25 GXXXZ
# 26 GXYYY
# 27 GYYYZ
# 28 GXZZZ
# 29 GYZZZ
# 30 GXXYY
# 31 GXXZZ
# 32 GYYZZ
# 33 GXXYZ
# 34 GXYYZ
# 35 GXYZZ
# 36 HZZZZZ (005)
# 37 HYZZZZ (014)
# 38 HYYZZZ (023)
# 39 HYYYZZ (032)
# 40 HYYYYZ (041)
# 41 HYYYYY (050)
# 42 HXZZZZ (104)
# 43 HXYZZZ (113)
# 44 HXYYZZ (122)
# 45 HXYYYZ (131)
# 46 HXYYYY (140)
# 47 HXXZZZ (203)
# 48 HXXYZZ (212)
# 49 HXXYYZ (221)
# 50 HXXYYY (230)
# 51 HXXXZZ (302)
# 52 HXXXYZ (311)
# 53 HXXXYY (320)
# 54 HXXXXZ (401)
# 55 HXXXXY (410)
# 56 HXXXXX (500)


CONVENTIONS = {
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyz', 'xyy', 'xzz', 'yzz', 'xyz'],
    (4, 'c'): ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz',
               'yzzz', 'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz'],
    (5, 'c'): ['zzzzz', 'yzzzz', 'yyzzz', 'yyyzz', 'yyyyz', 'yyyyy', 'xzzzz',
               'xyzzz', 'xyyzz', 'xyyyz', 'xyyyy', 'xxzzz', 'xxyzz', 'xxyyz',
               'xxyyy', 'xxxzz', 'xxxyz', 'xxxyy', 'xxxxz', 'xxxxy', 'xxxxx'],
}


# Definition of primitives in the WFN format. This is the order of the primitive
# types as documented by aimall, used in the field TYPE ASSIGNMENTS.
PRIMITIVE_NAMES = sum([CONVENTIONS[(angmom, 'c')] for angmom in range(6)], [])


def _load_helper_num(lit: LineIterator) -> List[int]:
    """Read number of orbitals, primitives and atoms."""
    line = next(lit)
    if not line.startswith('GAUSSIAN'):
        lit.error("Expecting line to start with GAUSSIAN.")
    return [int(i) for i in line.split() if i.isdigit()]


def _load_helper_atoms(lit: LineIterator, num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read the coordinates of the atoms."""
    atnums = np.empty(num_atoms, int)
    atcoords = np.empty((num_atoms, 3), float)
    for atom in range(num_atoms):
        words = next(lit).split()
        atnums[atom] = sym2num[words[0].title()]
        atcoords[atom, :] = [words[4], words[5], words[6]]
    return atnums, atcoords


def _load_helper_section(lit: LineIterator, nprim: int, start: str, skip: int,
                         dtype: np.dtype) -> np.ndarray:
    """Read CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS sections."""
    section = []
    while len(section) < nprim:
        line = next(lit)
        assert line.startswith(start)
        words = line.split()
        section.extend(words[skip:])
    assert len(section) == nprim
    return np.array([word.replace('D', 'E') for word in section]).astype(dtype)


def _load_helper_mo(lit: LineIterator, nprim: int) -> Tuple[str, str, str, np.ndarray]:
    """Read one section of MO information."""
    line = next(lit)
    assert line.startswith('MO')
    words = line.split()
    count = words[1]
    occ, energy = words[-5], words[-1]
    coeffs = _load_helper_section(lit, nprim, ' ', 0, float)
    return count, occ, energy, coeffs


def _load_helper_energy(lit: LineIterator) -> float:
    """Read energy."""
    line = next(lit).lower()
    while 'energy' not in line and line is not None:
        line = next(lit).lower()
    energy = float(line.split('energy =')[1].split()[0])
    return energy


def load_wfn_low(lit: LineIterator) -> Tuple:
    """Load data from a WFN file into arrays.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    """
    # read sections of wfn file
    title = next(lit).strip()
    num_mo, nprim, num_atoms = _load_helper_num(lit)
    atnums, atcoords = _load_helper_atoms(lit, num_atoms)
    # centers are indexed from zero in HORTON
    icenters = _load_helper_section(lit, nprim, 'CENTRE ASSIGNMENTS', 2, int) - 1
    # The type assignments are integer indices for individual basis functions,
    # while in IOData, only the order within shells is fixed by configurable
    # conventions. In principle, the wfn format makes it possible for two
    # shells with the same angular momentum to have a different ordering of
    # the basis functions.
    type_assignments = _load_helper_section(lit, nprim, 'TYPE ASSIGNMENTS', 2, int) - 1
    exponent = _load_helper_section(lit, nprim, 'EXPONENTS', 1, float)
    mo_count = np.empty(num_mo, int)
    mo_occ = np.empty(num_mo, float)
    mo_energy = np.empty(num_mo, float)
    mo_coefficients = np.empty([nprim, num_mo], float)
    for mo in range(num_mo):
        mo_count[mo], mo_occ[mo], mo_energy[mo], mo_coefficients[:, mo] = \
            _load_helper_mo(lit, nprim)
    energy = _load_helper_energy(lit)
    return title, atnums, atcoords, icenters, type_assignments, exponent, \
        mo_count, mo_occ, mo_energy, mo_coefficients, energy


# pylint: disable=too-many-branches
def build_obasis(icenters: np.ndarray, type_assignments: np.ndarray,
                 exponents: np.ndarray, lit: LineIterator) -> Tuple[MolecularBasis, np.ndarray]:
    """Construct a basis set using the arrays read from a WFN or WFX file.

    Parameters
    ----------
    icenters
        The center indices for all basis functions. shape=(nbasis,). Lowest
        index is zero.
    type_assignments
        Integer codes for basis function names. shape=(nbasis,). Lowest index
        is zero.
    exponents
        The Gaussian exponents of all basis functions. shape=(nbasis,)

    """
    # Build the basis set, keeping track of permutations in case there are
    # deviations from the default ordering of primitives in a WFN file.
    shells = []
    ibasis = 0
    nbasis = len(icenters)
    permutation = np.zeros(nbasis, dtype=int)
    # Loop over all (batches of primitive) basis functions and extract shells.
    while ibasis < nbasis:
        # Determine the angular moment of the shell
        type_assignment = type_assignments[ibasis]
        if type_assignment == 0:
            angmom = 0
        else:
            # multiple different type assignments (codes for individual basis
            # functions) can match one angular momentum.
            angmom = len(PRIMITIVE_NAMES[type_assignments[ibasis]])
        # The number of cartesian functions for the current angular momentum
        ncart = len(CONVENTIONS[(angmom, 'c')])
        # Determine how many shells are to be read in one batch. E.g. for a
        # contracted p shell, the WFN format contains first all px basis
        # functions, the all py, finally all pz. These need to be regrouped into
        # shells.
        # This pattern can almost be used to reverse-engineer contractions.
        # One should also check (i) if the corresponding mo-coefficients are the
        # same (after fixing them for normalization) and (ii) if the functions
        # are centered on the same atom.
        # For now, this implementation makes no attempt to reverse-engineer
        # contractions, but it can be done.
        ncon = 1  # the contraction length
        if angmom > 0:
            # batches for s-type functions are not necessary and may result in
            # multiple centers being pulled into one batch.
            while (ibasis + ncon < len(type_assignments)
                   and type_assignments[ibasis + ncon] == type_assignment):
                ncon += 1
        # Check if the type assignment is consistent for remaining basis
        # functions in this batch.
        for ifn in range(ncart):
            if not (type_assignments[ibasis + ncon * ifn: ibasis + ncon * (ifn + 1)]
                    == type_assignments[ibasis + ncon * ifn]).all():
                lit.error("Inconcsistent type assignments in current batch of shells.")
        # Check if all basis functions in the current batch sit on
        # the same center. If not, IOData cannot read this file.
        icenter = icenters[ibasis]
        if not (icenters[ibasis: ibasis + ncon * ncart] == icenter).all():
            lit.error("Incomplete shells in WFN file not supported by IOData.")
        # Check if the same exponent is used for corresponding basis functions.
        batch_exponents = exponents[ibasis: ibasis + ncon]
        for ifn in range(ncart):
            if not (exponents[ibasis + ncon * ifn: ibasis + ncon * (ifn + 1)]
                    == batch_exponents).all():
                lit.error("Exponents must be the same for corresponding basis functions.")
        # A permutation is needed because we need to regroup basis functions
        # into shells.
        batch_primitive_names = [
            PRIMITIVE_NAMES[type_assignments[ibasis + ifn * ncon]]
            for ifn in range(ncart)]
        for irep in range(ncon):
            for i, primitive_name in enumerate(batch_primitive_names):
                ifn = CONVENTIONS[(angmom, 'c')].index(primitive_name)
                permutation[ibasis + irep * ncart + ifn] = ibasis + irep + i * ncon
        # WFN uses non-normalized primitives, which will be corrected for
        # when processing the MO coefficients. Normalized primitives will
        # be used here. No attempt is made here to reconstruct the contraction.
        for exponent in batch_exponents:
            shells.append(Shell(icenter, [angmom], ['c'], np.array([exponent]),
                                np.array([[1.0]])))
        # Move on to the next contraction
        ibasis += ncart * ncon
    obasis = MolecularBasis(shells, CONVENTIONS, 'L2')
    assert obasis.nbasis == nbasis
    return obasis, permutation


def get_mocoeff_scales(obasis: MolecularBasis) -> np.ndarray:
    """Get the normalization of the un-normalized Cartesian basis functions.

    Parameters
    ----------
    obasis
        The molecular orbital basis.

    Returns
    -------
    scales
        Scaling factors to be multiplied into the molecular orbital
        coefficients.

    """
    scales = []
    for shell in obasis.shells:
        angmom = shell.angmoms[0]
        for name in obasis.conventions[(angmom, 'c')]:
            if name == '1':
                nx, ny, nz = 0, 0, 0
            else:
                nx = name.count('x')
                ny = name.count('y')
                nz = name.count('z')
            scales.append(gob_cart_normalization(shell.exponents[0], np.array([nx, ny, nz])))
    return np.array(scales)


@document_load_one("WFN", ['atcoords', 'atnums', 'energy', 'mo', 'obasis', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    (title, atnums, atcoords, icenters, type_assignments, exponents,
     mo_count, mo_occ, mo_energy, mo_coefficients, energy) = load_wfn_low(lit)
    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(icenters, type_assignments, exponents, lit)
    # Re-order the mo coefficients.
    mo_coefficients = mo_coefficients[permutation]
    # Fix normalization
    mo_coefficients /= get_mocoeff_scales(obasis).reshape(-1, 1)
    norb = mo_coefficients.shape[1]
    # make the wavefunction
    if mo_occ.max() > 1.0:
        # closed-shell system
        mo = MolecularOrbitals(
            'restricted', norb, norb,
            mo_occ, mo_coefficients, mo_energy, None)
    else:
        # open-shell system
        # counting the number of alpha orbitals
        norba = 1
        while (norba < mo_coefficients.shape[1]
               and mo_energy[norba] >= mo_energy[norba - 1]
               and mo_count[norba] == mo_count[norba - 1] + 1):
            norba += 1
        mo = MolecularOrbitals(
            'unrestricted', norba, norb - norba,
            mo_occ, mo_coefficients, mo_energy, None)

    result = {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'obasis': obasis,
        'mo': mo,
        'energy': energy,
    }
    return result
