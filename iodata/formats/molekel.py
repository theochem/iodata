# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --
"""Module for handling MOLEKEL file format."""


from typing import Dict, Tuple, List

import numpy as np

from .molden import _fix_molden_from_buggy_codes
from ..utils import angstrom, str_to_shell_types, shells_to_nbasis, LineIterator


__all__ = ['load']


patterns = ['*.mkl']


def _load_helper_char_mult(lit: LineIterator) -> List[int]:
    return [int(word) for word in next(lit).split()]


def _load_helper_coordinates(lit: LineIterator) -> Tuple[np.ndarray, np.ndarray]:
    numbers = []
    coordinates = []
    for line in lit:
        if line.strip() == '$END':
            break
        words = line.split()
        numbers.append(int(words[0]))
        coordinates.append([float(words[1]), float(words[2]), float(words[3])])
    numbers = np.array(numbers, int)
    coordinates = np.array(coordinates) * angstrom
    return numbers, coordinates


def _load_helper_obasis(lit: LineIterator, coordinates: np.ndarray) -> Tuple[Dict, int]:
    shell_types = []
    shell_map = []
    nprims = []
    alphas = []
    con_coeffs = []

    center_counter = 0
    in_shell = False
    nprim = None
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        if line == "":
            continue
        if line == '$$':
            center_counter += 1
            in_shell = False
        else:
            words = line.split()
            if len(words) == 2:
                assert in_shell
                alpha = float(words[0])
                alphas.append(alpha)
                con_coeffs.append(float(words[1]))
                nprim += 1
            else:
                if nprim is not None:
                    nprims.append(nprim)
                shell_map.append(center_counter)
                # always assume pure basis functions
                shell_type = str_to_shell_types(words[1], pure=True)[0]
                shell_types.append(shell_type)
                in_shell = True
                nprim = 0
    if nprim is not None:
        nprims.append(nprim)

    shell_map = np.array(shell_map)
    nprims = np.array(nprims)
    shell_types = np.array(shell_types)
    alphas = np.array(alphas)
    con_coeffs = np.array(con_coeffs)

    obasis = {"centers": coordinates, "shell_map": shell_map, "nprims": nprims,
              "shell_types": shell_types, "alphas": alphas, "con_coeffs": con_coeffs}

    nbasis = shells_to_nbasis(shell_types)
    return obasis, nbasis


def _load_helper_coeffs(lit: LineIterator, nbasis: int) -> np.ndarray:
    coeffs = []
    energies = []

    in_orb = 0
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        if in_orb == 0:
            # read a1g line
            words = line.split()
            ncol = len(words)
            assert ncol > 0
            for word in words:
                assert word == 'a1g'
            cols = [np.zeros((nbasis, 1), float) for icol in range(ncol)]
            in_orb = 1
        elif in_orb == 1:
            # read energies
            words = line.split()
            assert len(words) == ncol
            for word in words:
                energies.append(float(word))
            in_orb = 2
            ibasis = 0
        elif in_orb == 2:
            # read expansion coefficients
            words = line.split()
            assert len(words) == ncol
            for icol in range(ncol):
                cols[icol][ibasis] = float(words[icol])
            ibasis += 1
            if ibasis == nbasis:
                in_orb = 0
                coeffs.extend(cols)

    return np.hstack(coeffs), np.array(energies)


def _load_helper_occ(lit: LineIterator) -> np.ndarray:
    occs = []
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        for word in line.split():
            occs.append(float(word))
    return np.array(occs)


# pylint: disable=too-many-branches,too-many-statements
def load(lit: LineIterator) -> Dict:
    """Load data from a MOLEKEL file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``coordinates``, ``numbers``, ``obasis``, ``orb_alpha``,
        ``orb_alpha_coeffs``, ``orb_alpha_energies`` & ``orb_alpha_occs`` keys and their
        corresponding values. It may also contain ``orb_beta``, ``orb_beta_coeffs``,
        ``orb_beta_energies`` & ``orb_beta_occs`` keys and their values.

    """
    charge = None
    numbers = None
    coordinates = None
    obasis = None
    coeff_alpha = None
    ener_alpha = None
    occ_alpha = None
    coeff_beta = None
    ener_beta = None
    occ_beta = None
    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            # There is no file-end marker we can use, so we only stop when
            # reaching the end of the file.
            break
        if line == '$CHAR_MULT':
            charge, _spinmult = _load_helper_char_mult(lit)
        elif line == '$COORD':
            numbers, coordinates = _load_helper_coordinates(lit)
        elif line == '$BASIS':
            obasis, nbasis = _load_helper_obasis(lit, coordinates)
        elif line == '$COEFF_ALPHA':
            coeff_alpha, ener_alpha = _load_helper_coeffs(lit, nbasis)
        elif line == '$OCC_ALPHA':
            occ_alpha = _load_helper_occ(lit)
        elif line == '$COEFF_BETA':
            coeff_beta, ener_beta = _load_helper_coeffs(lit, nbasis)
        elif line == '$OCC_BETA':
            occ_beta = _load_helper_occ(lit)

    if charge is None:
        lit.error('Charge and multiplicity not found.')
    if coordinates is None:
        lit.error('Coordinates not found.')
    if obasis is None:
        lit.error('Orbital basis not found.')
    if coeff_alpha is None:
        lit.error('Alpha orbitals not found.')
    if occ_alpha is None:
        lit.error('Alpha occupation numbers not found.')

    nelec = numbers.sum() - charge
    if coeff_beta is None:
        assert nelec % 2 == 0
        assert abs(occ_alpha.sum() - nelec) < 1e-7
        orb_alpha = (nbasis, coeff_alpha.shape[1])
        orb_alpha_coeffs = coeff_alpha
        orb_alpha_energies = ener_alpha
        orb_alpha_occs = occ_alpha / 2
        orb_beta = None
    else:
        if occ_beta is None:
            lit.error('Beta occupation numbers not found in mkl file while '
                      'beta orbitals were present.')
        nalpha = int(np.round(occ_alpha.sum()))
        nbeta = int(np.round(occ_beta.sum()))
        assert nelec == nalpha + nbeta
        assert coeff_alpha.shape == coeff_beta.shape
        assert ener_alpha.shape == ener_beta.shape
        assert occ_alpha.shape == occ_beta.shape
        orb_alpha = (nbasis, coeff_alpha.shape[1])
        orb_alpha_coeffs = coeff_alpha
        orb_alpha_energies = ener_alpha
        orb_alpha_occs = occ_alpha
        orb_beta = (nbasis, coeff_beta.shape[1])
        orb_beta_coeffs = coeff_beta
        orb_beta_energies = ener_beta
        orb_beta_occs = occ_beta

    result = {
        'coordinates': coordinates,
        'orb_alpha': orb_alpha,
        'orb_alpha_coeffs': orb_alpha_coeffs,
        'orb_alpha_energies': orb_alpha_energies,
        'orb_alpha_occs': orb_alpha_occs,
        'numbers': numbers,
        'obasis': obasis,
    }
    if orb_beta is not None:
        result['orb_beta'] = orb_beta
        result['orb_beta_coeffs'] = orb_beta_coeffs
        result['orb_beta_energies'] = orb_beta_energies
        result['orb_beta_occs'] = orb_beta_occs
    _fix_molden_from_buggy_codes(result, lit.filename)
    return result
