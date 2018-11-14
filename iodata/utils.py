# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
# pragma pylint: disable=wrong-import-order
"""Utility functions module."""


import numpy as np

from scipy.linalg import eigh
from typing import List, Dict, Tuple

from .overlap import get_shell_nbasis


__all__ = ['set_four_index_element']


angstrom = 1.0e-10 / 0.5291772083e-10
electronvolt = 1.602176462e-19 / 4.35974381e-18


def str_to_shell_types(s: str, pure: bool = False) -> List[int]:
    """Convert a string into a list of contraction types"""
    if pure:
        d = {'s': 0, 'p': 1, 'd': -2, 'f': -3, 'g': -4, 'h': -5, 'i': -6}
    else:
        d = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    return [d[c] for c in s.lower()]


def shell_type_to_str(shell_type: np.ndarray) -> Dict:
    """Convert a shell type into a character"""
    return {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i'}[abs(shell_type)]


def set_four_index_element(four_index_object: np.ndarray, i: int, j: int, k: int, l: int,
                           value: float):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation

    Parameters
    ----------
    four_index_object
        The four-index object. It will be written to.
        shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
    i, j, k, l
        The indices to assign to.
    value
        The value of the matrix element to store.
    """
    four_index_object[i, j, k, l] = value
    four_index_object[j, i, l, k] = value
    four_index_object[k, j, i, l] = value
    four_index_object[i, l, k, j] = value
    four_index_object[k, l, i, j] = value
    four_index_object[l, k, j, i] = value
    four_index_object[j, k, l, i] = value
    four_index_object[l, i, j, k] = value


def shells_to_nbasis(shell_types: np.ndarray) -> int:
    nbasis_shell = [get_shell_nbasis(i) for i in shell_types]
    return sum(nbasis_shell)


def volume(rvecs: np.ndarray) -> float:
    """Calculates cell volume

    Parameters
    ----------
    rvecs
        a numpy matrix of shape (x,3) where x is in {1,2,3}
    """
    nvecs = rvecs.shape[0]
    if len(rvecs.shape) == 1 or nvecs == 1:
        return np.linalg.norm(rvecs)
    elif nvecs == 2:
        return np.linalg.norm(np.cross(rvecs[0], rvecs[1]))
    elif nvecs == 3:
        return np.linalg.det(rvecs)
    else:
        raise ValueError("Argument rvecs should be of shape (x, 3), where x is in {1, 2, 3}")


def derive_naturals(dm: np.ndarray, overlap: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Derive natural orbitals from a given density matrix and assign the result to self.

    Parameters
    ----------
    dm
        The density matrix.
        shape=(nbasis, nbasis)
    overlap
        The overlap matrix
        shape=(nbasis, nbasis)

    Returns
    -------
    coeffs
        Orbital coefficients
        shape=(nbasis, nfn)
    occs
        Orbital occupations
        shape=(nfn, )
    energies
        Orbital energies
        shape=(nfn, )
    """
    # Transform density matrix to Fock-like form
    sds = np.dot(overlap.T, np.dot(dm, overlap))
    # Diagonalize and compute eigenvalues
    evals, evecs = eigh(sds, overlap)
    coeffs = np.zeros_like(overlap)
    coeffs = evecs[:, :coeffs.shape[1]]
    occs = evals
    energies = np.zeros(overlap.shape[0])
    return coeffs, occs, energies


def check_dm(dm: np.ndarray, overlap: np.ndarray, eps: float = 1e-4, occ_max: float = 1.0):
    """Check if the density matrix has eigenvalues in the proper range.

    Parameters
    ----------
    dm
        The density matrix
        shape=(nbasis, nbasis), dtype=float
    overlap
        The overlap matrix
        shape=(nbasis, nbasis), dtype=float
    eps
        The threshold on the eigenvalue inequalities.
    occ_max
        The maximum occupation.

    Raises
    ------
    ValueError
        When the density matrix has wrong eigenvalues.
    """
    # construct natural orbitals
    coeffs, occupations, energies = derive_naturals(dm, overlap)
    if occupations.min() < -eps:
        raise ValueError('The density matrix has eigenvalues considerably smaller than '
                         'zero. error=%e' % (occupations.min()))
    if occupations.max() > occ_max + eps:
        raise ValueError('The density matrix has eigenvalues considerably larger than '
                         'max. error=%e' % (occupations.max() - 1))


def compute_1dm(coeffs: np.ndarray, occs: np.ndarray) -> np.ndarray:
    r"""Compute first-order reduced density matrix (1DM).

    .. math::

    Parameters
    ----------
    coeffs
        Coefficients of spin orbitals.
    occs
        Occupations of spin orbitals.

    Returns
    -------
    out : array_like
        First-order reduced density matrix (1DM).
    """
    return np.dot(coeffs * occs, coeffs.T)


def compute_2dm_slater(coeffs: np.ndarray, occs: np.ndarray) -> np.ndarray:
    r"""Compute second-order reduced density matrix (2DM) of a Slater determinant wave-function."""
    raise NotImplementedError
