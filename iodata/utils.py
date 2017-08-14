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

import numpy as np
from scipy.linalg import eigh
from .overlap import get_shell_nbasis

__all__ = ['set_four_index_element']

angstrom = 1.0e-10 / 0.5291772083e-10
electronvolt = 1.602176462e-19 / 4.35974381e-18


def str_to_shell_types(s, pure=False):
    """Convert a string into a list of contraction types"""
    if pure:
        d = {'s': 0, 'p': 1, 'd': -2, 'f': -3, 'g': -4, 'h': -5, 'i': -6}
    else:
        d = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    return [d[c] for c in s.lower()]


def shell_type_to_str(shell_type):
    """Convert a shell type into a character"""
    return {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i'}[abs(shell_type)]


def set_four_index_element(four_index_object, i, j, k, l, value):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation

    Parameters
    ----------
    four_index_object : np.ndarray, shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
        The four-index object
    i, j, k, l: int
        The indices to assign to.
    value : float
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


def shells_to_nbasis(shell_types):
    nbasis_shell = [get_shell_nbasis(i) for i in shell_types]
    return sum(nbasis_shell)


def volume(rvecs):
    """Takes a numpy matrix of shape (x,3) where x is in {1,2,3}"""
    nvecs = rvecs.shape[0]
    if len(rvecs.shape) == 1 or nvecs == 1:
        return np.linalg.norm(rvecs)
    elif nvecs == 2:
        return np.linalg.norm(np.cross(rvecs[0], rvecs[1]))
    elif nvecs == 3:
        return np.linalg.det(rvecs)
    else:
        print("1: Expected rvecs to be of shape (x,3), where x is in {1,2,3}")
        raise ValueError


def derive_naturals(dm, overlap):
    """Derive natural orbitals from a given density matrix and assign the result to self.

    Parameters
    ----------
    dm : np.ndarray, shape=(nbasis, nbasis)
        The density matrix.
    overlap : np.ndarray, shape=(nbasis, nbasis)
        The overlap matrix

    Returns
    -------
    coeffs : np.ndarray, shape=(nbasis, nfn)
        Orbital coefficients
    occs : np.ndarray, shape=(nfn, )
        Orbital occupations
    energies : np.ndarray, shape=(nfn, )
        Orbital energies
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


def check_dm(dm, overlap, eps=1e-4, occ_max=1.0):
    """Check if the density matrix has eigenvalues in the proper range.

    Parameters
    ----------
    dm : np.ndarray, shape=(nbasis, nbasis), dtype=float
        The density matrix
    overlap : np.ndarray, shape=(nbasis, nbasis), dtype=float
        The overlap matrix
    eps : float
        The threshold on the eigenvalue inequalities.
    occ_max : float
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
