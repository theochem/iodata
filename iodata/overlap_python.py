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


import numpy as np

from typing import Tuple
from scipy.special import binom, factorialk

from .overlap_helper import tfs


def compute_overlap_matrix(centers: np.ndarray, shell_index: np.ndarray, shell_type: np.ndarray,
                           shell_nprim: np.ndarray, prim_alpha: np.ndarray,
                           prim_coeff: np.ndarray) -> np.ndarray:
    r"""Compute the overlap matrix of atomic basis functions (i.e. atomic orbitals).

    Parameters
    ----------
    centers
        Coordinates of centers on which Gaussian basis functions are located.
    shell_index
        Index of center on which basis shells are located.
    shell_nprim
        Number of primitive basis functions of basis functions in each shell.
    shell_type
        Angular momentum quantum number of basis shell.
    prim_alpha
        Exponents of primitive Gaussian basis functions.
    prim_coeff
        Coefficients of primitive Gaussian basis functions.

    Returns
    -------
    out : array_like
        The overlap matrix of atomic basis functions (i.e. atomic orbital overlap matrix).

    """
    # compute number of basis functions
    basis_count = np.array([count_basis(l) for l in shell_type])
    nbasis = np.sum(basis_count)
    basis_offset = np.cumsum(np.insert(basis_count, 0, 0), dtype=int)
    # compute normalization of primitive basis functions
    con_norms, con_index = compute_norm_contracted_basis(prim_alpha, shell_nprim, shell_type)
    # compute offsets
    offset = np.insert(shell_nprim, 0, 0)
    offset = np.cumsum(offset)
    # compute overlap matrix elements
    matrix = np.zeros((nbasis, nbasis))
    for i in range(len(shell_type)):
        # basis function center, type, starting & ending offset of basis i
        bi_r = centers[shell_index[i], :]
        bi_l = shell_type[i]
        bi_s = basis_offset[i]
        bi_e = basis_offset[i + 1]
        # primitive exponents, coefficients & offsets of basis i
        pi_a = prim_alpha[offset[i]: offset[i + 1]]
        pi_c = prim_coeff[offset[i]: offset[i + 1]]
        pi_o = con_index[offset[i]: offset[i + 1]]
        # loop over lower triangle
        for j in range(0, i + 1):
            bj_r = centers[shell_index[j], :]
            bj_l = shell_type[j]
            bj_s = basis_offset[j]
            bj_e = basis_offset[j + 1]
            pj_a = prim_alpha[offset[j]: offset[j + 1]]
            pj_c = prim_coeff[offset[j]: offset[j + 1]]
            pj_o = con_index[offset[j]: offset[j + 1]]
            # block of overlap matrix
            result = np.zeros((count_basis(abs(bi_l)), count_basis(abs(bj_l))))
            for ai, oi, ci in zip(pi_a, pi_o, pi_c):
                ni = con_norms[oi:]
                for aj, oj, cj in zip(pj_a, pj_o, pj_c):
                    nj = con_norms[oj:]
                    for con_i, con_n_i in enumerate(get_iter_pow(abs(bi_l))):
                        for con_j, con_n_j in enumerate(get_iter_pow(abs(bj_l))):
                            v = compute_overlap_gaussian_3d(bi_r, bj_r, ai, aj, con_n_i, con_n_j)
                            v *= ni[con_i] * nj[con_j] * ci * cj
                            result[con_i, con_j] += v
            # cart to pure
            if bi_l < -1:
                result = np.dot(tfs[abs(bi_l)], result)
            if bj_l < -1:
                result = np.dot(result, tfs[abs(bj_l)].T)
            matrix[bi_s:bi_e, bj_s:bj_e] = result
            matrix[bj_s:bj_e, bi_s:bi_e] = result.T
    return matrix


def compute_overlap_gaussian_3d(r1, r2, a1, a2, n1, n2):
    """Compute overlap integral of two Gaussian functions in three-dimensions."""
    value = compute_overlap_gaussian_1d(r1[0], r2[0], a1, a2, n1[0], n2[0])
    value *= compute_overlap_gaussian_1d(r1[1], r2[1], a1, a2, n1[1], n2[1])
    value *= compute_overlap_gaussian_1d(r1[2], r2[2], a1, a2, n1[2], n2[2])
    return value


def compute_overlap_gaussian_1d(x1, x2, a1, a2, n1, n2):
    """Compute overlap integral of two Gaussian functions in one-dimensions."""
    # compute total exponent and new x
    at = a1 + a2
    xn = (a1 * x1 + a2 * x2) / at
    pf = np.exp(-a1 * a2 * (x1 - x2) ** 2 / at)
    x1 = xn - x1
    x2 = xn - x2
    # compute overlap
    value = 0
    for i in range(n1 + 1):
        pf_i = binom(n1, i) * x1 ** (n1 - i)
        for j in range(n2 + 1):
            if (i + j) % 2 == 0:
                integ = factorialk(i + j - 1, 2) / (2 * at) ** ((i + j) / 2)
                value += pf_i * binom(n2, j) * x2 ** (n2 - j) * integ
    value *= pf * np.sqrt(np.pi / at)
    return value


def compute_norm_contracted_basis(prim_alphas: np.ndarray, shell_nprim: np.ndarray,
                                  shell_type: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalization constants and offsets contracted Gaussian basis functions.

    Parameters
    ----------
    prim_alphas
        Exponents of primitive cartesian Gaussian basis functions.
    shell_nprim
        Number of primitives in each basis shell.
    shell_type
        The angular momentum of each basis shell.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The normalization factors for each shell

    """
    con_basis_count = sum([count_basis(abs(l)) * p for l, p in zip(shell_type, shell_nprim)])
    con_basis_norms = np.zeros(con_basis_count)
    con_basis_index = np.zeros(sum(shell_nprim), dtype=int)

    counter, start = 0, 0
    vfac = np.vectorize(factorialk)
    # loop over basis functions
    for s, bt in enumerate(shell_type):
        # look over primitives
        for p in range(shell_nprim[s]):
            con_basis_index[start + p] = counter
            a = prim_alphas[start + p]
            for n in get_iter_pow(abs(bt)):
                # compute norm
                norm = np.sqrt((4 * a)**sum(n) * (2 * a / np.pi)**1.5 / np.prod(vfac(2 * n - 1, 2)))
                con_basis_norms[counter] = norm
                counter += 1
        start += shell_nprim[s]
    return con_basis_norms, con_basis_index


def count_basis(shell_type: int) -> int:
    r"""Return number of basis in a given shell.

    Parameters
    ----------
    shell_type
        Angular momentum quantum number, :math:`l`.

    Returns
    -------
    out : int
        The number of orbitals (i.e. magnetic quantum numbers :math:`m_l`).

    """
    if shell_type >= 0:
        # case of cartesian orbitals
        return (shell_type + 1) * (shell_type + 2) // 2
    elif shell_type == -1:
        raise ValueError("Argument shell={0} is not recognized.".format(shell_type))
    else:
        # case of pure orbitals
        return -2 * shell_type + 1


def get_iter_pow(n: int) -> np.ndarray:
    """Give (n_x, n_y, n_z) so that n_x + n_y + n_z = n."""
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)
