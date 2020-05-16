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
"""Module for computing overlap of atomic orbital basis functions."""


import numpy as np
from scipy.special import binom, factorial2

from .overlap_python import compute_overlap_gaussian_3d
from .overlap_cartpure import tfs
from .basis import convert_conventions, iter_cart_alphabet, MolecularBasis
from .basis import HORTON2_CONVENTIONS as OVERLAP_CONVENTIONS


__all__ = ['OVERLAP_CONVENTIONS', 'compute_overlap_py', 'gob_cart_normalization']


def compute_overlap_py(obasis: MolecularBasis, atcoords: np.ndarray) -> np.ndarray:
    r"""Compute overlap matrix for the given molecular basis set.

    .. math::
        \braket{\psi_{i}}{\psi_{j}}

    This function takes into account the requested order of the basis functions
    in ``obasis.conventions``. Note that only L2 normalized primitives are
    supported at the moment.

    Parameters
    ----------
    obasis
        The orbital basis set.
    atcoords
        The atomic Cartesian coordinates (including those of ghost atoms).

    Returns
    -------
    overlap
        The matrix with overlap integrals, shape=(obasis.nbasis, obasis.nbasis).

    """
    if obasis.primitive_normalization != 'L2':
        raise ValueError('The overlap integrals are only implemented for L2 '
                         'normalization.')

    # Initialize result
    overlap = np.zeros((obasis.nbasis, obasis.nbasis))

    # Get a segmented basis, for simplicity
    obasis = obasis.get_segmented()

    # Compute the normalization constants of the primitives
    scales = [_compute_cart_shell_normalizations(shell) for shell in obasis.shells]

    # Loop over shell0
    begin0 = 0
    count = 0
    total = 0
    for i0, shell0 in enumerate(obasis.shells):
        r0 = atcoords[shell0.icenter]
        end0 = begin0 + shell0.nbasis

        # Loop over shell1 (lower triangular only, including diagonal)
        begin1 = 0
        for i1, shell1 in enumerate(obasis.shells[:i0 + 1]):
            r1 = atcoords[shell1.icenter]
            end1 = begin1 + shell1.nbasis

            # START of Cartesian coordinates. Shell types are positive
            result = np.zeros((len(scales[i0][0]), len(scales[i1][0])))

            a0 = np.min(shell0.exponents)
            a1 = np.min(shell1.exponents)
            p = np.exp(-a0 * a1 * np.linalg.norm(r0 - r1) ** 2 / (a0 + a1))
            total += 1
            if p > 1.e-15:
                count += 1
                # Loop over primitives in shell0 (Cartesian)
                for iexp0, (a0, cc0) in enumerate(zip(shell0.exponents, shell0.coeffs[:, 0])):
                    scales0 = scales[i0][iexp0]

                    # Loop over primitives in shell1 (Cartesian)
                    for iexp1, (a1, cc1) in enumerate(zip(shell1.exponents, shell1.coeffs[:, 0])):
                        scales1 = scales[i1][iexp1]

                        p = np.exp(-a0 * a1 * np.linalg.norm(r0 - r1) ** 2 / (a0 + a1))
                        if p < 1.0e-15:
                            continue

                        for s0, n0, in enumerate(iter_cart_alphabet(shell0.angmoms[0])):
                            for s1, n1, in enumerate(iter_cart_alphabet(shell1.angmoms[0])):
                                v = compute_overlap_gaussian_3d(r0, r1, a0, a1, n0, n1)
                                v *= cc0 * cc1 * scales0[s0] * scales1[s1]
                                result[s0, s1] += v
            # END of Cartesian coordinate system (if going to pure coordinates)

            # cart to pure
            if shell0.kinds[0] == 'p':
                result = np.dot(tfs[shell0.angmoms[0]], result)
            if shell1.kinds[0] == 'p':
                result = np.dot(result, tfs[shell1.angmoms[0]].T)

            # store lower triangular result
            overlap[begin0:end0, begin1:end1] = result
            # store upper triangular result
            overlap[begin1:end1, begin0:end0] = result.T

            begin1 = end1
        begin0 = end0

    permutation, signs = convert_conventions(obasis, OVERLAP_CONVENTIONS, reverse=True)
    overlap = overlap[permutation] * signs.reshape(-1, 1)
    overlap = overlap[:, permutation] * signs
    print("count = ", count, total)
    return overlap


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
            m = i + j
            if m % 2 == 0:
                integ = factorial2(i + j - 1, 2) / (2 * at) ** (m / 2)
                value += pf_i * binom(n2, j) * x2 ** (n2 - j) * integ
    value *= pf * np.sqrt(np.pi / at)
    return value


def _compute_cart_shell_normalizations(shell: 'Shell') -> np.ndarray:
    """Return normalization constants for the primitives in a given shell.

    Parameters
    ----------
    shell
        The shell for which the normalization constants must be computed.

    Returns
    -------
    np.ndarray
        The normalization constants, always for Cartesian functions, even when
        shell is pure.

    """
    shell = shell._replace(kinds=['c'] * shell.ncon)
    result = []
    for angmom in shell.angmoms:
        for exponent in shell.exponents:
            row = []
            for n in iter_cart_alphabet(angmom):
                row.append(gob_cart_normalization(exponent, n))
            result.append(np.array(row))
    return result


def gob_cart_normalization(alpha: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Compute normalization of exponent.

    Parameters
    ----------
    alpha
        Gaussian basis exponents
    n
        Cartesian subshell angular momenta

    Returns
    -------
    np.ndarray
        The normalization constant for the gaussian cartesian basis.

    """
    vfac2 = np.vectorize(factorial2)
    return np.sqrt((4 * alpha)**sum(n) * (2 * alpha / np.pi)**1.5
                   / np.prod(vfac2(2 * n - 1)))
