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

from typing import Optional

import attr
import numpy as np
from scipy.special import binom, factorial2

from .overlap_cartpure import tfs
from .basis import convert_conventions, iter_cart_alphabet, MolecularBasis
from .basis import HORTON2_CONVENTIONS as OVERLAP_CONVENTIONS

__all__ = ['OVERLAP_CONVENTIONS', 'compute_overlap', 'gob_cart_normalization']


# pylint: disable=too-many-nested-blocks,too-many-statements,too-many-branches
def compute_overlap(obasis0: MolecularBasis, atcoords0: np.ndarray,
                    obasis1: Optional[MolecularBasis] = None,
                    atcoords1: Optional[np.ndarray] = None,) -> np.ndarray:
    r"""Compute overlap matrix for the given molecular basis set(s).

    .. math::
        \braket{\psi_{i}}{\psi_{j}}

    When only one basis set is given, the overlap matrix of that basis (with
    itself) is computed. If a second basis set (with its atomic coordinates) is
    provided, the overlap between the two basis sets is computed.

    This function takes into account the requested order of the basis functions
    in ``obasis0.conventions`` (and ``obasis1.conventions``). Note that only L2
    normalized primitives are supported at the moment.

    Parameters
    ----------
    obasis0
        The orbital basis set.
    atcoords0
        The atomic Cartesian coordinates (including those of ghost atoms).
    obasis1
        An optional second orbital basis set.
    atcoords1
        An optional second array with atomic Cartesian coordinates
        (including those of ghost atoms).

    Returns
    -------
    overlap
        The matrix with overlap integrals, ``shape=(obasis0.nbasis, obasis1.nbasis)``.

    """
    if obasis0.primitive_normalization != 'L2':
        raise ValueError('The overlap integrals are only implemented for L2 '
                         'normalization.')

    # Get a segmented basis, for simplicity
    obasis0 = obasis0.get_segmented()

    # Handle optional arguments
    if obasis1 is None:
        if atcoords1 is not None:
            raise TypeError("When no second basis is given, no second second "
                            "array of atomic coordinates is expected.")
        obasis1 = obasis0
        atcoords1 = atcoords0
        identical = True
    else:
        if obasis1.primitive_normalization != 'L2':
            raise ValueError('The overlap integrals are only implemented for L2 '
                             'normalization.')
        if atcoords1 is None:
            raise TypeError("When a second basis is given, a second second "
                            "array of atomic coordinates is expected.")
        # Get a segmented basis, for simplicity
        obasis1 = obasis1.get_segmented()
        identical = False

    # Initialize result
    overlap = np.zeros((obasis0.nbasis, obasis1.nbasis))

    # Compute the normalization constants of the Cartesian primitives, with the
    # contraction coefficients multiplied in.
    scales0 = [_compute_cart_shell_normalizations(shell) * shell.coeffs
               for shell in obasis0.shells]
    if identical:
        scales1 = scales0
    else:
        scales1 = [_compute_cart_shell_normalizations(shell) * shell.coeffs
                   for shell in obasis1.shells]

    n_max = max(np.max(shell.angmoms) for shell in obasis0.shells)
    if not identical:
        n_max = max(n_max, max(np.max(shell.angmoms) for shell in obasis1.shells))
    go = GaussianOverlap(n_max)

    # define a python ufunc (numpy function) for broadcasted calling over angular momentums
    compute_overlap_1d = np.frompyfunc(go.compute_overlap_gaussian_1d, 5, 1)

    # Loop over shell0
    begin0 = 0

    # pylint: disable=too-many-nested-blocks
    for i0, shell0 in enumerate(obasis0.shells):
        r0 = atcoords0[shell0.icenter]
        end0 = begin0 + shell0.nbasis

        # Loop over shell1 (lower triangular only, including diagonal)
        begin1 = 0
        if identical:
            nshell1 = i0 + 1
        else:
            nshell1 = len(obasis1.shells)
        for i1, shell1 in enumerate(obasis1.shells[:nshell1]):
            r1 = atcoords1[shell1.icenter]
            end1 = begin1 + shell1.nbasis

            # prepare some constants to save FLOPS later on
            rij = r0 - r1
            rij_norm_sq = np.dot(rij, rij)

            # Check if the result is going to significant.
            a0_min = np.min(shell0.exponents)
            a1_min = np.min(shell1.exponents)
            prefactor_max = np.exp(-a0_min * a1_min * rij_norm_sq / (a0_min + a1_min))
            if prefactor_max > 1e-15:
                # START of Cartesian coordinates. Shell types are positive

                # arrays of angular momentums [[2, 0, 0], [0, 2, 0], ..., [0, 1, 1]]
                n0 = np.array(list(iter_cart_alphabet(shell0.angmoms[0])))
                n1 = np.array(list(iter_cart_alphabet(shell1.angmoms[0])))
                shell_overlap = np.zeros((n0.shape[0], n1.shape[0]))

                # Loop over primitives in shell0 (Cartesian)
                for shell_scales0, a0 in zip(scales0[i0], shell0.exponents):
                    a0_r0 = a0 * r0

                    # Loop over primitives in shell1 (Cartesian)
                    for shell_scales1, a1 in zip(scales1[i1], shell1.exponents):
                        at = a0 + a1
                        prefactor = np.exp(-a0 * a1 / at * rij_norm_sq)
                        if prefactor < 1e-15:
                            continue
                        # prepare some pre-factors to save FLOPS in inner loop
                        two_at = 2 * at
                        prefactor *= (np.pi / at) ** (3 / 2)
                        rn = (a0_r0 + a1 * r1) / at
                        rn_0 = rn - r0
                        rn_1 = rn - r1

                        # Note that frompyfunc-ed functions return arrays with
                        # dtype=object. This is converted back to floats as
                        # early as possible to improve performance of subsequent
                        # array operations.
                        vs = compute_overlap_1d(rn_0, rn_1, n0[:, None, :], n1[None, :, :], two_at)
                        v = np.prod(vs.astype(float), axis=2)
                        v *= prefactor
                        v *= shell_scales0[:, None]
                        v *= shell_scales1[None, :]
                        shell_overlap += v

                # END of Cartesian coordinate system (if going to pure coordinates)

                # cart to pure
                if shell0.kinds[0] == 'p':
                    shell_overlap = np.dot(tfs[shell0.angmoms[0]], shell_overlap)
                if shell1.kinds[0] == 'p':
                    shell_overlap = np.dot(shell_overlap, tfs[shell1.angmoms[0]].T)

                # store lower triangular result
                overlap[begin0:end0, begin1:end1] = shell_overlap
                if identical:
                    # store upper triangular result
                    overlap[begin1:end1, begin0:end0] = shell_overlap.T

            begin1 = end1
        begin0 = end0

    permutation0, signs0 = convert_conventions(obasis0, OVERLAP_CONVENTIONS, reverse=True)
    overlap = overlap[permutation0] * signs0.reshape(-1, 1)
    if identical:
        permutation1, signs1 = permutation0, signs0
    else:
        permutation1, signs1 = convert_conventions(obasis1, OVERLAP_CONVENTIONS, reverse=True)
    overlap = overlap[:, permutation1] * signs1
    return overlap


class GaussianOverlap:
    """Gaussian Overlap Class."""

    def __init__(self, n_max):
        """Initialize class.

        Parameters
        ----------
        n_max : int
            Maximum angular momentum.

        """
        self.binomials = [[binom(n, i) for i in range(n + 1)] for n in range(n_max + 1)]
        facts = [factorial2(m, 2) for m in range(2 * n_max)]
        facts.insert(0, 1)
        self.facts = np.array(facts)

    def compute_overlap_gaussian_1d(self, x1, x2, n1, n2, two_at):
        """Compute overlap integral of two Gaussian functions in one-dimensions."""
        # compute overlap
        value = 0
        for i in range(n1 + 1):
            pf_i = self.binomials[n1][i] * x1 ** (n1 - i)
            for j in range(i % 2, n2 + 1, 2):
                m = i + j
                integ = self.facts[m] / two_at ** (m / 2)   # TODO // 2
                value += pf_i * self.binomials[n2][j] * x2 ** (n2 - j) * integ
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
    shell = attr.evolve(shell, kinds=['c'] * shell.ncon)
    result = []
    for angmom in shell.angmoms:
        for exponent in shell.exponents:
            row = []
            for n in iter_cart_alphabet(angmom):
                row.append(gob_cart_normalization(exponent, n))
            result.append(row)
    return np.array(result)


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
    return np.sqrt((4 * alpha) ** sum(n) * (2 * alpha / np.pi) ** 1.5
                   / np.prod(factorial2(2 * n - 1)))
