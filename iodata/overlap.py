"""
Computes the overlap integral. Used for calculating normalization in molden and wfn files.
"""
from typing import List, Tuple

import numpy as np

from .overlap_accel import add_overlap
from .overlap_helper import tfs, iter_pow


def compute_overlap(centers: np.ndarray, shell_map: np.ndarray, nprims: np.ndarray,
                    shell_types: np.ndarray, alphas: np.ndarray,
                    con_coeffs: np.ndarray) -> np.ndarray:
    r"""
    Computes overlap matrix. Follows same parameter convention as horton.GOBasis

    .. math::
        \braket{\psi_{i}}{\psi_{j}}

    Parameters
    ----------
    centers
        A numpy array with centers for the basis functions.
        shape = (ncenter, 3)
    shell_map
        An array with the center index for each shell.
        shape = (nshell,)
    nprims
        The number of primitives in each shell.
        shape = (nshell,)
    shell_types
        An array with contraction types: 0 = S, 1 = P, 2 = Cartesian D,
        3 = Cartesian F, ..., -2 = pure D, -3 = pure F, ...
        shape = (nshell,)
    alphas
        The exponents of the primitives in one shell.
        shape = (sum(nprims),)
    con_coeffs
        The contraction coefficients of the primitives for each
        contraction in a contiguous array. The coefficients are ordered
        according to the shells. Within each shell, the coefficients are
        grouped per exponent.
        shape = (sum(nprims),)

    Returns
    -------
    np.ndarray
        The overlap integral

    """

    # Initialize helper variables
    nshell = len(shell_types)
    nbasis = sum([get_shell_nbasis(i) for i in shell_types])

    shell_offsets = _get_shell_offsets(shell_types, nbasis)
    scales, scales_offsets = init_scales(alphas, nprims, shell_types)

    # Initialize result
    integral = np.zeros((nbasis, nbasis))

    # Reorganize arrays in pythonic manner
    alphas_split = _split_data_by_prims(alphas, nprims)
    con_coeffs_split = _split_data_by_prims(con_coeffs, nprims)
    scales_offsets_split = _split_data_by_prims(scales_offsets, nprims)

    # Loop over shell0
    for big_tuple0 in zip(list(range(nshell)), shell_map, shell_types, shell_offsets,
                          shell_offsets[1:],
                          alphas_split, con_coeffs_split, scales_offsets_split):

        ishell0, center0, shell_type0, start0, \
        end0, alphas0, con_coeffs0, scales_offsets0 = big_tuple0

        r0 = centers[center0, :]

        # Loop over shell1 (lower triangular only)
        for big_tuple1 in zip(shell_map[:ishell0 + 1], shell_types[:ishell0 + 1], shell_offsets,
                              shell_offsets[1:], alphas_split[:ishell0 + 1],
                              con_coeffs_split[:ishell0 + 1], scales_offsets_split[:ishell0 + 1]):

            center1, shell_type1, start1, end1, alphas1, con_coeffs1, scales_offsets1 = big_tuple1

            r1 = centers[center1, :]

            # START of Cartesian coordinates. Shell types are positive
            result = np.zeros((get_shell_nbasis(abs(shell_type0)),
                               get_shell_nbasis(abs(shell_type1))))
            # print "shell type", shell_type0, shell_type1
            # Loop over primitives in shell0 (Cartesian)
            for a0, so0, cc0 in zip(alphas0, scales_offsets0, con_coeffs0):
                s0 = scales[so0:]

                # Loop over primitives in shell1 (Cartesian)
                for a1, so1, cc1 in zip(alphas1, scales_offsets1, con_coeffs1):
                    s1 = scales[so1:]

                    add_overlap(cc0 * cc1, a0, a1, s0, s1, r0, r1, iter_pow[abs(shell_type0)],
                                iter_pow[abs(shell_type1)], result)
                    # print result

            # END of Cartesian coordinate system (if going to pure coordinates)

            # cart to pure
            if shell_type0 < -1:
                result = np.dot(tfs[abs(shell_type0)], result)
            if shell_type1 < -1:
                result = np.dot(result, tfs[abs(shell_type1)].T)

            # store lower triangular result
            integral[start0:end0, start1:end1] = result

            # store upper triangular result
            integral[start1:end1, start0:end0] = result.T
    return integral


def _split_data_by_prims(x: np.ndarray, nprims: np.ndarray) -> List[np.ndarray]:
    """Returns nested lists according to the number of primitives per shell"""
    nprims = np.insert(nprims, 0, 0)
    nprims = np.cumsum(nprims)
    return [x[s:e] for s, e in zip(nprims, nprims[1:])]


def _get_shell_offsets(shell_types, nbasis):
    """Calculates index offset for shells"""
    shell_offsets = []
    last = 0
    for i in shell_types:
        shell_offsets.append(last)
        last += get_shell_nbasis(i)
    shell_offsets.append(nbasis)
    return shell_offsets


def init_scales(alphas: np.ndarray, nprims: np.ndarray, shell_types: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Returns normalization constants and offsets per shell

    Parameters
    ----------
    alphas
        Gaussian basis exponents
    nprims
        Number of primitives in each shell
    shell_types
        The angular momentum of each shell

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The normalization factors for each shell

    """
    counter, oprim = 0, 0
    nscales = sum([get_shell_nbasis(abs(s)) * p for s, p in zip(shell_types, nprims)])
    scales = np.zeros(nscales)
    scales_offsets = np.zeros(sum(nprims), dtype=int)

    for s in range(len(shell_types)):
        for p in range(nprims[s]):
            scales_offsets[oprim + p] = counter
            alpha = alphas[oprim + p]
            for n in _get_iter_pow(abs(shell_types[s])):
                scales[counter] = gob_cart_normalization(alpha, n)
                counter += 1
        oprim += nprims[s]

    return scales, scales_offsets


def gob_cart_normalization(alpha: np.ndarray, n: np.ndarray) -> np.ndarray:  # from utils
    """
    Check normalization of exponents

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
    vfac2 = np.vectorize(_fac2_slow)
    return np.sqrt((4 * alpha) ** sum(n) * (2 * alpha / np.pi) ** 1.5 / np.prod(vfac2(2 * n - 1)))


def get_shell_nbasis(shell: int) -> int:
    """
    Returns number of basis functions within a shell.
    Negative shell numbers refer to pure functions.

    Parameters
    ----------
    shell
        Angular momentum quantum number

    Returns
    -------
    int
        The number of basis functions in the shell

    """
    if shell > 0:  # Cartesian
        return int((shell + 1) * (shell + 2) / 2)
    elif shell == -1:
        raise ValueError
    else:  # Pure
        return -2 * shell + 1


def _fac2_slow(n: int) -> int:
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


def _get_iter_pow(n: int) -> np.ndarray:
    """Gives the ordering within shells.
    See http://theochem.github.io/horton/2.1.0b1/tech_ref_gaussian_basis.html for details."""
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)
