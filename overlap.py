

import numpy as np
from overlap_helper import tfs


def compute_overlap(centers, shell_map, nprims, shell_types, alphas, con_coeffs):
    """Computes overlap matrix. Follows same parameter convention as horton.GOBasis"""
    nshell = len(shell_types)
    nbasis = sum([get_shell_nbasis(i) for i in shell_types])

    shell_offsets = get_shell_offsets(shell_types, nbasis)

    integral = np.zeros((nbasis, nbasis))
    scales, scales_offsets = init_scales(alphas, nprims, shell_types)

    alphas_split = split_data_by_prims(alphas, nprims)
    con_coeffs_split = split_data_by_prims(con_coeffs, nprims)
    scales_offsets_split = split_data_by_prims(scales_offsets, nprims)

    for big_tuple0 in zip(range(nshell), shell_map, shell_types, shell_offsets, shell_offsets[1:],
                          alphas_split, con_coeffs_split, scales_offsets_split):

        ishell0, center0, shell_type0, start0, \
        end0, alphas0, con_coeffs0, scales_offsets0 = big_tuple0

        r0 = centers[center0, :]

        # loop over symmetric portion only
        for big_tuple1 in zip(shell_map[:ishell0 + 1], shell_types[:ishell0 + 1], shell_offsets,
                              shell_offsets[1:], alphas_split[:ishell0 + 1],
                              con_coeffs_split[:ishell0 + 1], scales_offsets_split[:ishell0 + 1]):

            center1, shell_type1, start1, end1, alphas1, con_coeffs1, scales_offsets1 = big_tuple1

            r1 = centers[center1, :]

            # START of Cartesian coordinates
            result = np.zeros((get_shell_nbasis(abs(shell_type0)),
                               get_shell_nbasis(abs(shell_type1))))
            for a0, so0, cc0 in zip(alphas0, scales_offsets0, con_coeffs0):
                s0 = scales[so0:]

                for a1, so1, cc1 in zip(alphas1, scales_offsets1, con_coeffs1):
                    s1 = scales[so1:]

                    cc = cc0 * cc1
                    add_overlap(cc, a0, a1, s0, s1, r0, r1,
                                shell_type0, shell_type1, result)
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


def split_data_by_prims(x, nprims):
    """Returns nested lists according to the number of primitives per shell"""
    nprims = np.insert(nprims, 0, 0)
    nprims = np.cumsum(nprims)
    return [x[s:e] for s, e in zip(nprims, nprims[1:])]


def get_shell_offsets(shell_types, nbasis):
    """Calculates index offset for shells"""
    shell_offsets = []
    last = 0
    for i in shell_types:
        shell_offsets.append(last)
        last += get_shell_nbasis(i)
    shell_offsets.append(nbasis)
    return shell_offsets


def init_scales(alphas, nprims, shell_types):
    """Returns normalization constants and offsets per shell"""
    counter, oprim = 0, 0
    nscales = sum([get_shell_nbasis(abs(s)) * p for s, p in zip(shell_types, nprims)])
    scales = np.zeros(nscales)
    scales_offsets = np.zeros(sum(nprims), dtype=int)

    for s in range(len(shell_types)):
        for p in range(nprims[s]):
            scales_offsets[oprim + p] = counter
            alpha = alphas[oprim + p]
            for n in get_iter_pow(abs(shell_types[s])):
                scales[counter] = gob_cart_normalization(alpha, n)
                counter += 1
        oprim += nprims[s]

    return scales, scales_offsets


def gob_cart_normalization(alpha, n):  # from utils
    vfac2 = np.vectorize(fac2)
    return np.sqrt((4 * alpha) ** sum(n) * (2 * alpha / np.pi) ** 1.5 / np.prod(vfac2(2 * n - 1)))


def add_overlap(coeff, alpha0, alpha1, scales0, scales1, r0, r1, shell_type0, shell_type1, result):
    """Calculates overlap integrals within a primitive basis function and adds to result"""
    # This entire function is in Cartesian coordinates
    gamma_inv = 1.0 / (alpha0 + alpha1)
    pre = coeff * np.exp(-alpha0 * alpha1 * gamma_inv * dist_sq(r0, r1))
    gpt_center = compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv)

    nshell0 = get_shell_nbasis(abs(shell_type0))  # In cartesian coordinates
    nshell1 = get_shell_nbasis(abs(shell_type1))  # In cartesian coordinates
    for n0, s0 in zip(get_iter_pow(abs(shell_type0)), range(nshell0)):
        for n1, s1 in zip(get_iter_pow(abs(shell_type1)), range(nshell1)):
            result[s0, s1] += pre * np.prod(vec_gb_overlap_int1d(n0, n1,
                                                                 gpt_center - r0,
                                                                 gpt_center - r1,
                                                                 gamma_inv)) \
                              * scales0[s0] * scales1[s1]
            # print np.ravel_multi_index((s0, s1), result.shape), "|", result[s0, s1], "|", pre, \
            #     vec_gb_overlap_int1d(n0, n1, gpt_center - r0, gpt_center - r1, gamma_inv), \
            #     scales0[s0], scales1[s1]
    return result


def get_shell_nbasis(shell):
    """Returns number of basis functions within a shell.
    Negative shell numbers refer to pure functions."""
    if shell > 0:  # Cartesian
        return (shell + 1) * (shell + 2) / 2
    elif shell == -1:
        raise ValueError
    else:  # Pure
        return -2 * shell + 1


def dist_sq(r0, r1):
    """The square of the distance between two coordinates"""
    return sum((r0 - r1) ** 2)


def compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv):
    return gamma_inv * (alpha0 * r0 + alpha1 * r1)


def gb_overlap_int1d(n0, n1, pa, pb, gamma_inv):
    """The overlap integral in one dimension. This function will be Numpy vectorized."""
    kmax = (n0 + n1) / 2
    result = 0.0
    for k in range(kmax + 1):
        result += fac2(2 * k - 1) * gpt_coeff(2 * k, n0, n1, pa, pb) * np.power(0.5 * gamma_inv, k)

    return np.sqrt(np.pi * gamma_inv) * result


vec_gb_overlap_int1d = np.vectorize(gb_overlap_int1d)


def fac2(n):
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


def gpt_coeff(k, n0, n1, pa, pb):
    result = 0.0
    i0 = k - n1
    if i0 < 0:
        i0 = 0
    i1 = k - i0

    while True:
        result += binom(n0, i0) * binom(n1, i1) * np.power(pa, n0 - i0) * np.power(pb, n1 - i1)
        i0 += 1
        i1 -= 1
        if not (i1 >= 0 and i0 <= n0):
            break
    return result


def binom(n, m):
    numer, denom = 1, 1
    while n > m:
        numer *= n
        denom *= n - m
        n -= 1
    return numer / denom


def get_iter_pow(n):
    """Gives the ordering within shells.
    See http://theochem.github.io/horton/2.1.0b1/tech_ref_gaussian_basis.html for details."""
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)
