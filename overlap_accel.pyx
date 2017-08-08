# cython: profile=True
from libc.math cimport sqrt, pow, exp, abs
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void add_overlap(double coeff, double alpha0, double alpha1, double[::1] scales0,
                  double[::1] scales1, double[::1] r0, double[::1] r1,
                  long[:, ::1] iterpow0, long[:, ::1] iterpow1, double [:, ::1] result):
    # This entire function is in Cartesian coordinates
    cdef double gamma_inv = 1.0 / (alpha0 + alpha1)
    cdef double pre = coeff * exp(-alpha0 * alpha1 * gamma_inv * dist_sq(r0, r1))
    cdef double[::1] gpt_center = compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv)

    cdef long s0, s1
    cdef long[::1] n0, n1

    for s0, n0, in enumerate(iterpow0):
        for s1, n1, in enumerate(iterpow1):
            result[s0, s1] += pre * vec_gb_overlap_int1d(n0, n1, r0, r1, gpt_center,
                                                         gamma_inv) * scales0[s0] * scales1[s1]
            # print np.ravel_multi_index((s0, s1), result.shape), "|", result[s0, s1], "|", pre, \
            #     vec_gb_overlap_int1d(n0, n1, gpt_center - r0, gpt_center - r1, gamma_inv), \
            #     scales0[s0], scales1[s1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dist_sq(double[::1] r0, double[::1] r1) nogil:
    return pow((r0[0] - r1[0]), 2) \
           + pow((r0[1] - r1[1]), 2) \
           + pow((r0[2] - r1[2]), 2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1] compute_gpt_center(double alpha0, double[::1] r0, double alpha1, double[::1] r1, double gamma_inv):
    arr = np.zeros(3, dtype=np.float)
    cdef double[::1] gpt_center = arr
    gpt_center[0] = gamma_inv * (alpha0 * r0[0] + alpha1 * r1[0])
    gpt_center[1] = gamma_inv * (alpha0 * r0[1] + alpha1 * r1[1])
    gpt_center[2] = gamma_inv * (alpha0 * r0[2] + alpha1 * r1[2])
    return gpt_center


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double vec_gb_overlap_int1d(long[::1] n0, long[::1] n1, double[::1] r0, double[::1] r1, double[::1] gptc, double gamma_inv) nogil:
    return gb_overlap_int1d(n0[0], n1[0], gptc[0] - r0[0], gptc[0] - r1[0], gamma_inv) * \
           gb_overlap_int1d(n0[1], n1[1], gptc[1] - r0[1], gptc[1] - r1[1], gamma_inv) * \
           gb_overlap_int1d(n0[2], n1[2], gptc[2] - r0[2], gptc[2] - r1[2], gamma_inv)

cdef double gb_overlap_int1d(long n0, long n1, double pa, double pb, double gamma_inv) nogil:
    """The overlap integral in one dimension."""
    cdef long k, kmax
    cdef double result = 0.0

    kmax = (n0 + n1) / 2
    for k in range(kmax + 1):
        result += fac2(2 * k - 1) * gpt_coeff(2 * k, n0, n1, pa, pb) * pow(0.5 * gamma_inv, k)

    return sqrt(3.14159265358979323846 * gamma_inv) * result

cdef double gpt_coeff(long k, long n0, long n1, double pa, double pb) nogil:
    cdef double result = 0.0
    cdef long i0, i1
    i0 = k - n1
    if i0 < 0:
        i0 = 0
    i1 = k - i0

    while True:
        result += binom(n0, i0) * binom(n1, i1) * pow(pa, n0 - i0) * pow(pb, n1 - i1)
        i0 += 1
        i1 -= 1
        if not (i1 >= 0 and i0 <= n0):
            break
    return result

cpdef long binom(long n, long m) nogil:
    cdef long numer = 1
    cdef long denom = 1
    while n > m:
        numer *= n
        denom *= n - m
        n -= 1
    return numer / denom

cpdef long fac2(long n) nogil:
    cdef long result = 1
    while n > 1:
        result *= n
        n -= 2
    return result
