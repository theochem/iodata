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
"""Test iodata.overlap & iodata.overlap_accel modules."""

import attr
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pytest import raises

from ..api import load_one
from ..basis import MolecularBasis, Shell
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS, convert_vector_basis
from ..overlap_cartpure import tfs

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_normalization_basics_segmented():
    for angmom in range(7):
        shells = [Shell(0, [angmom], ['c'], np.array([0.23]), np.array([[1.0]]))]
        if angmom >= 2:
            shells.append(Shell(0, [angmom], ['p'], np.array([0.23]), np.array([[1.0]])))
        obasis = MolecularBasis(shells, OVERLAP_CONVENTIONS, 'L2')
        atcoords = np.zeros((1, 3))
        overlap = compute_overlap(obasis, atcoords)
        assert_allclose(np.diag(overlap), np.ones(obasis.nbasis))


def test_normalization_basics_generalized():
    for angmom in range(2, 7):
        shells = [Shell(0, [angmom] * 2, ['c', 'p'], np.array([0.23]), np.array([[1.0, 1.0]]))]
        obasis = MolecularBasis(shells, OVERLAP_CONVENTIONS, 'L2')
        atcoords = np.zeros((1, 3))
        overlap = compute_overlap(obasis, atcoords)
        assert_allclose(np.diag(overlap), np.ones(obasis.nbasis))


def test_load_fchk_hf_sto3g_num():
    with path('iodata.test.data', 'load_fchk_hf_sto3g_num.npy') as fn_npy:
        ref = np.load(str(fn_npy))
    with path('iodata.test.data', 'hf_sto3g.fchk') as fn_fchk:
        data = load_one(fn_fchk)
    assert_allclose(ref, compute_overlap(data.obasis, data.atcoords), rtol=1.e-5, atol=1.e-8)


def test_load_fchk_o2_cc_pvtz_pure_num():
    with path('iodata.test.data',
              'load_fchk_o2_cc_pvtz_pure_num.npy') as fn_npy:
        ref = np.load(str(fn_npy))
    with path('iodata.test.data', 'o2_cc_pvtz_pure.fchk') as fn_fchk:
        data = load_one(fn_fchk)
    assert_allclose(ref, compute_overlap(data.obasis, data.atcoords), rtol=1.e-5, atol=1.e-8)


def test_load_fchk_o2_cc_pvtz_cart_num():
    with path('iodata.test.data',
              'load_fchk_o2_cc_pvtz_cart_num.npy') as fn_npy:
        ref = np.load(str(fn_npy))
    with path('iodata.test.data', 'o2_cc_pvtz_cart.fchk') as fn_fchk:
        data = load_one(fn_fchk)
    obasis = attr.evolve(data.obasis, conventions=OVERLAP_CONVENTIONS)
    assert_allclose(ref, compute_overlap(obasis, data.atcoords), rtol=1.e-5, atol=1.e-8)


def test_overlap_l1():
    dbasis = MolecularBasis([], {}, 'L1')
    atcoords = np.zeros((1, 3))
    with raises(ValueError):
        _ = compute_overlap(dbasis, atcoords)


def test_converting_between_orthonormal_basis_set():
    # Test converting from basis set of M Elements to N Elements where M <= N.
    # All basis sets are assumed orthonormal and 1st basis set is assumed to be
    # in the second basis set.
    M = np.random.randint(1, 25)
    N = np.random.randint(M, 50)
    coeffs1 = np.random.random(M)
    basis2_overlap = np.eye(N)  # Since it is orthonormal, it is identity.

    basis21_overlap = np.zeros((N, M))
    for i in range(0, M):
        basis21_overlap[i, i] = 1.  # Since it is a subset.

    coeffs2 = convert_vector_basis(coeffs1, basis2_overlap, basis21_overlap)
    assert_allclose(coeffs1, coeffs2[:M], rtol=1.e-5, atol=1.e-8)
    assert_allclose(np.zeros(coeffs2[M:].shape), coeffs2[M:], rtol=1.e-5, atol=1.e-8)

    # Test converting in the reverse direction ie Large to small.
    coeffs2 = np.random.random(N)
    basis1_overlap = np.eye(M)
    basis12_overlap = np.zeros((M, N))
    for i in range(0, M):
        basis12_overlap[i, i] = 1.  # Since it is a subset.

    coeffs1 = convert_vector_basis(coeffs2, basis1_overlap, basis12_overlap)
    assert_equal(len(coeffs1), M)
    assert_allclose(coeffs1, coeffs2[:M], rtol=1.e-5, atol=1.e-8)


def test_converting_from_cartesian_to_pure():
    # Test converting simple one coefficient, S-type.
    overlap_cp = tfs[0]
    coeff = np.array([5.])
    coeff2 = convert_vector_basis(coeff, np.eye(1), overlap_cp)
    assert_allclose(coeff, coeff2, rtol=1.e-5, atol=1.e-8)

    # Test converting p-type.
    overlap_cp = tfs[1]
    coeff = np.array([1., 2., 3.])
    coeff2 = convert_vector_basis(coeff, np.eye(3), overlap_cp)
    desired = np.array([3., 1., 2.])
    assert_allclose(coeff2, desired, rtol=1.e-5, atol=1.e-8)

    # Test converting d-type.
    overlap_cp = tfs[2]
    coeff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    coeff2 = convert_vector_basis(coeff, np.eye(5), overlap_cp)
    desired = overlap_cp.dot(coeff)
    assert_allclose(coeff2, desired, rtol=1.e-5, atol=1.e-8)


def test_converting_from_pure_to_cartesian():
    # Test converting S-type.
    overlap_cp = tfs[0]
    coeff = np.array([5.])
    coeff2 = convert_vector_basis(coeff, np.eye(1), overlap_cp.T)
    assert_allclose(coeff, coeff2, rtol=1.e-5, atol=1.e-8)

    # Test converting P-type.
    overlap_cp = tfs[1]
    coeff = np.array([1., 2., 3.])
    coeff2 = convert_vector_basis(coeff, np.eye(3), overlap_cp.T)
    desired = np.linalg.lstsq(overlap_cp, coeff, rcond=None)[0]
    assert_allclose(coeff2, desired, rtol=1.e-5, atol=1.e-8)

    # Test converting D-type.
    overlap_cp = tfs[2]
    pcoeff = np.array([1., 2., 3., 4., 5.])
    ccoeff = convert_vector_basis(pcoeff, np.eye(6), np.linalg.pinv(overlap_cp))
    desired =  np.linalg.lstsq(overlap_cp, pcoeff, rcond=None)[0]
    assert_allclose(ccoeff, desired, rtol=1.e-5, atol=1.e-8)
