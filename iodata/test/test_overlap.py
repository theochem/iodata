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

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from pytest import raises

from ..api import load_one
from ..basis import MolecularBasis, Shell
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS
from ..overlap_accel import fac2, _binom

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
        atcoords = np.zeros((3, 1))
        overlap = compute_overlap(obasis, atcoords)
        assert_allclose(np.diag(overlap), np.ones(obasis.nbasis))


def test_normalization_basics_generalized():
    for angmom in range(2, 7):
        shells = [Shell(0, [angmom] * 2, ['c', 'p'], np.array([0.23]), np.array([[1.0, 1.0]]))]
        obasis = MolecularBasis(shells, OVERLAP_CONVENTIONS, 'L2')
        atcoords = np.zeros((3, 1))
        overlap = compute_overlap(obasis, atcoords)
        assert_allclose(np.diag(overlap), np.ones(obasis.nbasis))


def test_fac2():
    assert_equal(fac2(-20), 1)
    assert_equal(fac2(0), 1)
    assert_equal(fac2(1), 1)
    assert_equal(fac2(2), 2)
    assert_equal(fac2(3), 3)
    assert_equal(fac2(4), 8)
    assert_equal(fac2(5), 15)


def test_binom():
    assert_equal(_binom(1, 1), 1)
    assert_equal(_binom(5, 3), 10)
    assert_equal(_binom(3, 2), 3)
    assert_equal(_binom(10, 4), 210)
    assert_equal(_binom(18, 14), 3060)
    assert_equal(_binom(5, 1), 5)
    assert_equal(_binom(5, 0), 1)
    assert_equal(_binom(0, 0), 1)
    assert_equal(_binom(5, 5), 1)


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
    obasis = data.obasis._replace(conventions=OVERLAP_CONVENTIONS)
    assert_allclose(ref, compute_overlap(obasis, data.atcoords), rtol=1.e-5, atol=1.e-8)


def test_overlap_l1():
    dbasis = MolecularBasis([], {}, 'L1')
    atcoords = np.zeros((3, 1))
    with raises(ValueError):
        _ = compute_overlap(dbasis, atcoords)
