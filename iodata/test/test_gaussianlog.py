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
"""Test iodata.formats.log module."""

from numpy.testing import assert_equal, assert_allclose

from ..api import load_one

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def load_log_helper(fn_log):
    """Load a testing Gaussian log file with iodata.load_one."""
    with path('iodata.test.data', fn_log) as fn:
        return load_one(fn)


def test_load_operators_water_sto3g_hf_g03():
    eps = 1e-5
    mol = load_log_helper('water_sto3g_hf_g03.log')

    olp = mol.one_ints['olp']
    kin_ao = mol.one_ints['kin_ao']
    na_ao = mol.one_ints['na_ao']
    er_ao = mol.two_ints['er_ao']

    assert_equal(olp.shape, (7, 7))
    assert_equal(kin_ao.shape, (7, 7))
    assert_equal(na_ao.shape, (7, 7))
    assert_equal(er_ao.shape, (7, 7, 7, 7))

    assert_allclose(olp[0, 0], 1.0, atol=eps)
    assert_allclose(olp[0, 1], 0.236704, atol=eps)
    assert_allclose(olp[0, 2], 0.0, atol=eps)
    assert_allclose(olp[-1, -3], (-0.13198), atol=eps)

    assert_allclose(kin_ao[2, 0], 0.0, atol=eps)
    assert_allclose(kin_ao[4, 4], 2.52873, atol=eps)
    assert_allclose(kin_ao[-1, 5], 0.00563279, atol=eps)
    assert_allclose(kin_ao[-1, -3], (-0.0966161), atol=eps)

    assert_allclose(na_ao[3, 3], 9.99259, atol=eps)
    assert_allclose(na_ao[-2, -1], 1.55014, atol=eps)
    assert_allclose(na_ao[2, 6], 2.76941, atol=eps)
    assert_allclose(na_ao[0, 3], 0.0, atol=eps)

    assert_allclose(er_ao[0, 0, 0, 0], 4.78506575204, atol=eps)
    assert_allclose(er_ao[6, 6, 6, 6], 0.774605944194, atol=eps)
    assert_allclose(er_ao[6, 5, 0, 5], 0.0289424337101, atol=eps)
    assert_allclose(er_ao[5, 4, 0, 1], 0.0027414529147, atol=eps)


def test_load_operators_water_ccpvdz_pure_hf_g03():
    eps = 1e-5
    mol = load_log_helper('water_ccpvdz_pure_hf_g03.log')

    olp = mol.one_ints['olp']
    kin_ao = mol.one_ints['kin_ao']
    na_ao = mol.one_ints['na_ao']
    er_ao = mol.two_ints['er_ao']

    assert_equal(olp.shape, (24, 24))
    assert_equal(kin_ao.shape, (24, 24))
    assert_equal(na_ao.shape, (24, 24))
    assert_equal(er_ao.shape, (24, 24, 24, 24))

    assert_allclose(olp[0, 0], 1.0, atol=eps)
    assert_allclose(olp[0, 1], 0.214476, atol=eps)
    assert_allclose(olp[0, 2], 0.183817, atol=eps)
    assert_allclose(olp[10, 16], 0.380024, atol=eps)
    assert_allclose(olp[-1, -3], 0.000000, atol=eps)

    assert_allclose(kin_ao[2, 0], 0.160648, atol=eps)
    assert_allclose(kin_ao[11, 11], 4.14750, atol=eps)
    assert_allclose(kin_ao[-1, 5], -0.0244025, atol=eps)
    assert_allclose(kin_ao[-1, -6], -0.0614899, atol=eps)

    assert_allclose(na_ao[3, 3], 12.8806, atol=eps)
    assert_allclose(na_ao[-2, -1], 0.0533113, atol=eps)
    assert_allclose(na_ao[2, 6], 0.173282, atol=eps)
    assert_allclose(na_ao[-1, 0], 1.24131, atol=eps)

    assert_allclose(er_ao[0, 0, 0, 0], 4.77005841522, atol=eps)
    assert_allclose(er_ao[23, 23, 23, 23], 0.785718708997, atol=eps)
    assert_allclose(er_ao[23, 8, 23, 2], -0.0400337571969, atol=eps)
    assert_allclose(er_ao[15, 2, 12, 0], -0.0000308196281033, atol=eps)
