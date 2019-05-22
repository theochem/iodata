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
# pylint: disable=unsubscriptable-object,no-member
"""Test iodata.formats.molekel module."""

from numpy.testing import assert_equal, assert_allclose

from .common import check_orthonormal
from ..api import load_one
from ..overlap import compute_overlap
from ..utils import angstrom

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_mkl_ethanol():
    with path('iodata.test.data', 'ethanol.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))

    # Direct checks with mkl file
    assert_equal(mol.atnums.shape, (9,))
    assert_equal(mol.atnums[0], 1)
    assert_equal(mol.atnums[4], 6)
    assert_equal(mol.atcoords.shape, (9, 3))
    assert_allclose(mol.atcoords[2, 1] / angstrom, 2.239037, atol=1.e-5)
    assert_allclose(mol.atcoords[5, 2] / angstrom, 0.948420, atol=1.e-5)
    assert mol.obasis.nbasis == 39
    assert_allclose(mol.obasis.shells[0].exponents[0], 18.731137000)
    assert_allclose(mol.obasis.shells[4].exponents[0], 7.868272400)
    assert_allclose(mol.obasis.shells[7].exponents[1], 2.825393700)
    # No correspondence due to correction of the normalization of
    # the primivitves:
    # assert_allclose(mol.obasis.shells[2].coeffs[1, 0], 0.989450608)
    # assert_allclose(mol.obasis.shells[2].coeffs[3, 0], 2.079187061)
    # assert_allclose(mol.obasis.shells[-1].coeffs[-1, -1], 0.181380684)
    assert_equal([shell.icenter for shell in mol.obasis.shells[:5]], [0, 0, 1, 1, 1])
    assert_equal([shell.angmoms[0] for shell in mol.obasis.shells[:5]], [0, 0, 0, 0, 1])
    assert_equal([shell.nprim for shell in mol.obasis.shells[:5]], [3, 1, 6, 3, 3])
    assert_equal(mol.mo.coeffs.shape, (39, 39))
    assert_equal(mol.mo.energies.shape, (39,))
    assert_equal(mol.mo.occs.shape, (39,))
    assert_equal(mol.mo.occs[:13], 2.0)
    assert_equal(mol.mo.occs[13:], 0.0)
    assert_allclose(mol.mo.energies[4], -1.0206976)
    assert_allclose(mol.mo.energies[-1], 2.0748685)
    assert_allclose(mol.mo.coeffs[0, 0], 0.0000119)
    assert_allclose(mol.mo.coeffs[1, 0], -0.0003216)
    assert_allclose(mol.mo.coeffs[-1, -1], -0.1424743)


def test_load_mkl_li2():
    with path('iodata.test.data', 'li2.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_load_mkl_h2():
    with path('iodata.test.data', 'h2_sto3g.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)
