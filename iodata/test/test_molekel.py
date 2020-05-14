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

import os

from numpy.testing import assert_equal, assert_allclose
import pytest

from .common import check_orthonormal, compare_mols
from ..api import load_one, dump_one
from ..overlap import compute_overlap
from ..utils import angstrom, FileFormatWarning

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def check_load_dump_consistency(fn, tmpdir):
    """Check if data is preserved after dumping and loading a Molekel file.

    Parameters
    ----------
    fn : str
        The Molekel filename to load
    tmpdir : str
        The temporary directory to dump and load the file.

    """
    with path('iodata.test.data', fn) as file_name:
        mol1 = load_one(str(file_name))
    fn_tmp = os.path.join(tmpdir, 'foo.bar')
    dump_one(mol1, fn_tmp, fmt='molekel')
    mol2 = load_one(fn_tmp, fmt='molekel')
    compare_mols(mol1, mol2)


def test_load_dump_consistency_h2(tmpdir):
    with pytest.warns(FileFormatWarning) as record:
        check_load_dump_consistency('h2_sto3g.mkl', tmpdir)
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]


def test_load_dump_consistency_ethanol(tmpdir):
    with pytest.warns(FileFormatWarning) as record:
        check_load_dump_consistency('ethanol.mkl', tmpdir)
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]


def test_load_dump_consistency_li2(tmpdir):
    with pytest.warns(FileFormatWarning) as record:
        check_load_dump_consistency('li2.mkl', tmpdir)
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]


def test_load_mkl_ethanol():
    with path('iodata.test.data', 'ethanol.mkl') as fn_mkl:
        with pytest.warns(FileFormatWarning) as record:
            mol = load_one(str(fn_mkl))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]

    # Direct checks with mkl file
    assert_equal(mol.atnums.shape, (9,))
    assert_equal(mol.atnums[0], 1)
    assert_equal(mol.atnums[4], 6)
    assert_equal(mol.atcoords.shape, (9, 3))
    assert_allclose(mol.atcoords[2, 1] / angstrom, 2.239037, atol=1.e-5)
    assert_allclose(mol.atcoords[5, 2] / angstrom, 0.948420, atol=1.e-5)
    assert_equal(mol.atcharges['mulliken'].shape, (9,))
    q = [0.143316, -0.445861, 0.173045, 0.173021, 0.024542, 0.143066, 0.143080, -0.754230, 0.400021]
    assert_allclose(mol.atcharges['mulliken'], q)
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
        with pytest.warns(FileFormatWarning) as record:
            mol = load_one(str(fn_mkl))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]
    assert_equal(mol.atcharges['mulliken'].shape, (2,))
    assert_allclose(mol.atcharges['mulliken'], [0.5, 0.5])
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_load_mkl_h2():
    with path('iodata.test.data', 'h2_sto3g.mkl') as fn_mkl:
        with pytest.warns(FileFormatWarning) as record:
            mol = load_one(str(fn_mkl))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]
    assert_equal(mol.atcharges['mulliken'].shape, (2,))
    assert_allclose(mol.atcharges['mulliken'], [0, 0])
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)
