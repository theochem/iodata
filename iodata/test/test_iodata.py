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
# pylint: disable=no-member
"""Test iodata.iodata module."""


import numpy as np

from numpy.testing import assert_raises, assert_allclose

from .common import compute_1rdm
from ..iodata import load_one, IOData
from ..overlap import compute_overlap
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_typecheck():
    m = IOData(coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert np.issubdtype(m.coordinates.dtype, np.floating)
    assert not hasattr(m, 'numbers')
    m = IOData(numbers=np.array([2.0, 3.0]), pseudo_numbers=np.array([1, 1]),
               coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert np.issubdtype(m.numbers.dtype, np.integer)
    assert np.issubdtype(m.pseudo_numbers.dtype, np.floating)
    assert hasattr(m, 'numbers')
    del m.numbers
    assert not hasattr(m, 'numbers')


def test_typecheck_raises():
    # check attribute type
    assert_raises(TypeError, IOData, coordinates=np.array([[1, 2], [2, 3]]))
    assert_raises(TypeError, IOData, numbers=np.array([[1, 2], [2, 3]]))
    # check inconsistency between various attributes
    numbers, pseudo_numbers, coordinates = np.array(
        [2, 3]), np.array([1]), np.array([[1, 2, 3]])
    assert_raises(TypeError, IOData, numbers=numbers,
                  pseudo_numbers=pseudo_numbers)
    assert_raises(TypeError, IOData, numbers=numbers, coordinates=coordinates)
    assert_raises(TypeError, IOData, cube_data=np.array([1, 2]))
    cube_data = np.array([[1, 2], [2, 3], [3, 2]])
    assert_raises(TypeError, IOData, coordinates=coordinates,
                  cube_data=cube_data)


def test_unknown_format():
    assert_raises(ValueError, load_one, 'foo.unknown_file_extension')


def test_dm_water_sto3g_hf():
    with path('iodata.test.data', 'water_sto3g_hf_g03.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))
    dm = mol.dm_full_scf
    assert_allclose(dm[0, 0], 2.10503807, atol=1.e-7)
    assert_allclose(dm[0, 1], -0.439115917, atol=1.e-7)
    assert_allclose(dm[1, 1], 1.93312061, atol=1.e-7)


def test_dm_lih_sto3g_hf():
    with path('iodata.test.data', 'li_h_3-21G_hf_g09.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))

    dm_full = mol.dm_full_scf
    assert_allclose(dm_full[0, 0], 1.96589709, atol=1.e-7)
    assert_allclose(dm_full[0, 1], 0.122114249, atol=1.e-7)
    assert_allclose(dm_full[1, 1], 0.0133112081, atol=1.e-7)
    assert_allclose(dm_full[10, 10], 4.23924688E-01, atol=1.e-7)

    dm_spin = mol.dm_spin_scf
    assert_allclose(dm_spin[0, 0], 1.40210760E-03, atol=1.e-9)
    assert_allclose(dm_spin[0, 1], -2.65370873E-03, atol=1.e-9)
    assert_allclose(dm_spin[1, 1], 5.38701212E-03, atol=1.e-9)
    assert_allclose(dm_spin[10, 10], 4.23889148E-01, atol=1.e-7)


def test_dm_ch3_rohf_g03():
    with path('iodata.test.data', 'ch3_rohf_sto3g_g03.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))
    olp = compute_overlap(mol.obasis)
    dm = compute_1rdm(mol)
    assert_allclose(np.einsum('ab,ba', olp, dm), 9, atol=1.e-6)
