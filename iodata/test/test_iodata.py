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
"""Test iodata.iodata module."""


import numpy as np
from numpy.testing import assert_allclose
import pytest

from .common import compute_1rdm
from ..api import load_one, IOData
from ..overlap import compute_overlap
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_typecheck():
    m = IOData(atcoords=np.array([[1, 2, 3], [2, 3, 1]]))
    assert np.issubdtype(m.atcoords.dtype, np.floating)
    assert m.atnums is None
    m = IOData(atnums=np.array([2.0, 3.0]), atcorenums=np.array([1, 1]),
               atcoords=np.array([[1, 2, 3], [2, 3, 1]]))
    assert np.issubdtype(m.atnums.dtype, np.integer)
    assert np.issubdtype(m.atcorenums.dtype, np.floating)
    assert m.atnums is not None
    m.atnums = None
    assert m.atnums is None


def test_typecheck_raises():
    # check attribute type
    pytest.raises(TypeError, IOData, atcoords=np.array([[1, 2], [2, 3]]))
    pytest.raises(TypeError, IOData, atnums=np.array([[1, 2], [2, 3]]))
    # check inconsistency between various attributes
    atnums, atcorenums, atcoords = np.array(
        [2, 3]), np.array([1]), np.array([[1, 2, 3]])
    pytest.raises(TypeError, IOData, atnums=atnums,
                  atcorenums=atcorenums)
    pytest.raises(TypeError, IOData, atnums=atnums, atcoords=atcoords)


def test_unknown_format():
    pytest.raises(ValueError, load_one, 'foo.unknown_file_extension')


def test_dm_water_sto3g_hf():
    with path('iodata.test.data', 'water_sto3g_hf_g03.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))
    dm = mol.one_rdms['scf']
    assert_allclose(dm[0, 0], 2.10503807, atol=1.e-7)
    assert_allclose(dm[0, 1], -0.439115917, atol=1.e-7)
    assert_allclose(dm[1, 1], 1.93312061, atol=1.e-7)


def test_dm_lih_sto3g_hf():
    with path('iodata.test.data', 'li_h_3-21G_hf_g09.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))

    dm = mol.one_rdms['scf']
    assert_allclose(dm[0, 0], 1.96589709, atol=1.e-7)
    assert_allclose(dm[0, 1], 0.122114249, atol=1.e-7)
    assert_allclose(dm[1, 1], 0.0133112081, atol=1.e-7)
    assert_allclose(dm[10, 10], 4.23924688E-01, atol=1.e-7)

    dm_spin = mol.one_rdms['scf_spin']
    assert_allclose(dm_spin[0, 0], 1.40210760E-03, atol=1.e-9)
    assert_allclose(dm_spin[0, 1], -2.65370873E-03, atol=1.e-9)
    assert_allclose(dm_spin[1, 1], 5.38701212E-03, atol=1.e-9)
    assert_allclose(dm_spin[10, 10], 4.23889148E-01, atol=1.e-7)


def test_dm_ch3_rohf_g03():
    with path('iodata.test.data', 'ch3_rohf_sto3g_g03.fchk') as fn_fchk:
        mol = load_one(str(fn_fchk))
    olp = compute_overlap(mol.obasis, mol.atcoords)
    dm = compute_1rdm(mol)
    assert_allclose(np.einsum('ab,ba', olp, dm), 9, atol=1.e-6)


def test_charge_nelec1():
    # One a blank IOData object, charge and nelec can be set independently.
    mol = IOData()
    mol.nelec = 4
    mol.charge = -1
    assert mol.nelec == 4
    assert mol.charge == -1


def test_charge_nelec2():
    # When atcorenums is set, nelec and charge are coupled
    mol = IOData()
    mol.atcorenums = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
    mol.nelec = 10
    assert mol.charge == 0
    mol.charge = 1
    assert mol.nelec == 9


def test_charge_nelec3():
    # When atnums is set, nelec and charge are coupled
    mol = IOData()
    mol.atnums = np.array([6, 1, 1, 1, 1])
    mol.nelec = 10
    assert mol.charge == 0
    mol.charge = 1
    assert mol.nelec == 9


def test_undefined():
    # One a blank IOData object, accessing undefined charge and nelec should raise
    # an AttributeError.
    mol = IOData()
    assert mol.charge is None
    assert mol.nelec is None
    assert mol.spinpol is None
    assert mol.natom is None
    mol.nelec = 5
    assert mol.charge is None
    mol = IOData()
    mol.charge = 1
    assert mol.nelec is None


def test_natom():
    assert IOData(atcoords=np.zeros((4, 3))).natom == 4
    assert IOData(atcorenums=np.zeros(4)).natom == 4
    assert IOData(atgradient=np.zeros((4, 3))).natom == 4
    assert IOData(atfrozen=[False, True, False, True]).natom == 4
    assert IOData(atmasses=[0, 0, 0, 0]).natom == 4
    assert IOData(atnums=[1, 1, 1, 1]).natom == 4
