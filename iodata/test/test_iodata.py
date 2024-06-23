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
# ruff: noqa: SLF001
"""Test iodata.iodata module."""

from importlib.resources import as_file, files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import IOData, load_one
from ..overlap import compute_overlap
from ..utils import FileFormatError
from .common import compute_1rdm


def test_typecheck():
    m = IOData(atcoords=np.array([[1, 2, 3], [2, 3, 1]]))
    assert np.issubdtype(m.atcoords.dtype, np.floating)
    assert m.atnums is None
    m = IOData(
        atnums=np.array([2.0, 3.0]),
        atcorenums=np.array([1, 1]),
        atcoords=np.array([[1, 2, 3], [2, 3, 1]]),
    )
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
    atnums, atcorenums, atcoords = np.array([2, 3]), np.array([1]), np.array([[1, 2, 3]])
    pytest.raises(TypeError, IOData, atnums=atnums, atcorenums=atcorenums)
    pytest.raises(TypeError, IOData, atnums=atnums, atcoords=atcoords)


def test_unknown_format():
    with pytest.raises(FileFormatError, match="Cannot find file format with feature"):
        load_one("foo.unknown_file_extension")


def test_dm_water_sto3g_hf():
    with as_file(files("iodata.test.data").joinpath("water_sto3g_hf_g03.fchk")) as fn_fchk:
        mol = load_one(str(fn_fchk))
    dm = mol.one_rdms["scf"]
    assert_allclose(dm[0, 0], 2.10503807, atol=1.0e-7)
    assert_allclose(dm[0, 1], -0.439115917, atol=1.0e-7)
    assert_allclose(dm[1, 1], 1.93312061, atol=1.0e-7)


def test_dm_lih_sto3g_hf():
    with as_file(files("iodata.test.data").joinpath("li_h_3-21G_hf_g09.fchk")) as fn_fchk:
        mol = load_one(str(fn_fchk))

    dm = mol.one_rdms["scf"]
    assert_allclose(dm[0, 0], 1.96589709, atol=1.0e-7)
    assert_allclose(dm[0, 1], 0.122114249, atol=1.0e-7)
    assert_allclose(dm[1, 1], 0.0133112081, atol=1.0e-7)
    assert_allclose(dm[10, 10], 4.23924688e-01, atol=1.0e-7)

    dm_spin = mol.one_rdms["scf_spin"]
    assert_allclose(dm_spin[0, 0], 1.40210760e-03, atol=1.0e-9)
    assert_allclose(dm_spin[0, 1], -2.65370873e-03, atol=1.0e-9)
    assert_allclose(dm_spin[1, 1], 5.38701212e-03, atol=1.0e-9)
    assert_allclose(dm_spin[10, 10], 4.23889148e-01, atol=1.0e-7)


def test_dm_ch3_rohf_g03():
    with as_file(files("iodata.test.data").joinpath("ch3_rohf_sto3g_g03.fchk")) as fn_fchk:
        mol = load_one(str(fn_fchk))
    olp = compute_overlap(mol.obasis, mol.atcoords)
    dm = compute_1rdm(mol)
    assert_allclose(np.einsum("ab,ba", olp, dm), 9, atol=1.0e-6)


def test_charge_nelec1():
    # One a blank IOData object, charge and nelec can be set independently.
    mol = IOData()
    mol.nelec = 4
    mol.charge = -1
    assert mol.nelec == 4
    assert mol.charge == -1
    # The values should be stored as private attributes.
    assert mol._nelec == 4
    assert mol._charge == -1


def test_charge_nelec2():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.atcorenums = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
    mol.nelec = 10
    assert mol.charge == 0
    mol.charge = 1
    assert mol.nelec == 9
    # Only _nelec should be set.
    assert mol._nelec == 9
    assert mol._charge is None


def test_charge_nelec3():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.atnums = np.array([6, 1, 1, 1, 1])
    mol.nelec = 10
    assert mol.charge == 0
    # Accessing charge should assign _atcorenums.
    assert_equal(mol._atcorenums, np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
    # Changing charge should change nelec.
    mol.charge = 1
    assert mol.nelec == 9
    assert mol.charge == 1
    # Only _nelec should be set.
    assert mol._nelec == 9
    assert mol._charge is None


def test_charge_nelec4():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.atnums = np.array([6, 1, 1, 1, 1])
    mol.charge = 1
    # _atcorenums, _nelec and charge should be set, not _charge
    assert_equal(mol._atcorenums, np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
    assert mol._nelec == 9
    assert mol._charge is None
    assert mol.charge == 1


def test_charge_nelec5():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.charge = 1
    mol.atnums = np.array([6, 1, 1, 1, 1])
    # Access atcorenums to trigger the setter
    assert mol.atcorenums.sum() == 10
    # _atcorenums, _nelec and charge should be set, not _charge
    assert_equal(mol._atcorenums, np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
    assert mol._nelec == 9
    assert mol._charge is None
    assert mol.charge == 1


def test_charge_nelec6():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.nelec = 8
    mol.atnums = np.array([6, 1, 1, 1, 1])
    # Access atcorenums to trigger the setter
    assert mol.atcorenums.sum() == 10
    # _atcorenums, _nelec and charge should be set, not _charge
    assert_equal(mol._atcorenums, np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
    assert mol._nelec == 8
    assert mol._charge is None
    assert mol.charge == 2


def test_charge_nelec7():
    # When atcorenums is set, nelec and charge become coupled.
    mol = IOData()
    mol.nelec = 8
    mol.charge = 1  # This will be discarded by setting _atcorenums.
    mol.atnums = np.array([6, 1, 1, 1, 1])
    # Access atcorenums to trigger the setter
    assert mol.atcorenums.sum() == 10
    # _atcorenums, _nelec and charge should be set, not _charge
    assert_equal(mol._atcorenums, np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
    assert mol._nelec == 8
    assert mol._charge is None
    assert mol.charge == 2


def test_charge_nelec8():
    mol = IOData()
    mol.charge = 0.0
    mol.atcorenums = None
    assert mol.charge == 0.0
    assert mol.atcorenums is None
    assert mol.nelec is None


def test_charge_nelec9():
    mol = IOData()
    mol.charge = 1.0
    mol.atcorenums = np.array([8.0, 1.0, 1.0])
    assert mol.charge == 1.0
    assert mol._charge is None  # charge is derived
    assert_equal(mol._atcorenums, np.array([8.0, 1.0, 1.0]))
    assert mol.nelec == 9.0
    mol.charge = None
    assert mol.nelec is None


def test_charge_nelec10():
    mol = IOData()
    mol.charge = 1.0
    mol.atnums = np.array([8, 1, 1])
    assert mol.charge == 1.0  # This triggers atcorenums to be initialized.
    assert mol._charge is None
    assert_equal(mol._atcorenums, np.array([8.0, 1.0, 1.0]))
    assert mol.nelec == 9.0
    mol.charge = None
    assert mol.nelec is None


def test_charge_nelec11():
    mol = IOData()
    mol.atnums = np.array([8, 1, 1])
    mol.charge = 1.0
    assert mol.nelec == 9.0
    mol.charge = None
    assert mol.nelec is None


def test_charge_nelec12():
    mol = IOData()
    mol.atnums = np.array([8, 1, 1])
    mol.nelec = 11.0
    assert mol.charge == -1.0
    mol.nelec = None
    assert mol.charge is None


def test_charge_nelec13():
    mol = IOData()
    mol.atcorenums = np.array([8, 1, 1])
    mol.charge = 1.0
    mol.atcorenums = None
    assert mol.charge == 1.0
    assert mol.nelec == 9.0
    assert mol.atcorenums is None


def test_charge_nelec14():
    mol = IOData()
    mol.nelec = 8
    mol.atcorenums = None
    mol.atcorenums = np.array([8, 1, 1])
    assert mol.atcorenums.dtype == float
    assert mol.charge == 2.0
    assert mol._charge is None
    mol.atcorenums = None
    assert mol.nelec == 8
    assert mol.charge == 2.0


def test_charge_nelec15():
    mol = IOData()
    mol.nelec = 8
    mol.atcorenums = np.array([8, 1, 1])
    assert mol.charge == 2.0
    mol.nelec = None
    assert mol.nelec is None
    assert mol.charge is None


def test_undefined():
    # One a blank IOData object, accessing undefined charge and nelec should
    # return None.
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


def test_spinpol1():
    mol = IOData(spinpol=3)
    assert mol.spinpol == 3


def test_spinpol2():
    mol = IOData()
    mol.spinpol = 3
    assert mol.spinpol == 3


def test_derived1():
    # When loading a file with molecular orbitals, nelec, charge and spinpol are
    # derived from the mo object:
    with as_file(files("iodata.test.data").joinpath("ch3_rohf_sto3g_g03.fchk")) as fn_fchk:
        mol = load_one(str(fn_fchk))
    assert mol.nelec == mol.mo.nelec
    assert mol.charge == mol.atcorenums.sum() - mol.mo.nelec
    assert mol.spinpol == mol.mo.spinpol
    assert mol._nelec is None
    assert mol._charge is None
    assert mol._spinpol is None
    with pytest.raises(TypeError):
        mol.charge = 2
    with pytest.raises(TypeError):
        mol.nelec = 3
    with pytest.raises(TypeError):
        mol.spinpol = 1


def test_derived2():
    mol = IOData(atnums=[1, 1, 8], charge=1)
    assert mol._charge is None
    assert mol._nelec == mol.atcorenums.sum() - 1
    assert mol.charge == 1.0
    assert mol.nelec == mol.atcorenums.sum() - 1


def test_natom():
    assert IOData(atcoords=np.zeros((4, 3))).natom == 4
    assert IOData(atcorenums=np.zeros(4)).natom == 4
    assert IOData(atgradient=np.zeros((4, 3))).natom == 4
    assert IOData(atfrozen=[False, True, False, True]).natom == 4
    assert IOData(atmasses=[0, 0, 0, 0]).natom == 4
    assert IOData(atnums=[1, 1, 1, 1]).natom == 4
