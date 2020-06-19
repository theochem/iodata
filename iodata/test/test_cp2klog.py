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
"""Test iodata.formats.cp2klog module."""

import pytest

from numpy.testing import assert_equal, assert_allclose

from .common import truncated_file, check_orthonormal

from ..api import load_one
from ..overlap import compute_overlap

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_atom_si_uks():
    with path('iodata.test.data', 'atom_si.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [14])
    assert_equal(mol.atcorenums, [4])
    assert mol.mo.kind == 'unrestricted'
    assert_equal(mol.mo.occsa, [1, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert_equal(mol.mo.occsb, [1, 0, 0, 0])
    assert_allclose(mol.mo.energiesa,
                    [-0.398761, -0.154896, -0.154896, -0.154896], atol=1.e-4)
    assert_allclose(mol.mo.energiesb,
                    [-0.334567, -0.092237, -0.092237, -0.092237], atol=1.e-4)
    assert_allclose(mol.energy, -3.761587698067, atol=1.e-10)
    assert len(mol.obasis.shells) == 3
    assert mol.obasis.shells[0].kinds == ['c', 'c']
    assert_equal(mol.obasis.shells[1].angmoms, [1, 1])
    assert mol.obasis.shells[1].kinds == ['c', 'c']
    assert_equal(mol.obasis.shells[2].angmoms, [2])
    assert mol.obasis.shells[2].kinds == ['p']
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_atom_o_rks():
    with path('iodata.test.data', 'atom_om2.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [8])
    assert_equal(mol.atcorenums, [6])
    assert mol.mo.kind == 'restricted'
    assert_equal(mol.mo.occs, [2, 2, 2, 2])
    assert_allclose(mol.mo.energies, [0.102709, 0.606458, 0.606458, 0.606458], atol=1.e-4)
    assert_allclose(mol.energy, -15.464982778766, atol=1.e-10)
    assert_equal(mol.obasis.shells[0].angmoms, [0, 0])
    assert len(mol.obasis.shells) == 3
    assert mol.obasis.shells[0].kinds == ['c', 'c']
    assert_equal(mol.obasis.shells[1].angmoms, [1, 1])
    assert mol.obasis.shells[1].kinds == ['c', 'c']
    assert_equal(mol.obasis.shells[2].angmoms, [2])
    assert mol.obasis.shells[2].kinds == ['p']
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)


def test_carbon_gs_ae_contracted():
    with path('iodata.test.data', 'carbon_gs_ae_contracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [6])
    assert mol.mo.kind == 'unrestricted'
    assert_allclose(mol.mo.occsa, [1, 1, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energiesa, [-10.058194, -0.526244, -0.214978, -0.214978, -0.214978])
    assert_allclose(mol.mo.occsb, [1, 1, 0, 0, 0])
    assert_allclose(mol.mo.energiesb, [-10.029898, -0.434300, -0.133323, -0.133323, -0.133323])
    assert_allclose(mol.energy, -37.836423363057)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_carbon_gs_ae_uncontracted():
    with path('iodata.test.data', 'carbon_gs_ae_uncontracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [6])
    assert mol.mo.kind == 'unrestricted'
    assert_allclose(mol.mo.occsa, [1, 1, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energiesa, [-10.050076, -0.528162, -0.217626, -0.217626, -0.217626])
    assert_allclose(mol.mo.occsb, [1, 1, 0, 0, 0])
    assert_allclose(mol.mo.energiesb, [-10.022715, -0.436340, -0.137135, -0.137135, -0.137135])
    assert_allclose(mol.energy, -37.842552743398)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_carbon_gs_pp_contracted():
    with path('iodata.test.data', 'carbon_gs_pp_contracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [4])
    assert mol.mo.kind == 'unrestricted'
    assert_allclose(mol.mo.occsa, [1, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert_allclose(mol.mo.energiesa, [-0.528007, -0.219974, -0.219974, -0.219974])
    assert_allclose(mol.mo.occsb, [1, 0, 0, 0])
    assert_allclose(mol.mo.energiesb, [-0.429657, -0.127060, -0.127060, -0.127060])
    assert_allclose(mol.energy, -5.399938535844)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_carbon_gs_pp_uncontracted():
    with path('iodata.test.data', 'carbon_gs_pp_uncontracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [4])
    assert mol.mo.kind == 'unrestricted'
    assert_allclose(mol.mo.occsa, [1, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energiesa, [-0.528146, -0.219803, -0.219803, -0.219803])
    assert_allclose(mol.mo.occsb, [1, 0, 0, 0])
    assert_allclose(mol.mo.energiesb, [-0.429358, -0.126411, -0.126411, -0.126411])
    assert_allclose(mol.energy, -5.402288849332)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_carbon_sc_ae_contracted():
    with path('iodata.test.data', 'carbon_sc_ae_contracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [6])
    assert mol.mo.kind == 'restricted'
    assert_allclose(mol.mo.occs, [2, 2, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energies, [-10.067251, -0.495823, -0.187878, -0.187878, -0.187878])
    assert_allclose(mol.energy, -37.793939631890)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)


def test_carbon_sc_ae_uncontracted():
    with path('iodata.test.data', 'carbon_sc_ae_uncontracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [6])
    assert mol.mo.kind == 'restricted'
    assert_allclose(mol.mo.occs, [2, 2, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energies, [-10.062206, -0.499716, -0.192580, -0.192580, -0.192580])
    assert_allclose(mol.energy, -37.800453482378)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)


def test_carbon_sc_pp_contracted():
    with path('iodata.test.data', 'carbon_sc_pp_contracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [4])
    assert mol.mo.kind == 'restricted'
    assert_allclose(mol.mo.occs, [2, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energies, [-0.500732, -0.193138, -0.193138, -0.193138])
    assert_allclose(mol.energy, -5.350765755382)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)


def test_carbon_sc_pp_uncontracted():
    with path('iodata.test.data', 'carbon_sc_pp_uncontracted.cp2k.out') as fn_out:
        mol = load_one(str(fn_out))
    assert_equal(mol.atnums, [6])
    assert_equal(mol.atcorenums, [4])
    assert mol.mo.kind == 'restricted'
    assert_allclose(mol.mo.occs, [2, 2 / 3, 2 / 3, 2 / 3])
    assert_allclose(mol.mo.energies, [-0.500238, -0.192365, -0.192365, -0.192365])
    assert_allclose(mol.energy, -5.352864672201)
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)


def test_errors(tmpdir):
    with path('iodata.test.data', 'carbon_sc_pp_uncontracted.cp2k.out') as fn_test:
        with truncated_file(fn_test, 0, 0, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(fn)
        with truncated_file(fn_test, 107, 10, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(fn)
        with truncated_file(fn_test, 357, 10, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(fn)
        with truncated_file(fn_test, 405, 10, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(fn)
    with path('iodata.test.data', 'carbon_gs_pp_uncontracted.cp2k.out') as fn_test:
        with truncated_file(fn_test, 456, 10, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(fn)
