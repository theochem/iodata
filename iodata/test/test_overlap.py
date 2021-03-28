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

import itertools

import attr
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..api import load_one
from ..basis import MolecularBasis, Shell, convert_conventions
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS

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
    with pytest.raises(ValueError):
        _ = compute_overlap(dbasis, atcoords)
    obasis = MolecularBasis([], {}, 'L2')
    with pytest.raises(ValueError):
        _ = compute_overlap(obasis, atcoords, dbasis, atcoords)


def test_overlap_two_basis_exceptions():
    with path('iodata.test.data', 'hf_sto3g.fchk') as fn_fchk:
        data = load_one(fn_fchk)
    with pytest.raises(TypeError):
        compute_overlap(data.obasis, data.atcoords, data.obasis, None)
    with pytest.raises(TypeError):
        compute_overlap(data.obasis, data.atcoords, None, data.atcoords)


FNS_TWO_BASIS = [
    "h_sto3g.fchk",
    "hf_sto3g.fchk",
    "2h-azirine-cc.fchk",
    "water_ccpvdz_pure_hf_g03.fchk"
]


@pytest.mark.parametrize("fn", FNS_TWO_BASIS)
def test_overlap_two_basis_same(fn):
    with path('iodata.test.data', fn) as pth:
        mol = load_one(pth)
    olp_a = compute_overlap(mol.obasis, mol.atcoords, mol.obasis, mol.atcoords)
    olp_b = compute_overlap(mol.obasis, mol.atcoords)
    assert_allclose(olp_a, olp_b, rtol=0, atol=1e-14)


@pytest.mark.parametrize("fn0,fn1", itertools.combinations_with_replacement(FNS_TWO_BASIS, 2))
def test_overlap_two_basis_different(fn0, fn1):
    with path('iodata.test.data', fn0) as pth0:
        mol0 = load_one(pth0)
    with path('iodata.test.data', fn1) as pth1:
        mol1 = load_one(pth1)
    # Direct computation of the off-diagonal block.
    olp_a = compute_overlap(mol0.obasis, mol0.atcoords, mol1.obasis, mol1.atcoords)
    # Poor-man's approach: combine two molecules into one and compute its
    # overlap matrix.
    atcoords = np.concatenate([mol0.atcoords, mol1.atcoords])
    shells = mol0.obasis.shells + [
        attr.evolve(shell, icenter=shell.icenter + mol0.natom)
        for shell in mol1.obasis.shells
    ]
    obasis = MolecularBasis(shells, OVERLAP_CONVENTIONS, "L2")
    olp_big = compute_overlap(obasis, atcoords)
    # Get the off-diagonal block and reorder.
    olp_b = olp_big[:olp_a.shape[0], olp_a.shape[0]:]
    assert olp_a.shape == olp_b.shape
    permutation0, signs0 = convert_conventions(mol0.obasis, OVERLAP_CONVENTIONS, reverse=True)
    olp_b = olp_b[permutation0] * signs0.reshape(-1, 1)
    permutation1, signs1 = convert_conventions(mol1.obasis, OVERLAP_CONVENTIONS, reverse=True)
    olp_b = olp_b[:, permutation1] * signs1
    # Finally compare the numbers.
    assert_allclose(olp_a, olp_b, rtol=0, atol=1e-14)
