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
# pylint: disable=unsubscriptable-object
"""Test iodata.formats.orcalog module."""

from numpy.testing import assert_equal, assert_allclose

from ..api import load_one

from ..utils import angstrom, amu

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_crambin():
    # test CHARMM crd file of crambin
    with path('iodata.test.data', 'crambin.crd') as fn_crd:
        mol = load_one(str(fn_crd))
    assert len(mol.title) == 125
    assert mol.atcoords.shape == (648, 3)
    assert_allclose(mol.atcoords[-1] / angstrom, [7.35403, -5.09628, 2.73659])
    assert mol.atffparams['attypes'].shape == (648,)
    assert mol.atffparams['resnums'].shape == (648,)
    assert mol.atffparams['resnames'].shape == (648,)
    assert mol.atffparams['attypes'][-1] == 'OT2'
    assert_equal(mol.atffparams['resnums'][46:48], [4, 4])
    assert mol.atffparams['resnames'][-1] == 'ASN'
    assert mol.extra['segid'].shape == (648,)
    assert mol.extra['resid'].shape == (648,)
    assert mol.extra['segid'][-1] == 'MAIN'
    assert mol.extra['resid'][-1] == 46
    assert mol.atmasses[-1] == 15.99900 * amu
