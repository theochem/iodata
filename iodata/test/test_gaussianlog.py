# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
# pragma pylint: disable=invalid-name,fixme
"""Test iodata.log module."""


from .. gaussianlog import load
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


# TODO: shells_to_nbasis(obasis["shell_types"]) replacement test


def test_load_operators_water_sto3g_hf_g03():
    eps = 1e-5
    with path('iodata.test.data', 'water_sto3g_hf_g03.log') as fn:
        result = load(str(fn))

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    assert overlap.shape == (7, 7)
    assert kinetic.shape == (7, 7)
    assert nuclear_attraction.shape == (7, 7)
    assert electronic_repulsion.shape == (7, 7, 7, 7)

    assert abs(overlap[0, 0] - 1.0) < eps
    assert abs(overlap[0, 1] - 0.236704) < eps
    assert abs(overlap[0, 2] - 0.0) < eps
    assert abs(overlap[-1, -3] - (-0.13198)) < eps

    assert abs(kinetic[2, 0] - 0.0) < eps
    assert abs(kinetic[4, 4] - 2.52873) < eps
    assert abs(kinetic[-1, 5] - 0.00563279) < eps
    assert abs(kinetic[-1, -3] - (-0.0966161)) < eps

    assert abs(nuclear_attraction[3, 3] - 9.99259) < eps
    assert abs(nuclear_attraction[-2, -1] - 1.55014) < eps
    assert abs(nuclear_attraction[2, 6] - 2.76941) < eps
    assert abs(nuclear_attraction[0, 3] - 0.0) < eps

    assert abs(electronic_repulsion[0, 0, 0, 0] - 4.78506575204) < eps
    assert abs(electronic_repulsion[6, 6, 6, 6] - 0.774605944194) < eps
    assert abs(electronic_repulsion[6, 5, 0, 5] - 0.0289424337101) < eps
    assert abs(electronic_repulsion[5, 4, 0, 1] - 0.00274145291476) < eps


def test_load_operators_water_ccpvdz_pure_hf_g03():
    eps = 1e-5
    with path('iodata.test.data', 'water_ccpvdz_pure_hf_g03.log') as fn:
        result = load(str(fn))

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    assert overlap.shape == (24, 24)
    assert kinetic.shape == (24, 24)
    assert nuclear_attraction.shape == (24, 24)
    assert electronic_repulsion.shape == (24, 24, 24, 24)

    assert abs(overlap[0, 0] - 1.0) < eps
    assert abs(overlap[0, 1] - 0.214476) < eps
    assert abs(overlap[0, 2] - 0.183817) < eps
    assert abs(overlap[10, 16] - 0.380024) < eps
    assert abs(overlap[-1, -3] - 0.000000) < eps

    assert abs(kinetic[2, 0] - 0.160648) < eps
    assert abs(kinetic[11, 11] - 4.14750) < eps
    assert abs(kinetic[-1, 5] - (-0.0244025)) < eps
    assert abs(kinetic[-1, -6] - (-0.0614899)) < eps

    assert abs(nuclear_attraction[3, 3] - 12.8806) < eps
    assert abs(nuclear_attraction[-2, -1] - 0.0533113) < eps
    assert abs(nuclear_attraction[2, 6] - 0.173282) < eps
    assert abs(nuclear_attraction[-1, 0] - 1.24131) < eps

    assert abs(electronic_repulsion[0, 0, 0, 0] - 4.77005841522) < eps
    assert abs(electronic_repulsion[23, 23, 23, 23] - 0.785718708997) < eps
    assert abs(electronic_repulsion[23, 8, 23, 2] - (-0.0400337571969)) < eps
    assert abs(electronic_repulsion[15, 2, 12, 0] - (-0.0000308196281033)) < eps
