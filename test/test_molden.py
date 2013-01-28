# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--


import numpy as np

from horton import *
from horton.io.test.common import compute_mulliken


def test_load_molden_li2():
    fn_mkl = context.get_fn('test/li2.molden.input')
    sys = System.from_file(fn_mkl)

    # Check normalization
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5)
    sys.wfn.exp_beta.check_normalization(sys.get_overlap(), 1e-5)

    # Check charges
    charges = compute_mulliken(sys)
    expected_charges = np.array([0.5, 0.5])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_molden_h2o():
    fn_mkl = context.get_fn('test/h2o.molden.input')
    sys = System.from_file(fn_mkl)

    # Check normalization
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5)

    # Check charges
    charges = compute_mulliken(sys)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert abs(charges - expected_charges).max() < 1e-5
