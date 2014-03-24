# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
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
#pylint: skip-file


import numpy as np, os

from horton import *
from horton.io.test.common import compute_mulliken_charges, compare_data
from horton.test.common import tmpdir


def test_load_molden_li2():
    fn_mkl = context.get_fn('test/li2.molden.input')
    data = load_smart(fn_mkl)
    obasis = data['obasis']
    wfn = data['wfn']
    lf = data['lf']
    numbers = data['numbers']

    # Check normalization
    olp = lf.create_one_body()
    obasis.compute_overlap(olp)
    wfn.exp_alpha.check_normalization(olp, 1e-5)
    wfn.exp_beta.check_normalization(olp, 1e-5)

    # Check charges
    charges = compute_mulliken_charges(obasis, lf, numbers, wfn)
    expected_charges = np.array([0.5, 0.5])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_molden_h2o():
    fn_mkl = context.get_fn('test/h2o.molden.input')
    data = load_smart(fn_mkl)
    obasis = data['obasis']
    wfn = data['wfn']
    lf = data['lf']
    numbers = data['numbers']

    # Check normalization
    olp = lf.create_one_body()
    obasis.compute_overlap(olp)
    wfn.exp_alpha.check_normalization(olp, 1e-5)

    # Check charges
    charges = compute_mulliken_charges(obasis, lf, numbers, wfn)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert abs(charges - expected_charges).max() < 1e-5


def check_load_dump_consistency(fn):
    data1 = load_smart(context.get_fn(os.path.join('test', fn)))
    with tmpdir('horton.io.test.test_molden.check_load_dump_consistency.%s' % fn) as dn:
        fn_tmp = os.path.join(dn, 'foo.molden.input')
        dump_smart(fn_tmp, data1)
        data2 = load_smart(fn_tmp)
    compare_data(data1, data2)


def test_load_dump_consistency_h2o():
    check_load_dump_consistency('h2o.molden.input')


def test_load_dump_consistency_li2():
    check_load_dump_consistency('li2.molden.input')
