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


import tempfile, os, h5py as h5

from horton import *
from horton.test.common import compare_systems


def test_consistency_file():
    tmpdir = tempfile.mkdtemp('horton.io.test.test_chk.test_consistency_file')
    try:
        fn_chk = '%s/chk.h5' % tmpdir
        fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
        fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
        sys1 = System.from_file(fn_fchk, fn_log, chk=None)
        sys1.to_file(fn_chk)
        sys2 = System.from_file(fn_chk, chk=None)
        compare_systems(sys1, sys2)
    finally:
        if os.path.isfile(fn_chk):
            os.remove(fn_chk)
        os.rmdir(tmpdir)


def test_consistency_core():
    with h5.File('horton.io.test.test_chk.test_consistency_core', driver='core', backing_store=False) as chk:
        fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
        fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
        sys1 = System.from_file(fn_fchk, fn_log, chk=None)
        sys1.to_file(chk)
        sys2 = System.from_file(chk, chk=None)
        compare_systems(sys1, sys2)
