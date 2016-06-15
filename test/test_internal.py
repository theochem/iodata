# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
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


import h5py as h5

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

from horton.test.common import tmpdir, compare_mols


def test_consistency_file():
    with tmpdir('horton.io.test.test_chk.test_consistency_file') as dn:
        fn_h5 = '%s/foo.h5' % dn
        fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
        fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
        mol1 = IOData.from_file(fn_fchk, fn_log)
        mol1.to_file(fn_h5)
        mol2 = IOData.from_file(fn_h5)
        compare_mols(mol1, mol2)


def test_consistency_core():
    with h5.File('horton.io.test.test_chk.test_consistency_core', driver='core', backing_store=False) as f:
        fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
        fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
        mol1 = IOData.from_file(fn_fchk, fn_log)
        mol1.to_file(f)
        mol2 = IOData.from_file(f)
        compare_mols(mol1, mol2)
