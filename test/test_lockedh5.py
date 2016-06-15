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


from nose.tools import assert_raises

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

from horton.test.common import tmpdir


def test_locked1():
    # just a silly test
    with tmpdir('horton.scripts.test.test_common.test_locked1') as dn:
        with LockedH5File('%s/foo.h5' % dn) as f:
            pass


def test_locked2():
    # test error handling in h5.File constructor
    with assert_raises(IOError):
        with tmpdir('horton.scripts.test.test_common.test_locked2') as dn:
            with LockedH5File('%s/foo.h5' % dn, mode='r', wait=0.1, count=3) as f:
                pass


def test_locked3():
    # test error handling in h5.File constructor
    with assert_raises(ValueError):
        with LockedH5File('horton.scripts.test.test_common.test_locked3.h5', driver='fubar', wait=0.1, count=3) as f:
            pass


def test_locked4():
    # test error handling of wrong driver
    with assert_raises(ValueError):
        with tmpdir('horton.scripts.test.test_common.test_locked4') as dn:
            with LockedH5File('%s/foo.h5' % dn, driver='core') as f:
                pass


def test_locked5():
    # test error handling in with clause
    with assert_raises(RuntimeError):
        with tmpdir('horton.scripts.test.test_common.test_locked5') as dn:
            with LockedH5File('%s/foo.h5' % dn) as f:
                raise RuntimeError
