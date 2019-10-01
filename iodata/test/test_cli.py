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
"""Unit tests for iodata.__main__."""


import os
import functools
import subprocess

from numpy.testing import assert_equal, assert_allclose

from ..__main__ import convert
from ..api import load_one, load_many

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def _check_convert_one(myconvert, tmpdir):
    outfn = os.path.join(tmpdir, 'tmp.xyz')
    with path('iodata.test.data', 'hf_sto3g.fchk') as infn:
        myconvert(infn, outfn)
    iodata = load_one(outfn)
    assert iodata.natom == 2
    assert_equal(iodata.atnums, [9, 1])
    assert_allclose(iodata.atcoords,
                    [[0.0, 0.0, 0.190484394], [0.0, 0.0, -1.71435955]])


def test_convert_one_autofmt(tmpdir):
    myconvert = functools.partial(convert, many=False, infmt=None, outfmt=None)
    _check_convert_one(myconvert, tmpdir)


def test_convert_one_manfmt(tmpdir):
    myconvert = functools.partial(convert, many=False, infmt='fchk', outfmt='xyz')
    _check_convert_one(myconvert, tmpdir)


def test_script_one_autofmt(tmpdir):
    def myconvert(infn, outfn):
        subprocess.run(['python', '-m', 'iodata.__main__', infn, outfn],
                       check=True)
    _check_convert_one(myconvert, tmpdir)


def test_script_one_manfmt(tmpdir):
    def myconvert(infn, outfn):
        subprocess.run(['python', '-m', 'iodata.__main__', infn, outfn,
                        '-i', 'fchk', '-o', 'xyz'], check=True)
    _check_convert_one(myconvert, tmpdir)


def _check_convert_many(myconvert, tmpdir):
    outfn = os.path.join(tmpdir, 'tmp.xyz')
    with path('iodata.test.data', 'peroxide_relaxed_scan.fchk') as infn:
        myconvert(infn, outfn)
    trj = list(load_many(outfn))
    assert len(trj) == 13
    for iodata in trj:
        assert iodata.natom == 4
        assert_equal(iodata.atnums, [8, 8, 1, 1])
    assert_allclose(trj[1].atcoords[3],
                    [-1.85942837, -1.70565735, 0.0], atol=1e-5)
    assert_allclose(trj[5].atcoords[0],
                    [0.0, 1.32466211, 0.0], atol=1e-5)


def test_convert_many_autofmt(tmpdir):
    myconvert = functools.partial(convert, many=True, infmt=None, outfmt=None)
    _check_convert_many(myconvert, tmpdir)


def test_convert_many_manfmt(tmpdir):
    myconvert = functools.partial(convert, many=True, infmt='fchk', outfmt='xyz')
    _check_convert_many(myconvert, tmpdir)


def test_script_many_autofmt(tmpdir):
    def myconvert(infn, outfn):
        subprocess.run(['python', '-m', 'iodata.__main__', infn, outfn, '-m'],
                       check=True)
    _check_convert_many(myconvert, tmpdir)


def test_script_many_manfmt(tmpdir):
    def myconvert(infn, outfn):
        subprocess.run(['python', '-m', 'iodata.__main__', infn, outfn,
                        '-m', '-i', 'fchk', '-o', 'xyz'], check=True)
    _check_convert_many(myconvert, tmpdir)
