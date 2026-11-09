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
"""Unit tests for iodata.api.

Relatively simple formats are used in this module to keep testing simple
and focus on the functionality of the API rather than the formats.
"""

import os

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ..api import dump_many, dump_one, load_many, write_input
from ..iodata import IOData
from ..utils import DumpError, FileFormatError, PrepareDumpError


def test_json_no_pattern(tmpdir):
    path_json = os.path.join(tmpdir, "name.json")
    with pytest.raises(FileFormatError):
        dump_one(IOData(atnums=[1, 2, 3]), path_json)
    assert not os.path.isfile(path_json)


def test_nonexisting_format(tmpdir):
    path = os.path.join(tmpdir, "foobar")
    with pytest.raises(FileFormatError):
        dump_one(IOData(atnums=[1, 2, 3]), path, fmt="file-format-does-not-exist-at-all")
    assert not os.path.isfile(path)


def test_nodump(tmpdir):
    path = os.path.join(tmpdir, "foobar")
    with pytest.raises(FileFormatError):
        dump_one(IOData(atnums=[1, 2, 3]), path, fmt="cp2klog")
    assert not os.path.isfile(path)


def test_noinput(tmpdir):
    path = os.path.join(tmpdir, "foobar")
    with pytest.raises(FileFormatError):
        write_input(IOData(atnums=[1, 2, 3]), path, fmt="this-input-format-does-not-exist")
    assert not os.path.isfile(path)


def test_empty_dump_many_no_file(tmpdir):
    path_xyz = os.path.join(tmpdir, "empty.xyz")
    with pytest.raises(DumpError):
        dump_many([], path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_one_missing_attribute_no_file(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    with pytest.raises(PrepareDumpError):
        dump_one(IOData(atnums=[1, 2, 3]), path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_many_missing_attribute_first(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    iodatas = [
        IOData(atnums=[1, 1]),
        IOData(atnums=[1, 1], atcoords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    with pytest.raises(PrepareDumpError):
        dump_many(iodatas, path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_many_missing_attribute_second(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    iodata0 = IOData(atnums=[1, 1], atcoords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    iodatas = [iodata0, IOData(atnums=[1, 1])]
    with pytest.raises(PrepareDumpError):
        dump_many(iodatas, path_xyz)
    assert os.path.isfile(path_xyz)
    iodatas = list(load_many(path_xyz))
    assert len(iodatas) == 1
    assert_array_equal(iodatas[0].atnums, iodata0.atnums)
    assert_allclose(iodatas[0].atcoords, iodata0.atcoords)


def test_dump_many_generator(tmpdir):
    path_xyz = os.path.join(tmpdir, "traj.xyz")

    iodata0 = IOData(atnums=[1, 1], atcoords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    iodata1 = IOData(atnums=[2, 2], atcoords=[[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    def iodata_generator():
        yield iodata0
        yield iodata1

    dump_many(iodata_generator(), path_xyz)
    assert os.path.isfile(path_xyz)
    iodatas = list(load_many(path_xyz))
    assert len(iodatas) == 2
    assert_array_equal(iodatas[0].atnums, iodata0.atnums)
    assert_array_equal(iodatas[1].atnums, iodata1.atnums)
    assert_allclose(iodatas[0].atcoords, iodata0.atcoords)
    assert_allclose(iodatas[1].atcoords, iodata1.atcoords)
