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

from ..api import dump_many, dump_one
from ..iodata import IOData
from ..utils import FileFormatError


def test_empty_dump_many_no_file(tmpdir):
    path_xyz = os.path.join(tmpdir, "empty.xyz")
    with pytest.raises(FileFormatError):
        dump_many([], path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_one_missing_attribute_no_file(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    with pytest.raises(FileFormatError):
        dump_one(IOData(atnums=[1, 2, 3]), path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_many_missing_attribute_first(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    iodatas = [
        IOData(atnums=[1, 1]),
        IOData(atnums=[1, 1], atcoords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    with pytest.raises(FileFormatError):
        dump_many(iodatas, path_xyz)
    assert not os.path.isfile(path_xyz)


def test_dump_many_missing_attribute_second(tmpdir):
    path_xyz = os.path.join(tmpdir, "missing_atcoords.xyz")
    iodatas = [
        IOData(atnums=[1, 1], atcoords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        IOData(atnums=[1, 1]),
    ]
    with pytest.raises(FileFormatError):
        dump_many(iodatas, path_xyz)
    assert os.path.isfile(path_xyz)
