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
"""Test iodata.formats.json module."""

import os

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from .common import compute_mulliken_charges, compare_mols, check_orthonormal
from ..api import load_one, dump_one
from ..basis import convert_conventions
from ..formats.molden import _load_low
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS
from ..utils import LineIterator, angstrom, FileFormatWarning


try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water_hf_gradient():
    with path('iodata.test.data', 'water_hf_gradient.json') as json_in:
        mol = load_one(str(json_in))

    assert_equal(mol.atnums, [8, 1, 1])
    assert_allclose(mol.atcoords, np.array([[0,0,-0.1294],[0,-1.4941,1.0274],[0,1.4941,1.0274]]))


