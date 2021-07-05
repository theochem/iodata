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
# pylint: disable=unsubscriptable-object
"""Test iodata.formats.molden module."""

import os

import attr
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

def test_load_molden_h_cfour():
    # The file tested here is created with CFOUR 2.1.
    with path('iodata.test.data', 'h_donly_sph.molden') as fn_molden:
        mol = load_one(str(fn_molden))

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

