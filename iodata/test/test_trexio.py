
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
"""Test iodata.formats.trexio module."""

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from ..api import load_one, dump_one
from ..iodata import IOData

# Skip tests if trexio is not installed
trexio = pytest.importorskip("trexio")


def test_load_dump_consistency(tmpdir):
    """Check if dumping and loading a TREXIO file results in the same data."""
    # Create a dummy IOData object
    atcoords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atnums = np.array([1, 1])
    nelec = 2
    spinpol = 0
    data = IOData(atcoords=atcoords, atnums=atnums, nelec=nelec, spinpol=spinpol)

    # Write trexio file in a temporary folder
    fn_tmp = os.path.join(tmpdir, "test.trexio")
    dump_one(data, fn_tmp)
    
    # Load it back
    loaded_data = load_one(fn_tmp)

    # Check consistency
    assert_allclose(loaded_data.atcoords, data.atcoords, atol=1e-5)
    assert_equal(loaded_data.atnums, data.atnums)
    assert loaded_data.nelec == data.nelec
    assert loaded_data.spinpol == data.spinpol
