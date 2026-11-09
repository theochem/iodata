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
"""Unit tests for iodata.utils."""

import numpy as np
import pytest
from numpy.testing import assert_equal

from ..utils import amu, strtobool, triangle_to_dense


def test_triangle_to_dense_small():
    assert_equal(triangle_to_dense(np.array([3.0])), [[3.0]])
    assert_equal(triangle_to_dense(np.array([1.0, 2.0, 3.0])), [[1.0, 2.0], [2.0, 3.0]])
    assert_equal(
        triangle_to_dense(np.arange(1.0, 7.0)),
        [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0]],
    )


@pytest.mark.parametrize("nrow", [1, 2, 5, 12])
def test_triangle_to_dense_random(nrow):
    rng = np.random.default_rng(nrow)
    matrix = rng.normal(size=(nrow, nrow))
    matrix = matrix + matrix.T
    result = triangle_to_dense(matrix[np.tril_indices(nrow)])
    # The elements are copied, so there are no rounding errors.
    assert_equal(result, matrix)


@pytest.mark.parametrize("size", [2, 4, 5, 7])
def test_triangle_to_dense_wrong_size(size):
    with pytest.raises(ValueError, match="triangle"):
        triangle_to_dense(np.zeros(size))


def test_amu():
    assert abs(amu * 1.008 - 1837.47) < 1e-1


def test_strtobool():
    assert strtobool("T") is True
    assert strtobool("false") is False
    assert strtobool("y") is True
    with pytest.raises(ValueError):
        strtobool("whatever")
