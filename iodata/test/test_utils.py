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

from ..utils import amu, parse_fortran_float, parse_fortran_floats, strtobool


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("+1.5", 1.5),
        ("1", 1.0),
        ("1.5E+03", 1500.0),
        ("1.5e-03", 1.5e-3),
        ("1.5D+03", 1500.0),
        ("1.5d-03", 1.5e-3),
        ("-1.5D-03", -1.5e-3),
        # The letter of the exponent can be absent.
        ("1.5+03", 1500.0),
        ("1.5-03", 1.5e-3),
        ("1.234567-123", 1.234567e-123),
        ("-1.234567-123", -1.234567e-123),
        ("1.234567+123", 1.234567e123),
        # Spaces around the number are ignored, also with a sign.
        ("  1.5-123  ", 1.5e-123),
        ("  -1.5  ", -1.5),
        ("  -1.5-123", -1.5e-123),
        ("\t+1.5D+01\n", 15.0),
    ],
)
def test_parse_fortran_float(word, expected):
    assert parse_fortran_float(word) == expected


@pytest.mark.parametrize("word", ["", "abc", "1.5-", "1.5E", "--1.5", "1.5D+x", "1 2"])
def test_parse_fortran_float_error(word):
    with pytest.raises(ValueError):
        parse_fortran_float(word)


def test_parse_fortran_floats():
    # No Fortran numbers
    assert_equal(parse_fortran_floats(["1.5", "-2", "3e-4"]), [1.5, -2.0, 3e-4])
    # Some Fortran numbers
    assert_equal(
        parse_fortran_floats(["1.5D+03", "-2", "3.0-123", " -4.5 "]),
        [1500.0, -2.0, 3.0e-123, -4.5],
    )
    assert parse_fortran_floats([]).shape == (0,)
    assert parse_fortran_floats(["1.0D0"]).dtype == np.float64


def test_parse_fortran_floats_error():
    with pytest.raises(ValueError):
        parse_fortran_floats(["1.5", "nope"])


def test_amu():
    assert abs(amu * 1.008 - 1837.47) < 1e-1


def test_strtobool():
    assert strtobool("T") is True
    assert strtobool("false") is False
    assert strtobool("y") is True
    with pytest.raises(ValueError):
        strtobool("whatever")
