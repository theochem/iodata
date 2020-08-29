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
"""Unit tests for iodata.attrutils."""


import attr
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..attrutils import convert_array_to, validate_shape


@attr.s(auto_attribs=True, slots=True, on_setattr=attr.setters.convert)
class FooBar:
    """Just a silly class for testing convert_array_to."""

    spam: np.ndarray = attr.ib(converter=convert_array_to(float))


def test_convert_array_to_init():
    fb = FooBar(spam=None)
    assert fb.spam is None
    spam1 = np.array([1.0, 3.0, -1.0])
    fb = FooBar(spam1)
    assert fb.spam is spam1
    spam2 = np.array([3, 7, -2])
    fb = FooBar(spam2)
    assert fb.spam is not spam2
    assert_allclose(fb.spam, spam2)


def test_convert_array_to_assign():
    fb = FooBar(spam=None)
    assert fb.spam is None
    spam1 = np.array([1.0, 3.0, -1.0])
    fb.spam = spam1
    assert fb.spam is spam1
    spam2 = np.array([3, 7, -2])
    fb.spam = spam2
    assert fb.spam is not spam2
    assert_allclose(fb.spam, spam2)
    fb.spam = None
    assert fb.spam is None


@attr.s(auto_attribs=True, slots=True, on_setattr=attr.setters.validate)
class Spam:
    """Just a silly class for testing validate_shape."""

    egg0: np.ndarray = attr.ib(validator=validate_shape(1, None, None))
    egg1: np.ndarray = attr.ib(validator=validate_shape(("egg0", 2), ("egg2", 1)))
    egg2: np.ndarray = attr.ib(validator=validate_shape(2, ("egg1", 1)))
    egg3: np.ndarray = attr.ib(validator=validate_shape(("leg", 0)))
    leg: str = attr.ib(validator=validate_shape(("egg3", 0)))


def test_validate_shape_init():
    # Construct a Spam instance with valid arguments. This should just work
    spam = Spam(
        np.zeros((1, 7, 4)), np.zeros((4, 3)), np.zeros((2, 3)), np.zeros(5), "abcde"
    )
    # Double check
    attr.validate(spam)
    # Call constructor with invalid arguments
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7, 4)),
            np.zeros((4, 3)),
            np.zeros((2, 3)),
            np.zeros(5),
            "abcde",
        )
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7)),
            np.zeros((4, 3)),
            np.zeros((2, 3)),
            np.zeros(5),
            "abcde",
        )
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7, 4)),
            np.zeros((1, 3)),
            np.zeros((2, 3)),
            np.zeros(5),
            "abcde",
        )
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7, 4)),
            np.zeros((4, 9)),
            np.zeros((2, 3)),
            np.zeros(5),
            "abcde",
        )
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7, 4)),
            np.zeros((4, 3)),
            np.zeros((2, 3)),
            np.zeros(5),
            "abcde",
        )
    with pytest.raises(TypeError):
        _ = Spam(
            np.zeros((2, 7, 4)),
            np.zeros((4, 3)),
            np.zeros((2, 3)),
            np.zeros(5),
            4,
            "abcd",
        )


def test_validate_shape_assign():
    # Construct a Spam instance with valid arguments. This should just work
    spam = Spam(
        np.zeros((1, 7, 4)), np.zeros((4, 3)), np.zeros((2, 3)), np.zeros(5), "abcde"
    )
    # Double check
    attr.validate(spam)
    # assign invalid attributes
    with pytest.raises(TypeError):
        spam.egg0 = np.zeros((2, 7, 4))
    with pytest.raises(TypeError):
        spam.egg0 = np.zeros((2, 7))
    with pytest.raises(TypeError):
        spam.egg1 = np.zeros((1, 3))
    with pytest.raises(TypeError):
        spam.egg1 = np.zeros((4, 9))
    with pytest.raises(TypeError):
        spam.leg = "abcd"


@attr.s(slots=True)
class NoName0:
    """Test exception in validate_shape: unsupported item in shape_requirements."""

    xxx: str = attr.ib(validator=validate_shape(["asdfsa", 3]))


@attr.s(slots=True)
class NoName1:
    """Test exception in validate_shape: unsupported item in shape_requirements."""

    xxx: str = attr.ib(validator=validate_shape(("asdfsa",)))


@attr.s(slots=True)
class NoName2:
    """Test exception in validate_shape: other doest not exist."""

    xxx: str = attr.ib(validator=validate_shape("other"))


@attr.s(slots=True)
class NoName3:
    """Test exception in validate_shape: other is not an array."""

    xxx: str = attr.ib(validator=validate_shape(("other", 1)))
    other = attr.ib()


def test_validate_shape_exceptions():
    with pytest.raises(ValueError):
        _ = NoName0("aaa")
    with pytest.raises(ValueError):
        _ = NoName1("aaa")
    with pytest.raises(AttributeError):
        _ = NoName2("aaa")
    with pytest.raises(TypeError):
        _ = NoName3("aaa", None)
