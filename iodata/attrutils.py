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
"""Utilities for building attr classes."""


import numpy as np


__all__ = ["convert_array_to", "validate_shape"]


def convert_array_to(dtype):
    """Return a function to convert arrays to the given type."""
    def converter(array):
        if array is None:
            return None
        return np.array(array, copy=False, dtype=dtype)
    return converter


# pylint: disable=too-many-branches
def validate_shape(*shape):
    """Return a validator for the shape of an array or the length of an iterable.

    Parameters
    ----------
    shape
        The expected shape. For general iterables, this should be just one
        element. For arrays, also ``ndim`` is compared to the length of the
        given shape.

    Returns
    -------
    validator
        A validator function for the attr library

    Notes
    -----
    Each element of shape can be one of the following four things and should
    match the shape of the attribute being validated:

    1. An integer. In this case, this is the expected length along this axis.
    2. None. In this case, the length of the array along this axis is not checked.
    3. A string. In this case, it is the name of an integer attribute with the
       expected length.
    4. A tuple of a name and an integer. In this case, the name refers to another
       attribute which is an array or some iterable. When the integer is 0,
       just the length of the attribute is used.

    """
    def validator(obj, attribute, value):
        # Build the expected shape, with the rules from the docstring.
        expected_shape = []
        for item in shape:
            if isinstance(item, int) or item is None:
                expected_shape.append(item)
            elif isinstance(item, str):
                expected_shape.append(getattr(obj, item))
            elif isinstance(item, tuple) and len(item) == 2:
                other_name, other_axis = item
                other = getattr(obj, other_name)
                if other is None:
                    raise TypeError("Other attribute {} is not set.".format(other_name))
                if other_axis == 0:
                    expected_shape.append(len(other))
                else:
                    if other_axis >= other.ndim:
                        raise TypeError(
                            "Cannot get length along axis "
                            "{} of attribute {} with ndim {}.".format(
                                other_axis, other_name, other.ndim
                            )
                        )
                    expected_shape.append(other.shape[other_axis])
        # Get the actual shape
        if isinstance(value, np.ndarray):
            observed_shape = value.shape
        else:
            observed_shape = (len(value),)
        # Compare
        if len(expected_shape) != len(observed_shape):
            raise TypeError('Expect ndim {} for attribute {}, got {}'.format(
                len(expected_shape), attribute.name, len(observed_shape)))
        for axis, (es, os) in enumerate(zip(expected_shape, observed_shape)):
            if es is None:
                continue
            if es != os:
                raise TypeError(
                    'Expect size {} for axis {} of attribute {}, got {}'.format(
                        es, axis, attribute.name, os))
    return validator
