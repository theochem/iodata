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
def validate_shape(*shape_requirements: tuple):
    """Return a validator for the shape of an array or the length of an iterable.

    Parameters
    ----------
    shape_requirements
        Specifications for the required shape. Every item of the tuple describes
        the required size of the corresponding axis of an array. Also the
        number of items should match the dimensionality of the array. When the
        validator is used for general iterables, this tuple should contain just
        one element. Possible values for each item are explained in the "Notes"
        section below.

    Returns
    -------
    validator
        A validator function for the attr library.

    Notes
    -----
    Every element of ``shape_requirements`` defines the expected size of an
    array along the corresponding axis. An item in this tuple at position (or
    index) ``i`` can be one of the following:

    1. An integer, which is taken as the expected size along axis ``i``.
    2. None. In this case, the size of the array along axis ``i`` is not
       checked.
    3. A string, which should be the name of another integer attribute with
       the expected size along axis ``i``. The other attribute is always an
       attribute of the same object as the attribute being checked.
    4. A 2-tuple containing a name and an integer. In this case, the name refers
       to another attribute which is an array or an iterable. When the integer
       is 0, just the length of the other attribute is used. When the integer is
       non-zero, the other attribute must be an array and the integer selects an
       axis. The size of the other array along the selected axis is then used as
       the expected size of the array being checked along axis ``i``.

    """
    def validator(obj, attribute, value):
        # Build the expected shape, with the rules from the docstring.
        expected_shape = []
        for item in shape_requirements:
            if isinstance(item, int) or item is None:
                expected_shape.append(item)
            elif isinstance(item, str):
                expected_shape.append(getattr(obj, item))
            elif isinstance(item, tuple) and len(item) == 2:
                other_name, other_axis = item
                other = getattr(obj, other_name)
                if other is None:
                    raise TypeError(
                        "Other attribute '{}' is not set.".format(other_name)
                    )
                if other_axis == 0:
                    expected_shape.append(len(other))
                else:
                    if other_axis >= other.ndim or other_axis < 0:
                        raise TypeError(
                            "Cannot get length along axis "
                            "{} of attribute {} with ndim {}.".format(
                                other_axis, other_name, other.ndim
                            )
                        )
                    expected_shape.append(other.shape[other_axis])
            else:
                raise ValueError(f"Cannot interpret item in shape_requirements: {item}")
        expected_shape = tuple(expected_shape)
        # Get the actual shape
        if isinstance(value, np.ndarray):
            observed_shape = value.shape
        else:
            observed_shape = (len(value),)
        # Compare
        match = True
        if len(expected_shape) != len(observed_shape):
            match = False
        if match:
            for es, os in zip(expected_shape, observed_shape):
                if es is None:
                    continue
                if es != os:
                    match = False
                    break
        # Raise TypeError if needed.
        if not match:
            raise TypeError(
                "Expecting shape {} for attribute {}, got {}".format(
                    expected_shape, attribute.name, observed_shape
                )
            )

    return validator
