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
"""Preparation of IOData instances before they are dumped.

The ``prepare_*`` functions below can be used as building blocks in ``prepare_dump`` functions.
When the ``allow_changes`` argument is set to ``True``,
they can return a new IOData instance with some modified attributes.
Otherwise, they can raise an error if a conversion is needed.
When no conversion is needed, no errors or warnings are raised,
and the same IOData object is returned.
"""

from warnings import warn

import attrs

from .convert import convert_to_segmented, convert_to_unrestricted
from .iodata import IOData
from .utils import PrepareDumpError, PrepareDumpWarning

__all__ = ("prepare_segmented", "prepare_unrestricted_aminusb")


def prepare_unrestricted_aminusb(data: IOData, allow_changes: bool, filename: str, fmt: str):
    """If molecular orbitals have aminusb set, they are converted to unrestricted.

    Parameters
    ----------
    data
        The IOData instance with the molecular orbitals.
    allow_changes
        Whether conversion of the IOData object to a compatible form is allowed or not.
    filename
        The file to be written to, only used for error messages.
    fmt
        The file format whose dump function is calling this function, only used for error messages.

    Returns
    -------
    data
        The given data object if no conversion took place,
        or a shallow copy with some new attributes.

    Raises
    ------
    ValueError
        If the given data object has no molecular orbitals,
        or if the orbitals are generalized.
    PrepareDumpError
        If ``allow_changes == False`` and a conversion is required.
    PrepareDumpWarning
        If ``allow_changes == True`` and a conversion is required.

    """
    # Check: possible, needed?
    if data.mo is None:
        raise ValueError("The given IOData instance has no molecular orbitals.")
    if data.mo.kind == "generalized":
        raise ValueError("prepare_unrestricted_aminusb is not applicable to generalized orbitals.")
    if data.mo.kind == "unrestricted":
        return data
    if data.mo.occs_aminusb is None:
        return data

    # Raise error or warning
    message = f"The {fmt} format does not support restricted orbitals with mo.occs_aminusb. "
    if not allow_changes:
        raise PrepareDumpError(
            message + "Set allow_changes to enable conversion to unrestricted.", filename
        )
    warn(
        PrepareDumpWarning(message + "The orbitals are converted to unrestricted", filename),
        stacklevel=2,
    )

    # Convert
    return attrs.evolve(data, mo=convert_to_unrestricted(data.mo))


def prepare_segmented(data: IOData, keep_sp: bool, allow_changes: bool, filename: str, fmt: str):
    """If needed, convert generalized contractions to segmented ones.

    Parameters
    ----------
    data
        The IOData instance with the orbital basis set.
    keep_sp
        Set to True if SP-shells should not be segmented.
    allow_changes
        Whether conversion of the IOData object to a compatible form is allowed or not.
    filename
        The file to be written to, only used for error messages.
    fmt
        The file format whose dump function is calling this function, only used for error messages.

    Returns
    -------
    data
        The given data object if no conversion took place,
        or a shallow copy with some new attributes.

    Raises
    ------
    ValueError
        If the given data object has no orbital basis set.
    PrepareDumpError
        If ``allow_changes == False`` and a conversion is required.
    PrepareDumpWarning
        If ``allow_changes == True`` and a conversion is required.
    """
    # Check: possible, needed?
    if data.obasis is None:
        raise ValueError("The given IOData instance has no orbital basis set.")
    if all(
        shell.ncon == 1 or (keep_sp and shell.ncon == 2 and (shell.angmoms == [0, 1]).all())
        for shell in data.obasis.shells
    ):
        return data

    # Raise error or warning
    message = f"The {fmt} format does not support generalized contractions"
    if keep_sp:
        message += " other than SP shells"
    message += ". "
    if not allow_changes:
        raise PrepareDumpError(
            message + "Set allow_changes to enable conversion to segmented shells.", filename
        )
    warn(
        PrepareDumpWarning(
            message + "The orbital basis is converted to segmented shells", filename
        ),
        stacklevel=2,
    )

    # Convert
    return attrs.evolve(data, obasis=convert_to_segmented(data.obasis, keep_sp))
