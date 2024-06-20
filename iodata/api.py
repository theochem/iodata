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
"""Functions to be used by end users."""

import os
from collections.abc import Iterable, Iterator
from fnmatch import fnmatch
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Callable, Optional

from .iodata import IOData
from .utils import FileFormatError, LineIterator, PrepareDumpError

__all__ = ["load_one", "load_many", "dump_one", "dump_many", "write_input"]


def _find_format_modules():
    """Return all file-format modules found with importlib."""
    result = {}
    for module_info in iter_modules(import_module("iodata.formats").__path__):
        if not module_info.ispkg:
            format_module = import_module("iodata.formats." + module_info.name)
            if hasattr(format_module, "PATTERNS"):
                result[module_info.name] = format_module
    return result


FORMAT_MODULES = _find_format_modules()


def _select_format_module(filename: str, attrname: str, fmt: Optional[str] = None) -> ModuleType:
    """Find a file format module with the requested attribute name.

    Parameters
    ----------
    filename
        The file to load or dump.
    attrname
        The required atrtibute of the file format module.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.

    Returns
    -------
    The module implementing the required file format.

    """
    basename = os.path.basename(filename)
    if fmt is None:
        for format_module in FORMAT_MODULES.values():
            if any(fnmatch(basename, pattern) for pattern in format_module.PATTERNS) and hasattr(
                format_module, attrname
            ):
                return format_module
    else:
        return FORMAT_MODULES[fmt]
    raise ValueError(f"Could not find file format with feature {attrname} for file {filename}")


def _find_input_modules():
    """Return all input modules found with importlib."""
    result = {}
    for module_info in iter_modules(import_module("iodata.inputs").__path__):
        if not module_info.ispkg:
            input_module = import_module("iodata.inputs." + module_info.name)
            if hasattr(input_module, "write_input"):
                result[module_info.name] = input_module
    return result


INPUT_MODULES = _find_input_modules()


def _select_input_module(fmt: str) -> ModuleType:
    """Find an input module.

    Parameters
    ----------
    fmt
        The name of the input module to use.

    Returns
    -------
    The module implementing the required input format.

    """
    if fmt in INPUT_MODULES:
        if not hasattr(INPUT_MODULES[fmt], "write_input"):
            raise ValueError(f"{fmt} input module does not have write_input!")
        return INPUT_MODULES[fmt]
    raise ValueError(f"Could not find input format {fmt}!")


def load_one(filename: str, fmt: Optional[str] = None, **kwargs) -> IOData:
    """Load data from a file.

    This function uses the extension or prefix of the filename to determine the
    file format. When the file format is detected, a specialized load function
    is called for the heavy lifting.

    Parameters
    ----------
    filename
        The file to load data from.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.
    **kwargs
        Keyword arguments are passed on to the format-specific load_one function.

    Returns
    -------
    The instance of IOData with data loaded from the input files.

    """
    format_module = _select_format_module(filename, "load_one", fmt)
    with LineIterator(filename) as lit:
        try:
            iodata = IOData(**format_module.load_one(lit, **kwargs))
        except StopIteration:
            lit.error("File ended before all data was read.")
    return iodata


def load_many(filename: str, fmt: Optional[str] = None, **kwargs) -> Iterator[IOData]:
    """Load multiple IOData instances from a file.

    This function uses the extension or prefix of the filename to determine the
    file format. When the file format is detected, a specialized load function
    is called for the heavy lifting.

    Parameters
    ----------
    filename
        The file to load data from.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.
    **kwargs
        Keyword arguments are passed on to the format-specific load_many function.

    Yields
    ------
    IOData
        An instance of IOData with data for one frame loaded for the file.

    """
    format_module = _select_format_module(filename, "load_many", fmt)
    with LineIterator(filename) as lit:
        try:
            for data in format_module.load_many(lit, **kwargs):
                yield IOData(**data)
        except StopIteration:
            return


def _check_required(iodata: IOData, dump_func: Callable):
    """Check that required attributes are not None before dumping to a file.

    Parameters
    ----------
    iodata
        The data to be written.
    dump_func
        The dump_one or dump_many function that will write the file.

    Raises
    ------
    PrepareDumpError
        When a required attribute is ``None``.
    """
    for attr_name in dump_func.required:
        if getattr(iodata, attr_name) is None:
            raise PrepareDumpError(
                f"Required attribute {attr_name}, for format {dump_func.fmt}, is None."
            )


def dump_one(iodata: IOData, filename: str, fmt: Optional[str] = None, **kwargs):
    """Write data to a file.

    This routine uses the extension or prefix of the filename to determine
    the file format. For each file format, a specialized function is
    called that does the real work.

    Parameters
    ----------
    iodata
        The object containing the data to be written.
    filename
        The file to write the data to.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.
    **kwargs
        Keyword arguments are passed on to the format-specific dump_one function.

    Raises
    ------
    PrepareDumpError
        When the iodata object is not compatible with the file format,
        e.g. due to missing attributes, and not conversion is available or allowed
        to make it compatible.
    """
    format_module = _select_format_module(filename, "dump_one", fmt)
    _check_required(iodata, format_module.dump_one)
    if hasattr(format_module, "prepare_dump"):
        format_module.prepare_dump(iodata)
    with open(filename, "w") as f:
        format_module.dump_one(f, iodata, **kwargs)


def dump_many(iodatas: Iterable[IOData], filename: str, fmt: Optional[str] = None, **kwargs):
    """Write multiple IOData instances to a file.

    This routine uses the extension or prefix of the filename to determine
    the file format. For each file format, a specialized function is
    called that does the real work.

    Parameters
    ----------
    iodatas
        An iterator over IOData instances.
    filename
        The file to write the data to.
    fmt
        The name of the file format module to use.
    **kwargs
        Keyword arguments are passed on to the format-specific dump_many function.

    Raises
    ------
    PrepareDumpError
        When the iodata object is not compatible with the file format,
        e.g. due to missing attributes, and not conversion is available or allowed
        to make it compatible.
    """
    format_module = _select_format_module(filename, "dump_many", fmt)

    # Check the first item before creating the file.
    # If the file already exists, this may prevent data loss:
    # The file is not overwritten when it is clear that writing will fail.
    iter_iodatas = iter(iodatas)
    try:
        first = next(iter_iodatas)
        _check_required(first, format_module.dump_many)
    except StopIteration as exc:
        raise FileFormatError("dump_many needs at least one iodata object.") from exc

    def checking_iterator():
        """Iterate over all iodata items, not checking the first."""
        # The first one was already checked.
        yield first
        for other in iter_iodatas:
            _check_required(other, format_module.dump_many)
            if hasattr(format_module, "prepare_dump"):
                format_module.prepare_dump(other)
            yield other

    with open(filename, "w") as f:
        format_module.dump_many(f, checking_iterator(), **kwargs)


def write_input(
    iodata: IOData,
    filename: str,
    fmt: str,
    template: Optional[str] = None,
    atom_line: Optional[Callable] = None,
    **kwargs,
):
    """Write input file using an instance of IOData for the specified software format.

    Parameters
    ----------
    iodata
        An IOData instance containing the information needed to write input.
    filename
        The input file name.
    fmt
        The name of the software for which input file is generated.
    template
        The template input string.
        If not given, a default template for the selected software is used.
    atom_line
        A function taking two arguments: an IOData instance, and an index of
        the atom. This function returns a formatted line for the corresponding
        atom. When omitted, a default atom_line function for the selected
        input format is used.
    **kwargs
        Keyword arguments are passed on to the input-specific write_input function.

    """
    input_module = _select_input_module(fmt)
    with open(filename, "w") as fh:
        input_module.write_input(fh, iodata, template, atom_line, **kwargs)
