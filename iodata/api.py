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
from typing import Iterator
from types import ModuleType
from fnmatch import fnmatch
from pkgutil import iter_modules
from importlib import import_module

from .iodata import IOData
from .utils import LineIterator


__all__ = ['load_one', 'load_many', 'dump_one', 'dump_many', 'write_input']


def _find_format_modules():
    """Return all file-format modules found with importlib."""
    result = {}
    for module_info in iter_modules(import_module('iodata.formats').__path__):
        if not module_info.ispkg:
            format_module = import_module('iodata.formats.' + module_info.name)
            if hasattr(format_module, 'PATTERNS'):
                result[module_info.name] = format_module
    return result


FORMAT_MODULES = _find_format_modules()


def _select_format_module(filename: str, attrname: str, fmt: str = None) -> ModuleType:
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
    format_module
        The module implementing the required file format.

    """
    basename = os.path.basename(filename)
    if fmt is None:
        for format_module in FORMAT_MODULES.values():
            if any(fnmatch(basename, pattern) for pattern in format_module.PATTERNS):
                if hasattr(format_module, attrname):
                    return format_module
    else:
        return FORMAT_MODULES[fmt]
    raise ValueError('Could not find file format with feature {} for file {}'.format(
        attrname, filename))


def _find_input_modules():
    """Return all input modules found with importlib."""
    result = {}
    for module_info in iter_modules(import_module('iodata.inputs').__path__):
        if not module_info.ispkg:
            input_module = import_module('iodata.inputs.' + module_info.name)
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
    format_module
        The module implementing the required input format.

    """
    if fmt in INPUT_MODULES:
        if not hasattr(INPUT_MODULES[fmt], 'write_input'):
            raise ValueError(f'{fmt} input module does not have write_input!')
        return INPUT_MODULES[fmt]
    raise ValueError(f"Could not find input format {fmt}!")


def load_one(filename: str, fmt: str = None, **kwargs) -> IOData:
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
    out
        The instance of IOData with data loaded from the input files.

    """
    format_module = _select_format_module(filename, 'load_one', fmt)
    lit = LineIterator(filename)
    try:
        iodata = IOData(**format_module.load_one(lit, **kwargs))
    except StopIteration:
        lit.error("File ended before all data was read.")
    return iodata


def load_many(filename: str, fmt: str = None, **kwargs) -> Iterator[IOData]:
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
    out
        An instance of IOData with data for one frame loaded for the file.

    """
    format_module = _select_format_module(filename, 'load_many', fmt)
    lit = LineIterator(filename)
    for data in format_module.load_many(lit, **kwargs):
        try:
            yield IOData(**data)
        except StopIteration:
            return


def dump_one(iodata: IOData, filename: str, fmt: str = None, **kwargs):
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

    """
    format_module = _select_format_module(filename, 'dump_one', fmt)
    with open(filename, 'w') as f:
        format_module.dump_one(f, iodata, **kwargs)


def dump_many(iodatas: Iterator[IOData], filename: str, fmt: str = None, **kwargs):
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

    """
    format_module = _select_format_module(filename, 'dump_many', fmt)
    with open(filename, 'w') as f:
        format_module.dump_many(f, iodatas, **kwargs)


def write_input(iodata: IOData, filename: str, fmt: str, template: str = None, **kwargs):
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
    **kwargs
        Keyword arguments are passed on to the input-specific write_input function.

    """
    input_module = _select_input_module(fmt)
    with open(filename, 'w') as f:
        input_module.write_input(f, iodata, template=template, **kwargs)
