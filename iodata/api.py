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
import warnings
from collections.abc import Iterable, Iterator
from fnmatch import fnmatch
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Callable, Optional

from .iodata import IOData
from .utils import (
    DumpError,
    FileFormatError,
    LineIterator,
    LoadError,
    PrepareDumpError,
    WriteInputError,
)

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
        The required attribute of the file format module.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.

    Returns
    -------
    The module implementing the required file format.

    Raises
    ------
    FileFormatError
        When no file format module can be found that has a member named ``attrname``.
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
    raise FileFormatError(f"Cannot find file format with feature {attrname}", filename)


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


def _select_input_module(filename: str, fmt: str) -> ModuleType:
    """Find an input module.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    fmt
        The name of the input module to use.

    Returns
    -------
    The module implementing the required input format.


    Raises
    ------
    FileFormatError
        When the format ``fmt`` does not exist.
    """
    if fmt in INPUT_MODULES:
        if not hasattr(INPUT_MODULES[fmt], "write_input"):
            raise FileFormatError(f"{fmt} input module does not have write_input.", filename)
        return INPUT_MODULES[fmt]
    raise FileFormatError(f"Cannot find input format {fmt}.", filename)


def _reissue_warnings(func):
    """Correct stacklevel of warnings raised in functions called deeper in IOData.

    This function should be used as a decorator of end-user API functions.
    Adapted from https://stackoverflow.com/a/71635963/494584
    """

    def inner(*args, **kwargs):
        """Wrapper for func that reissues warnings."""
        warning_list = []
        try:
            with warnings.catch_warnings(record=True) as warning_list:
                result = func(*args, **kwargs)
        finally:
            for warning in warning_list:
                warnings.warn(warning.message, warning.category, stacklevel=2)
        return result

    return inner


@_reissue_warnings
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
            return IOData(**format_module.load_one(lit, **kwargs))
        except LoadError:
            raise
        except StopIteration as exc:
            raise LoadError("File ended before all data was read.", lit) from exc
        except Exception as exc:
            raise LoadError("Uncaught exception while loading file.", lit) from exc


@_reissue_warnings
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
        except LoadError:
            raise
        except Exception as exc:
            raise LoadError("Uncaught exception while loading file.", lit) from exc


def _check_required(filename: str, iodata: IOData, dump_func: Callable):
    """Check that required attributes are not None before dumping to a file.

    Parameters
    ----------
    filename
        The file to be dumped to, only used for error messages.
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
                f"Required attribute {attr_name}, for format {dump_func.fmt}, is None.", filename
            )


@_reissue_warnings
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
    DumpError
        When an error is encountered while dumping to a file.
        If the output file already existed, it is (partially) overwritten.
    PrepareDumpError
        When the iodata object is not compatible with the file format,
        e.g. due to missing attributes, and not conversion is available or allowed
        to make it compatible.
        If the output file already existed, it is not overwritten.
    """
    format_module = _select_format_module(filename, "dump_one", fmt)
    try:
        _check_required(filename, iodata, format_module.dump_one)
        if hasattr(format_module, "prepare_dump"):
            format_module.prepare_dump(filename, iodata)
    except PrepareDumpError:
        raise
    except Exception as exc:
        raise PrepareDumpError(
            "Uncaught exception while preparing for dumping to a file.", filename
        ) from exc
    with open(filename, "w") as f:
        try:
            format_module.dump_one(f, iodata, **kwargs)
        except DumpError:
            raise
        except Exception as exc:
            raise DumpError("Uncaught exception while dumping to a file", filename) from exc


@_reissue_warnings
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
    DumpError
        When an error is encountered while dumping to a file.
        If the output file already existed, it (partially) overwritten.
    PrepareDumpError
        When the iodata object is not compatible with the file format,
        e.g. due to missing attributes, and not conversion is available or allowed
        to make it compatible.
        If the output file already existed, it is not overwritten when this error
        is raised while processing the first IOData instance in the ``iodatas`` argument.
        When the exception is raised in later iterations, any existing file is overwritten.
    """
    format_module = _select_format_module(filename, "dump_many", fmt)

    # Check the first item before creating the file.
    # If the file already exists, this may prevent data loss:
    # The file is not overwritten when it is clear that writing will fail.
    iter_iodatas = iter(iodatas)
    try:
        first = next(iter_iodatas)
    except StopIteration as exc:
        raise DumpError("dump_many needs at least one iodata object.", filename) from exc
    try:
        _check_required(filename, first, format_module.dump_many)
        if hasattr(format_module, "prepare_dump"):
            format_module.prepare_dump(filename, first)
    except PrepareDumpError:
        raise
    except Exception as exc:
        raise PrepareDumpError(
            "Uncaught exception while preparing for dumping to a file.", filename
        ) from exc

    def checking_iterator():
        """Iterate over all iodata items, not checking the first."""
        # The first one was already checked.
        yield first
        for other in iter_iodatas:
            _check_required(filename, other, format_module.dump_many)
            if hasattr(format_module, "prepare_dump"):
                format_module.prepare_dump(filename, other)
            yield other

    with open(filename, "w") as f:
        try:
            format_module.dump_many(f, checking_iterator(), **kwargs)
        except (PrepareDumpError, DumpError):
            raise
        except Exception as exc:
            raise DumpError("Uncaught exception while dumping to a file.", filename) from exc


@_reissue_warnings
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
    input_module = _select_input_module(filename, fmt)
    with open(filename, "w") as fh:
        try:
            input_module.write_input(fh, iodata, template, atom_line, **kwargs)
        except Exception as exc:
            raise WriteInputError(
                "Uncaught exception while writing an input file.", filename
            ) from exc
