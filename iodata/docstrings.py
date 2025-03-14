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
"""Docstring decorators for file format implementations."""

from typing import Callable, Optional

__all__ = (
    "document_dump_many",
    "document_dump_one",
    "document_load_many",
    "document_load_one",
    "document_write_input",
)


def _document_load(
    template: str,
    fmt: str,
    guaranteed: list[str],
    ifpresent: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    if kwdocs is None:
        kwdocs = {}
    notes = "" if notes is None else f"\nNotes\n-----\n\n{notes.strip()}\n"
    ifpresent = ifpresent or []

    def decorator(func):
        if ifpresent:
            ifpresent_sentence = " The following may be loaded if present in the file: {}.".format(
                ", ".join(f"``{word}``" for word in ifpresent)
            )
        else:
            ifpresent_sentence = ""
        func.__doc__ = template.format(
            fmt=fmt,
            guaranteed=", ".join(f"``{word}``" for word in guaranteed),
            ifpresent=ifpresent_sentence,
            kwdocs="\n".join(
                "{}\n    {}".format(name, docu.replace("\n", " "))
                for name, docu in sorted(kwdocs.items())
            ),
            notes=(notes or ""),
        )
        func.fmt = fmt
        func.guaranteed = guaranteed
        func.ifpresent = ifpresent
        func.kwdocs = kwdocs
        func.notes = notes
        return func

    return decorator


LOAD_ONE_DOC_TEMPLATE = """\
Load a single frame from a {fmt} file.

Parameters
----------
lit
    The line iterator to read the data from.
{kwdocs}

Returns
-------
result: dict
    A dictionary with IOData attributes. The following attributes are guaranteed to be
    loaded: {guaranteed}.{ifpresent}
{notes}
"""


def document_load_one(
    fmt: str,
    guaranteed: list[str],
    ifpresent: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    """Decorate a load_one function to generate a docstring.

    Parameters
    ----------
    fmt
        The name of the file format.
    guaranteed
        A list of IOData attributes this format can certainly read.
    ifpresent
        A list of IOData attributes this format reads of present in the file.
    kwdocs
        A dictionary with documentation for keyword arguments. Each key is a
        keyword argument name and the corresponding value is text explaining the
        argument.
    notes
        Additional information to be added to the docstring.

    Returns
    -------
    A decorator function.

    """
    if kwdocs is None:
        kwdocs = {}
    return _document_load(LOAD_ONE_DOC_TEMPLATE, fmt, guaranteed, ifpresent, kwdocs, notes)


LOAD_MANY_DOC_TEMPLATE = """\
Load multiple frames from a {fmt} file.

Parameters
----------
lit
    The line iterator to read the data from.
{kwdocs}

Yields
------
result: dict
    A dictionary with IOData attributes. The following attribtues are guaranteed to be
    loaded: {guaranteed}.{ifpresent}
{notes}
"""


def document_load_many(
    fmt: str,
    guaranteed: list[str],
    ifpresent: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    """Decorate a load_many function to generate a docstring.

    Parameters
    ----------
    fmt
        The name of the file format.
    guaranteed
        A list of IOData attributes this format can certainly read.
    ifpresent
        A list of IOData attributes this format reads of present in the file.
    kwdocs
        A dictionary with documentation for keyword arguments. Each key is a
        keyword argument name and the corresponding value is text explaining the
        argument.
    notes
        Additional information to be added to the docstring.

    Returns
    -------
    A decorator function.

    """
    if kwdocs is None:
        kwdocs = {}
    return _document_load(LOAD_MANY_DOC_TEMPLATE, fmt, guaranteed, ifpresent, kwdocs, notes)


def _document_dump(
    template: str,
    fmt: str,
    required: list[str],
    optional: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    if kwdocs is None:
        kwdocs = {}
    optional = optional or []

    def decorator(func):
        if optional:
            optional_sentence = (
                " If the following attributes are present, they are also dumped into the file: {}."
            ).format(", ".join(f"``{word}``" for word in optional))
        else:
            optional_sentence = ""
        func.__doc__ = template.format(
            fmt=fmt,
            required=", ".join(f"``{word}``" for word in required),
            optional=optional_sentence,
            kwdocs="\n".join(
                "{}\n    {}".format(name, docu.replace("\n", " "))
                for name, docu in sorted(kwdocs.items())
            ),
            notes=(notes or ""),
        )
        func.fmt = fmt
        func.required = required
        func.optional = optional
        func.kwdocs = kwdocs
        func.notes = notes
        return func

    return decorator


DUMP_ONE_DOC_TEMPLATE = """\
Dump a single frame into a {fmt} file.

Parameters
----------
f
    A writeable file object.
data
    An IOData instance which must have the following attributes initialized:
    {required}.{optional}
{kwdocs}

Notes
-----
{notes}

"""


def document_dump_one(
    fmt: str,
    required: list[str],
    optional: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    """Decorate a dump_one function to generate a docstring.

    Parameters
    ----------
    fmt
        The name of the file format.
    required
        A list of mandatory IOData attributes needed to write the file.
    optional
        A list of optional IOData attributes which can be include when writing the file.
    kwdocs
        A dictionary with documentation for keyword arguments. Each key is a
        keyword argument name and the corresponding value is text explaining the
        argument.
    notes
        Additional information to be added to the docstring.

    Returns
    -------
    A decorator function.

    """
    if kwdocs is None:
        kwdocs = {}
    return _document_dump(DUMP_ONE_DOC_TEMPLATE, fmt, required, optional, kwdocs, notes)


DUMP_MANY_DOC_TEMPLATE = """\
Dump multiple frames into a {fmt} file.

Parameters
----------
f
    A writeable file object.
datas
    An iterator over IOData instances which must have the following attributes initialized:
    {required}.{optional}
{kwdocs}

Notes
-----
{notes}

"""


def document_dump_many(
    fmt: str,
    required: list[str],
    optional: Optional[list[str]] = None,
    kwdocs: Optional[dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    """Decorate a dump_many function to generate a docstring.

    Parameters
    ----------
    fmt
        The name of the file format.
    required
        A list of mandatory IOData attributes needed to write the file.
    optional
        A list of optional IOData attributes which can be include when writing the file.
    kwdocs
        A dictionary with documentation for keyword arguments. Each key is a
        keyword argument name and the corresponding value is text explaining the
        argument.
    notes
        Additional information to be added to the docstring.

    Returns
    -------
    A decorator function.

    """
    if kwdocs is None:
        kwdocs = {}
    return _document_dump(DUMP_MANY_DOC_TEMPLATE, fmt, required, optional, kwdocs, notes)


def _document_write(
    template: str,
    fmt: str,
    required: list[str],
    optional: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    if optional is None:
        optional = []

    def decorator(func):
        if optional:
            optional_sentence = (
                " If the following attributes are present, they are also written "
                "into the file: {}. If these attributes are not assigned, "
                "internal default values are used."
            ).format(", ".join(f"``{word}``" for word in optional))
        else:
            optional_sentence = ""
        func.__doc__ = template.format(
            fmt=fmt,
            required=", ".join(f"``{word}``" for word in required),
            optional=optional_sentence,
            notes=(notes or ""),
        )
        func.fmt = fmt
        func.required = required
        func.optional = optional
        func.notes = notes
        return func

    return decorator


WRITE_INPUT_DOC_TEMPLATE = """\
Write a {fmt} input file.

Parameters
----------
f
    A writeable file object.
data
    An IOData instance which must have the following attributes initialized:
    {required}.{optional}
template
    A template input string.
atom_line
    A function taking two arguments: an IOData instance, and an index of
    the atom. This function returns a formatted line for the corresponding
    atom. When omitted, a default atom_line function for the selected
    input format is used.
**kwargs
    Keyword arguments are passed on to the input-specific write_input function.

Notes
-----

{notes}

"""


def document_write_input(
    fmt: str,
    required: list[str],
    optional: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> Callable:
    """Decorate a write_input function to generate a docstring.

    Parameters
    ----------
    fmt
        The name of the file format.
    required
        A list of mandatory IOData attributes needed to write the file.
    optional
        A list of optional IOData attributes which can be include when writing the file.
    notes
        Additional information to be added to the docstring.

    Returns
    -------
    A decorator function.

    """
    return _document_write(WRITE_INPUT_DOC_TEMPLATE, fmt, required, optional, notes)
