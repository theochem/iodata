#!/usr/bin/env python3
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
# pylint: disable=unused-argument,redefined-builtin
"""Generate formats.rst."""

from iodata.api import FORMAT_MODULES

__all__ = []


HEADER = """

.. _file_formats:

Supported File Formats
######################

"""


FNNAMES = ["load_one", "dump_one", "load_many", "dump_many"]


def _format_words(words):
    return ", ".join(f"``{word}``" for word in words)


def _print_section(title, linechar):
    """Print a section title with a line underneath."""
    print(title)
    print(linechar * len(title))


# pylint: disable=too-many-branches,too-many-statements
def main():
    """Write formats.rst to stdout."""
    print(HEADER)
    for modname, module in sorted(FORMAT_MODULES.items()):
        skip = True
        for fnname in FNNAMES:
            if hasattr(module, fnname):
                skip = False
                break
        if skip:
            continue
        lines = module.__doc__.split("\n")
        # add labels for cross-referencing format (e.g. in formats table)
        print(f".. _format_{modname}:")
        print()
        _print_section(f"{lines[0][:-1]} (``{modname}``)", "=")
        print()
        for line in lines[2:]:
            print(line)

        print("Filename patterns: ", _format_words(module.PATTERNS))
        print()

        for fnname in FNNAMES:
            fn = getattr(module, fnname, None)
            if fn is not None:
                _print_section(f":py:func:`iodata.formats.{modname}.{fnname}`", "-")
                if fnname.startswith("load"):
                    print("- Always loads", _format_words(fn.guaranteed))
                    if fn.ifpresent:
                        print("- May load", _format_words(fn.ifpresent))
                else:
                    print("- Requires", _format_words(fn.required))
                    if fn.optional:
                        print("- May dump", _format_words(fn.optional))
                if fn.kwdocs:
                    print("- Keyword arguments", _format_words(fn.kwdocs))
                if fn.notes:
                    print()
                    print(fn.notes)
                print()
        print()
        print()


if __name__ == "__main__":
    main()
