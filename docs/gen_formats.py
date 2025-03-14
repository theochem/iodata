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
"""Generate formats.rst."""

from iodata.api import FORMAT_MODULES

__all__ = ("format_words", "main", "print_section")


HEADER = """

.. _file_formats:

Supported File Formats
######################

"""


FNNAMES = ["load_one", "dump_one", "load_many", "dump_many"]


def format_words(words):
    """Apply code formatting to words."""
    return ", ".join(f"``{word}``" for word in words)


def print_section(title, linechar, file):
    """Print a section title with a line underneath."""
    print(title, file=file)
    print(linechar * len(title), file=file)


def main():
    """Write formats.rst."""
    with open("formats.rst", "w") as file:
        print(HEADER, file=file)
        for modname, module in sorted(FORMAT_MODULES.items()):
            if not any(hasattr(module, fnname) for fnname in FNNAMES):
                continue

            # add labels for cross-referencing format (e.g. in formats table)
            title_line = module.__doc__.split("\n")[0].strip()
            print(f".. _format_{modname}:", file=file)
            print(file=file)
            print_section(f"{title_line} (``{modname}``)", "=", file=file)
            print(file=file)
            print(f"See :py:mod:`iodata.formats.{modname}` for details.", file=file)
            print(file=file)
            print("Filename patterns: ", format_words(module.PATTERNS), file=file)
            print(file=file)

            for fnname in FNNAMES:
                fn = getattr(module, fnname, None)
                if fn is not None:
                    print_section(f":py:func:`iodata.formats.{modname}.{fnname}`", "-", file=file)
                    if fnname.startswith("load"):
                        print("- Always loads", format_words(fn.guaranteed), file=file)
                        if fn.ifpresent:
                            print("- May load", format_words(fn.ifpresent), file=file)
                    else:
                        print("- Requires", format_words(fn.required), file=file)
                        if fn.optional:
                            print("- May dump", format_words(fn.optional), file=file)
                    if fn.kwdocs:
                        print("- Keyword arguments", format_words(fn.kwdocs), file=file)
                    if fn.notes:
                        print(file=file)
                        print(fn.notes, file=file)
                    print(file=file)
            print(file=file)
            print(file=file)


if __name__ == "__main__":
    main()
