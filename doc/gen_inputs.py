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

from gen_formats import _format_words, _print_section

from iodata.api import INPUT_MODULES

__all__ = []


HEADER = """

.. _input_formats:

Supported Input Formats
#######################

"""

TEMPLATE = """
Default Template
----------------
.. code-block:: python

    '''\\
{code_block_lines}
    '''
"""


def main():
    """Write inputs.rst to stdout."""
    print(HEADER)
    for modname, module in sorted(INPUT_MODULES.items()):
        if not hasattr(module, "write_input"):
            continue
        lines = module.__doc__.split("\n")
        # add labels for cross-referencing format (e.g. in formats table)
        print(f".. _input_{modname}:")
        print()
        _print_section(f"{lines[0][:-1]} (``{modname}``)", "=")
        print()
        for line in lines[2:]:
            print(line)

        _print_section(f":py:func:`iodata.formats.{modname}.write_input`", "-")
        fn = getattr(module, "write_input", None)
        print("- Requires", _format_words(fn.required))
        if fn.optional:
            print("- May use", _format_words(fn.optional))
        if fn.kwdocs:
            print("- Keyword arguments", _format_words(fn.kwdocs))
        if fn.notes:
            print()
            print(fn.notes)
        print()
        template = getattr(module, "default_template", None)
        if template:
            code_block_lines = ["    " + ell for ell in template.split("\n")]
            print(TEMPLATE.format(code_block_lines="\n".join(code_block_lines)))
        print()
        print()


if __name__ == "__main__":
    main()
