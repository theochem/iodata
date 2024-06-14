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
"""Generate inputs.rst."""

from gen_formats import format_words, print_section

from iodata.api import INPUT_MODULES

__all__ = ("main",)


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
    """Write inputs.rst."""
    with open("inputs.rst", "w") as file:
        print(HEADER, file=file)
        for modname, module in sorted(INPUT_MODULES.items()):
            if not hasattr(module, "write_input"):
                continue

            # add labels for cross-referencing format (e.g. in formats table)
            lines = module.__doc__.split("\n")
            print(f".. _input_{modname}:", file=file)
            print(file=file)
            print_section(f"{lines[0][:-1]} (``{modname}``)", "=", file=file)
            print(file=file)
            for line in lines[2:]:
                print(line, file=file)

            print_section(f":py:func:`iodata.inputs.{modname}.write_input`", "-", file=file)
            fn = getattr(module, "write_input", None)
            print("- Requires", format_words(fn.required), file=file)
            if fn.optional:
                print("- May use", format_words(fn.optional), file=file)
            if fn.notes:
                print(file=file)
                print(fn.notes, file=file)
            print(file=file)
            template = getattr(module, "default_template", None)
            if template:
                code_block_lines = ["    " + ell for ell in template.split("\n")]
                print(TEMPLATE.format(code_block_lines="\n".join(code_block_lines)), file=file)
            print(file=file)
            print(file=file)


if __name__ == "__main__":
    main()
