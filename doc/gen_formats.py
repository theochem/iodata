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


def _format_words(words):
    return ', '.join('``{}``'.format(word) for word in words)


# pylint: disable=too-many-branches,too-many-statements
def main():
    """Write formats.rst to stdout."""
    print(HEADER)
    for name, module in sorted(FORMAT_MODULES.items()):
        load_one = getattr(module, 'load_one', None)
        dump_one = getattr(module, 'dump_one', None)
        load_many = getattr(module, 'load_many', None)
        dump_many = getattr(module, 'dump_many', None)

        if not (load_one or dump_one or load_many or dump_many):
            continue

        lines = module.__doc__.split('\n')
        print(lines[0][:-1], '(``{}``)'.format(name))
        print('=' * (len(name) + len(lines[0][:-1]) + 7))
        print()
        for line in lines[2:]:
            print(line)

        print("Filename patterns: ", _format_words(module.PATTERNS))
        print()


        if load_one is not None:
            print("load_one")
            print("--------")
            print("- Always loads", _format_words(load_one.guaranteed))
            if load_one.ifpresent:
                print("- May load", _format_words(load_one.ifpresent))
            print()
            if load_one.notes:
                print(load_one.notes)
            print()

        if dump_one is not None:
            print("dump_one")
            print("--------")
            print("- Requires", _format_words(dump_one.required))
            if dump_one.optional:
                print("- May dump", _format_words(dump_one.optional))
            print()
            if dump_one.notes:
                print(dump_one.notes)
            print()

        if load_many is not None:
            print("load_many")
            print("---------")
            print("- Always loads", _format_words(load_many.guaranteed))
            if load_many.ifpresent:
                print("- May load", _format_words(load_many.ifpresent))
            print()
            if load_many.notes:
                print(load_many.notes)
            print()

        if dump_many is not None:
            print("dump_many")
            print("---------")
            print("- Requires", _format_words(dump_many.required))
            if dump_many.optional:
                print("- May dump", _format_words(dump_many.optional))
            print()
            if dump_many.notes:
                print(dump_many.notes)
            print()

        print()
        print()


if __name__ == '__main__':
    main()
