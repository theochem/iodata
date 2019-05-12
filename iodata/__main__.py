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
"""CLI for file conversion."""


import argparse
import numpy as np

from .api import load_one, dump_one

try:
    from iodata.version import __version__
except ImportError:
    __version__ = '0.0.0.post0'


__all__ = []


def main():
    """Convert files between two formats using command-line arguments."""
    # All, except underflows, is *not* fine.
    np.seterr(divide='raise', over='raise', invalid='raise')

    parser = argparse.ArgumentParser(
        prog='iodata-convert',
        description='Convert between file formats supported by IOData. This '
                    'only works if the input contains sufficient data for '
                    'the output.')
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s (IOData version {})".format(__version__))
    parser.add_argument('input', help='The input file.')
    parser.add_argument('output', help='The output file.')
    args = parser.parse_args()
    data = load_one(args.input)
    dump_one(data, args.output)


if __name__ == '__main__':
    main()
