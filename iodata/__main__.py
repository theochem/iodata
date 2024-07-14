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
from typing import Optional

import numpy as np

from .api import FORMAT_MODULES, dump_many, dump_one, load_many, load_one

try:
    from iodata.version import __version__
except ImportError:
    __version__ = "0.0.0.post0"


__all__ = ("convert",)


DESCRIPTION = """\
Convert between file formats supported by IOData. This only works if the input
contains sufficient data for the output.

List of supported formats:

load_one
    {load_one}
dump_one
    {dump_one}
load_many
    {load_many}
dump_many
    {dump_many}
""".format(
    load_one=" ".join(
        name for name, module in sorted(FORMAT_MODULES.items()) if hasattr(module, "load_one")
    ),
    dump_one=" ".join(
        name for name, module in sorted(FORMAT_MODULES.items()) if hasattr(module, "dump_one")
    ),
    load_many=" ".join(
        name for name, module in sorted(FORMAT_MODULES.items()) if hasattr(module, "load_many")
    ),
    dump_many=" ".join(
        name for name, module in sorted(FORMAT_MODULES.items()) if hasattr(module, "dump_many")
    ),
)


def parse_args():
    """Use argparse to to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="iodata-convert",
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s (IOData version {__version__})",
    )
    parser.add_argument(
        "-i", "--infmt", help="Select the input format, overrides automatic detection."
    )
    parser.add_argument(
        "-o", "--outfmt", help="Select the output format, overrides automatic detection."
    )
    parser.add_argument(
        "-c",
        "--allow-changes",
        default=False,
        action="store_true",
        help="Allow (not trivially reversible) conversion of the input data to make it compatible "
        "with the output format. Warnings will be emitted for all changes made.",
    )
    parser.add_argument(
        "-m",
        "--many",
        default=False,
        action="store_true",
        help="Convert many frames, e.g. for trajectories.",
    )
    parser.add_argument("input", help="The input file.")
    parser.add_argument("output", help="The output file.")
    return parser.parse_args()


def convert(
    infn: str,
    outfn: str,
    many: bool = False,
    infmt: Optional[str] = None,
    outfmt: Optional[str] = None,
    allow_changes: bool = False,
):
    """Convert file from one format to another.

    Parameters
    ----------
    infn
        The input file name.
    outfn
        The output file name.
    many
        When True, multpile frames are converted.
    infmt
        The input format.
    outfmt
        The output format.
    allow_changes
        Allow prepare_dump functions to modify the data
        to make it compatible with the output format.

    """
    if many:
        dump_many(load_many(infn, fmt=infmt), outfn, allow_changes=allow_changes, fmt=outfmt)
    else:
        dump_one(load_one(infn, fmt=infmt), outfn, allow_changes=allow_changes, fmt=outfmt)


def main():
    """Convert files between two formats using command-line arguments."""
    # All, except underflows, is *not* fine.
    np.seterr(divide="raise", over="raise", invalid="raise")

    args = parse_args()
    convert(args.input, args.output, args.many, args.infmt, args.outfmt, args.allow_changes)


if __name__ == "__main__":
    main()
