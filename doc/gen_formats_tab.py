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

import inspect
from collections import defaultdict

import iodata

__all__ = []


def _generate_all_format_parser():
    """Parse supported functionality from each module.

    Returns
    -------
    tuple(list, set, dict, dict, set, dict, dict)
        list[fmt_name]
        set{fmt_names_with_load_one}
        dict{attr_name: fmt_names_guaranteed}
        dict{attr_name: fmt_names_ifpresent}
        set{fmt_names_with_dump_one}
        dict{attr_name: fmt_names_required}
        dict{attr_name: fmt_names_optional}

    """
    # Inspect iodata format module.
    # Obtain a list of tuple [(module_name: str, module_object: obj)].
    format_modules = inspect.getmembers(iodata.formats, inspect.ismodule)
    # Store supported format names in a list.
    fmt_names = []
    # Store which formats support each attribute.
    has_load = set()
    guaranteed = defaultdict(list)
    ifpresent = defaultdict(list)
    has_dump = set()
    required = defaultdict(list)
    optional = defaultdict(list)
    for fmt_name, fmt_module in format_modules:
        # Add new format name to fmt_names.
        if fmt_name not in fmt_names:
            fmt_names.append(fmt_name)
        # Obtain supported attributes.
        if hasattr(fmt_module, "load_one"):
            has_load.add(fmt_name)
            for attr_name in fmt_module.load_one.guaranteed:
                guaranteed[attr_name].append(fmt_name)
            for attr_name in fmt_module.load_one.ifpresent:
                ifpresent[attr_name].append(fmt_name)
        if hasattr(fmt_module, "dump_one"):
            has_dump.add(fmt_name)
            for attr_name in fmt_module.dump_one.required:
                required[attr_name].append(fmt_name)
            for attr_name in fmt_module.dump_one.optional:
                optional[attr_name].append(fmt_name)
    return fmt_names, has_load, guaranteed, ifpresent, has_dump, required, optional


def generate_table_rst():
    """Construct table contents.

    Returns
    -------
    list[list[]]
        table with rows of each property and columns of each format

    """
    fmt_names, has_load, guaranteed, ifpresent, has_dump, required, optional = (
        _generate_all_format_parser()
    )

    # Sort rows by number of times the attribute is used in decreasing order.
    rows = sorted(attr_name for attr_name in dir(iodata.IOData) if not attr_name.startswith("_"))

    # Order columns based on number of guaranteed and ifpresent entries for each format.
    # Also keep track of which format has a load_one and dump_one function.
    cols = []
    for fmt_name in fmt_names:
        count = sum((fmt_name in value) for value in guaranteed.values())
        count += sum((fmt_name in value) for value in ifpresent.values())
        cols.append((count, fmt_name))
    cols = [fmt_name[1] for fmt_name in sorted(cols, reverse=True)]

    # Construct header with cross-referencing columns.
    header = ["Attribute"]
    for fmt_name in cols:
        col_name = f":ref:`{fmt_name} <format_{fmt_name}>`"
        col_name += ": {}{}".format(
            "L" if fmt_name in has_load else "", "D" if fmt_name in has_dump else ""
        )
        header.append(col_name)
    table = [header]
    for attr_name in rows:
        # If an attribute is a property, we mark it as "d" for "derived from
        # other attributes if possible".
        row = [f":py:attr:`{attr_name} <iodata.iodata.IOData.{attr_name}>`"]
        if isinstance(getattr(iodata.IOData, attr_name), property):
            row[0] += " *(d)*"
        # Loop over formats and set flags
        for fmt_name in cols:
            cell = ""
            if fmt_name in guaranteed[attr_name]:
                cell += "R"
            if fmt_name in ifpresent[attr_name]:
                cell += "r"
            if fmt_name in required[attr_name]:
                cell += "W"
            if fmt_name in optional[attr_name]:
                cell += "w"
            if cell == "":
                cell = "."
            row.append(cell)
        table.append(row)
    return table


def print_rst_table(table, nhead=1):
    """Print an RST table.

    Parameters
    ----------
    table : list[list[]]
        A list of rows. Each row must be a list of cells containing strings
    nhead : int, optional
        The number of header rows

    """

    def format_cell(cell):
        if cell is None or len(cell.strip()) == 0:
            return "\\ "
        return str(cell)

    # Determine the width of each column
    widths = {}
    for row in table:
        for icell, cell in enumerate(row):
            widths[icell] = max(widths.get(icell, 2), len(format_cell(cell)))

    def format_row(cells, margin):
        return " ".join(
            margin + format_cell(cell).rjust(widths[icell]) + margin
            for icell, cell in enumerate(cells)
        )

    # construct the column markers
    markers = format_row(["=" * widths[icell] for icell in range(len(widths))], "=")

    # top markers
    print(markers)

    # heading rows (if any)
    for irow in range(nhead):
        print(format_row(table[irow], " "))
    if nhead > 0:
        print(markers)

    # table body
    for irow in range(nhead, len(table)):
        print(format_row(table[irow], " "))

    # bottom markers
    print(markers)


if __name__ == "__main__":
    content_table = generate_table_rst()
    print_rst_table(content_table)
