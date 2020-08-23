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


from collections import defaultdict
import inspect
import importlib

import iodata


__all__ = []


def _generate_all_format_parser():
    """Parse supported functionality from each module.

    Returns
    -------
    tuple(dict, dict)
        dict{format_name: index}
        dict{proper_name: formats_name_supported}
    """
    # inspect iodata format module
    # obtain a list of tuple [(module_name: str, module_object: obj)]
    format_modules = inspect.getmembers(iodata.formats, inspect.ismodule)

    # store supported format name and index (position) in table
    fmt_names = []
    # store guaranteed and ifpresent attributes & corresponding formats
    prop_guaranteed = defaultdict(list)
    prop_ifpresent = defaultdict(list)
    for fmt_name, fmt_module in format_modules:
        # add new format name to fmt_names
        if fmt_name not in fmt_names:
            fmt_names.append(fmt_name)
        # obtaining supported properties
        for attrname in fmt_module.load_one.guaranteed:
            # add format to its supported property list
            prop_guaranteed[attrname].append(fmt_name)
        for attrname in fmt_module.load_one.ifpresent:
            prop_ifpresent[attrname].append(fmt_name)
    return fmt_names, prop_guaranteed, prop_ifpresent


def generate_table_rst():
    """Construct table contents.

    Returns
    -------
    list[list[]]
        table with rows of each property and columns of each format
    """
    table = []
    fmt_names, prop_guaranteed, prop_ifpresent = _generate_all_format_parser()

    # Sort rows by number of times the attribute is used in decreasing order.
    rows = [name for name in dir(iodata.IOData) if not name.startswith('_')]
    rows.sort(key=(lambda name: len(prop_ifpresent[name]) + len(prop_guaranteed[name])), reverse=True)

    # order columns based on number of guaranteed and ifpresent entries for each format
    cols = []
    for fmt_names in fmt_names:
        count = sum((fmt_names in value) for value in prop_guaranteed.values())
        count += sum((fmt_names in value) for value in prop_ifpresent.values())
        cols.append((count, fmt_names))
    cols = [item[1] for item in sorted(cols)[::-1]]

    # construct header with cross-referencing columns
    table = [["Properties"] + [f':ref:`{col} <format_{col}>`' for col in cols]]
    # attributes of IOData that have property type are going to be marked as always present,
    # because they are derived from other attributes. However, this is not true for 'fcidump'
    # and 'gaussianlog' formats (because they do not load information which is used to derive
    # these property attributes), so we manually exclude these at the moment.
    temp_index = [cols.index(fmt_names) for fmt_names in ['fcidump', 'gaussianlog']]
    for prop in rows:
        # construct default row entries
        row = ["."] * len(cols)
        # set property attributes as always present expect for 'fcidump' & 'gaussianlog'
        if isinstance(getattr(iodata.IOData, prop), property):
            row = [item if index in temp_index else u"\u2713" for index, item in enumerate(row)]
        # add attribute name as the first item on the row
        row.insert(0, prop)
        # check whether attribute is guaranteed or ifpresent for a format
        for fmt_names in prop_guaranteed[prop]:
            row[cols.index(fmt_names) + 1] = u"\u2713"
        for fmt_names in prop_ifpresent[prop]:
            row[cols.index(fmt_names) + 1] = 'm'
        table.append(row)
    return table


def print_rst_table(table, nhead=1):
    """Write an RST table to file f.

    Parameters
    ----------
    f : file object
        A writable file object
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
