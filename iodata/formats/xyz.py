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
"""XYZ file format.

Usually, the different frames in a trajectory describe different geometries of the same
molecule, with atoms in the same order. The ``load_many`` and ``dump_many`` functions
below can also handle an XYZ with different molecules, e.g. a molecular database.

The ``load_*`` and ``dump_*`` functions all accept the optional argument
``atom_columns``. This argument fixes the meaning of the columns to be loaded
from or dumped to an XYZ file. The following example defines, in addition to the
conventional columns, also a column with atomic charges and three columns with
atomic forces.

.. code-block :: python

    atom_columns = iodata.formats.xyz.DEFAULT_ATOM_COLUMNS + [
        # Atomic charges are stored in a dictionary atcharges and they key
        # refers to the name of the partitioning method.
        ("atcharges", "mulliken", (), float, float, "{:10.5f}".format),
        # Note that in IOData, the energy gradient is stored, which contains the
        # negative forces.
        ("atgradient", None, (3,), float,
         (lambda word: -float(word)),
         (lambda value: "{:15.10f}".format(-value)))
    ]

    mol = load_one("test.xyz", atom_columns=atom_columns)
    # The following attributes are present:
    print(mol.atnums)
    print(mol.atcoords)
    print(mol.atcharges["mulliken"])
    print(mol.atgradient)

When defining ``atom_columns``, no columns can be skipped, such that all
information loaded from a file can also be written back out when dumping it.

"""

import shlex
from typing import TextIO, Iterator

import numpy as np

from ..docstrings import (document_load_one, document_load_many, document_dump_one,
                          document_dump_many)
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.xyz']


DEFAULT_ATOM_COLUMNS = [
    ("atnums", None, (), int,
     (lambda word: int(word) if word.isdigit() else sym2num[word.title()]),
     (lambda atnum: "{:2s}".format(num2sym[atnum]))),
    ("atcoords", None, (3,), float,
     (lambda word: float(word) * angstrom),
     (lambda value: "{:15.10f}".format(value / angstrom)))
]


ATOM_COLUMNS_DOC = """\
A list of atomic fields to be loaded. Each field as a tuple with the following
items: **attribute** (``str``), **key** (``None`` or ``str``, when ``str`` the
``IOData`` attribute is a ``dict``), **shape** for one atom (``tuple``),
**dtype**, **load_word** (function taking string and returning a value with the
correct type), **dump_word** (function taking a value and returning a formatted
string).
"""


BOOL_MAP = {"T": True, "F": False, "True": True, "False": False}


def convert_title_value(value: str):
    """Search for the correct dtype and convert the string."""
    list_of_splits = value.split()
    # If it is just one item, first try int, then float and finally return a bool or str
    if len(list_of_splits) == 1:
        try:
            converted_value = int(value)
        except ValueError:
            try:
                converted_value = float(value)
            except ValueError:
                if value in BOOL_MAP.keys():
                    converted_value = BOOL_MAP[value]
                else:
                    converted_value = value
    else:
        # Do the same but return it as a numpy array
        try:
            converted_value = np.array(list_of_splits, dtype=np.int)
        except ValueError:
            try:
                converted_value = np.array(list_of_splits, dtype=np.float)
            except ValueError:
                try:
                    converted_value = np.array([BOOL_MAP[bool_key] for bool_key in list_of_splits],
                                               dtype=np.bool)
                except KeyError:
                    converted_value = np.array(list_of_splits, dtype=np.str)
    return converted_value


def parse_properties(properties: str):
    """Parse the properties into atom_columns."""
    atom_columns = []
    # Maps the dtype to the atom_columns dtype, load_word and dump_word
    dtype_map = {"S": (np.dtype('U25'), str, "{:10s}".format),
                 "R": (float, float, "{:15.10f}".format),
                 "I": (int, int, "{:10d}".format),
                 "L": (bool, lambda word: BOOL_MAP[word], lambda boolean: "T" if boolean else "F")}
    # Some predefined iodata attributes which can be mapped
    # Only pos is assumed to be in angstrom, no unit convertion takes place for the other attributes
    atom_column_map = {'pos': ('atcoords', None, (3,), float,
                               (lambda word: float(word) * angstrom),
                               (lambda value: "{:15.10f}".format(value / angstrom))),
                       'mass': ('atmasses', None, (), float, float, "{:10.5f}".format),
                       'force': ('atgradient', None, (3,), float,
                                 (lambda word: -float(word)),
                                 (lambda value: "{:15.10f}".format(-value)))}
    atnum_column = ("atnums", None, (), int,
                    (lambda word: int(word) if word.isdigit() else sym2num[word.title()]),
                    (lambda atnum: "{:2s}".format(num2sym[atnum])))
    splitted_properties = properties.split(':')
    assert len(splitted_properties) % 3 == 0
    # Each property has 3 values: its name, dtype and shape
    names = splitted_properties[::3]
    dtypes = splitted_properties[1::3]
    shapes = splitted_properties[2::3]
    if 'Z' in names:
        # Try to map 'Z' to the 'atnums' attribute
        atom_column_map['Z'] = atnum_column
    elif 'species' in names:
        # If 'Z' is not present, use 'species'
        atom_column_map['species'] = atnum_column
    for name, dtype, shape in zip(names, dtypes, shapes):
        if name in atom_column_map.keys():
            atom_columns.append(atom_column_map[name])
        else:
            # Use the 'extra' attribute to store values which are not predefined in iodata
            if shape == '1':
                shape_suffix = ()
            else:
                shape_suffix = (int(shape),)
            atom_columns.append(('extra', name, shape_suffix, *dtype_map[dtype]))
    return atom_columns


def parse_title(title: str):
    """Parse the title in an extended xyz file."""
    key_value_pairs = shlex.split(title)
    # A dict of predefined iodata atrributes with their names and dtype convertion functions

    def load_cellvecs(word):
        return np.array(word.split(), dtype=np.float).reshape([3, 3]) * angstrom
    iodata_attrs = {'energy': ('energy', float),
                    'Lattice': ('cellvecs', load_cellvecs),
                    'charge': ('charge', float)}
    data = {}
    for key_value_pair in key_value_pairs:
        if '=' in key_value_pair:
            key, value = key_value_pair.split('=')
            if key == 'Properties':
                atom_columns = parse_properties(value)
            elif key in iodata_attrs.keys():
                data[iodata_attrs[key][0]] = iodata_attrs[key][1](value)
            else:
                data.setdefault('extra', {})[key] = convert_title_value(value)
        else:
            # If no value is given, set it True
            data.setdefault('extra', {})[key_value_pair] = True
    return atom_columns, data


@document_load_one("XYZ", ['atcoords', 'atnums', 'title'],
                   [], {"atom_columns": ATOM_COLUMNS_DOC})
def load_one(lit: LineIterator, atom_columns=None) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Load the header.
    natom = int(next(lit))
    title = next(lit).strip()
    data = {}
    if atom_columns is None:
        atom_columns = DEFAULT_ATOM_COLUMNS
    elif atom_columns == 'EXT':
        # The extended xyz format defines the atom_columns in the title
        atom_columns, data = parse_title(title)
    data["title"] = title
    # Initialize the arrays to be loaded from the XYZ file.
    for attrname, keyname, shapesuffix, dtype, _loadword, _dumpword in atom_columns:
        array = np.zeros((natom,) + shapesuffix, dtype=dtype)
        if keyname is None:
            # Store the initial array as a normal attribute.
            data[attrname] = array
        else:
            # Store the initial array as a value in an dictionary attribute.
            data.setdefault(attrname, {})[keyname] = array
    # Load the atom lines.
    for iatom in range(natom):
        words = next(lit).split()
        for attrname, keyname, _shapesuffix, _dtype, loadword, _dumpword in atom_columns:
            # Get the slice of the array where properties for the current atom
            # must be stored.
            if keyname is None:
                # The array is a normal attribute.
                atom_array = data[attrname][iatom: iatom + 1]
            else:
                # The array is a value of a dictionary attribute.
                atom_array = data[attrname][keyname][iatom: iatom + 1]
            # Fill in array elements with atomic properties. For each new value
            # to be loaded, the first element of the list words is consumed and
            # converted to the right format for IOData.
            for ifield in range(atom_array.size):
                atom_array.flat[ifield] = loadword(words.pop(0))
    return data


@document_load_many("XYZ", ['atcoords', 'atnums', 'title'],
                    [], {"atom_columns": ATOM_COLUMNS_DOC})
def load_many(lit: LineIterator, atom_columns=None) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # XYZ Trajectory files are a simple concatenation of individual XYZ files,'
    # making it trivial to load many frames.
    while True:
        try:
            # Check for and skip empty lines at the end of file
            line = next(lit)
            if line.strip() == "":
                return
            lit.back(line)
            yield load_one(lit, atom_columns)
        except StopIteration:
            return


@document_dump_one("XYZ", ['atcoords', 'atnums'], ['title'],
                   {"atom_columns": ATOM_COLUMNS_DOC})
def dump_one(f: TextIO, data: IOData, atom_columns=None):
    """Do not edit this docstring. It will be overwritten."""
    if atom_columns is None:
        atom_columns = DEFAULT_ATOM_COLUMNS
    # Write the header
    print(data.natom, file=f)
    print(data.title or 'Created with IOData', file=f)
    # Write the atom lines
    for iatom in range(data.natom):
        words = []
        for attrname, keyname, _shapesuffix, _dtype, _loadword, dumpword in atom_columns:
            values = getattr(data, attrname)
            if keyname is not None:
                # The data to be written is a value of a dictionary attribute.
                values = values[keyname]
            for value in values[iatom].flat:
                words.append(dumpword(value))
        print(" ".join(words), file=f)


@document_dump_many("XYZ", ['atcoords', 'atnums'], ['title'],
                    {"atom_columns": ATOM_COLUMNS_DOC})
def dump_many(f: TextIO, datas: Iterator[IOData], atom_columns=None):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data, atom_columns)
