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

    atom_columns = iodata.formats.xzy.DEFAULT_ATOM_COLUMNS + [
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


@document_load_one("XYZ", ['atcoords', 'atnums', 'title'],
                   [], {"atom_columns": ATOM_COLUMNS_DOC})
def load_one(lit: LineIterator, atom_columns=None) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    if atom_columns is None:
        atom_columns = DEFAULT_ATOM_COLUMNS
    # Load the header
    natom = int(next(lit))
    data = {"title": next(lit).strip()}
    # Load the atom lines.
    for iatom in range(natom):
        words = next(lit).split()
        for attrname, keyname, shapesuffix, dtype, loadword, _dumpword in atom_columns:
            if iatom == 0:
                # In case of the first atom, the array with atomic properties
                # is created and put in the right place.
                dest = np.zeros((natom,) + shapesuffix, dtype=dtype)
                if keyname is None:
                    data[attrname] = dest
                else:
                    data.setdefault(attrname, {})[keyname] = dest
                dest = dest[:1]
            else:
                # For all remaining atoms, the array with atomic properties is
                # just retrieved.
                if keyname is None:
                    dest = data[attrname][iatom: iatom + 1]
                else:
                    dest = data[attrname][keyname][iatom: iatom + 1]
            # Fill in array elements with atomic properties.
            for ifield in range(dest.size):
                dest.flat[ifield] = loadword(words.pop(0))
    return data


@document_load_many("XYZ", ['atcoords', 'atnums', 'title'],
                    [], {"atom_columns": ATOM_COLUMNS_DOC})
def load_many(lit: LineIterator, atom_columns=None) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # XYZ Trajectory files are a simple concatenation of individual XYZ files,'
    # making it trivial to load many frames.
    while True:
        try:
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
