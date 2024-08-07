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
"""Molpro 2012 FCIDUMP file format.

Notes
-----
1. This function works only for restricted wave-functions.
2. One- and two-electron integrals are stored in chemists' notation in an FCIDUMP file,
   while IOData internally uses physicists' notation.
3. Keep in mind that the FCIDUMP format changed in MOLPRO 2012, so files generated with
   older versions are not supported.

"""

from typing import TextIO
from warnings import warn

import numpy as np

from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..utils import LineIterator, LoadError, LoadWarning, set_four_index_element

__all__ = ()


PATTERNS = ["*FCIDUMP*", "*.fcidump"]


LOAD_ONE_NOTES = """
IOData stores four-index objects in physicists' notation internally and
assumes they are stored in an FCIDUMP file in chemists' notation.
"""


@document_load_one(
    "Molpro 2012 FCIDUMP",
    ["core_energy", "one_ints", "nelec", "spinpol", "two_ints"],
    notes=LOAD_ONE_NOTES,
)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # check header
    line = next(lit)
    if not line.startswith(" &FCI NORB="):
        raise LoadError(f"Incorrect file header: {line.strip()}", lit)

    # read info from header
    words = line[5:].split(",")
    header_info = {}
    for word in words:
        if word.count("=") == 1:
            key, value = word.split("=")
            header_info[key.strip()] = value.strip()
    nbasis = int(header_info["NORB"])
    nelec = int(header_info["NELEC"])
    spinpol = int(header_info["MS2"])

    # skip rest of header
    for line in lit:
        words = line.split()
        if words[0] == "&END" or words[0] == "/END" or words[0] == "/":
            break

    # read the integrals
    one_mo = np.zeros((nbasis, nbasis))
    two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis))
    core_energy = 0.0

    for line in lit:
        words = line.split()
        if len(words) != 5:
            raise LoadError(
                f"Expecting 5 fields on each data line in FCIDUMP, got {len(words)}.", lit
            )
        value = float(words[0])
        if words[3] != "0":
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            ik = int(words[3]) - 1
            il = int(words[4]) - 1
            if two_mo[ii, ik, ij, il] != 0.0:
                warn(
                    LoadWarning("Duplicate entries in the FCIDUMP file are ignored", lit),
                    stacklevel=2,
                )
            set_four_index_element(two_mo, ii, ik, ij, il, value)
        elif words[1] != "0":
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            one_mo[ii, ij] = value
            one_mo[ij, ii] = value
        else:
            core_energy = value

    return {
        "nelec": nelec,
        "spinpol": spinpol,
        "one_ints": {"core_mo": one_mo},
        "two_ints": {"two_mo": two_mo},
        "core_energy": core_energy,
    }


DUMP_ONE_NOTES = """
The dictionary ``one_ints`` must contain a field ``core_mo``.
Similarly, ``two_ints`` must contain ``two_mo``.
IOData stores four-index objects in physicists' notation internally and
dumps them to an FCIDUMP file in chemists' notation.
"""


@document_dump_one(
    "Molpro 2012 FCIDUMP",
    ["one_ints", "two_ints"],
    ["core_energy", "nelec", "spinpol"],
    notes=DUMP_ONE_NOTES,
)
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    one_mo = data.one_ints["core_mo"]

    # Write header
    nactive = one_mo.shape[0]
    nelec = data.nelec or 0
    spinpol = data.spinpol or 0
    print(f" &FCI NORB={nactive:d},NELEC={nelec:d},MS2={spinpol:d},", file=f)
    print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
    print("  ISYM=1", file=f)
    print(" &END", file=f)

    # Write integrals and core energy
    two_mo = data.two_ints["two_mo"]
    for i0 in range(nactive):
        for i1 in range(i0 + 1):
            for i2 in range(nactive):
                for i3 in range(i2 + 1):
                    if (i0 * (i0 + 1)) / 2 + i1 >= (i2 * (i2 + 1)) / 2 + i3:
                        value = two_mo[i0, i2, i1, i3]
                        if value != 0.0:
                            print(
                                f"{value:23.16e} {i0 + 1:4d} {i1 + 1:4d} {i2 + 1:4d} {i3 + 1:4d}",
                                file=f,
                            )
    for i0 in range(nactive):
        for i1 in range(i0 + 1):
            value = one_mo[i0, i1]
            if value != 0.0:
                print(f"{value:23.16e} {i0 + 1:4d} {i1 + 1:4d} {0:4d} {0:4d}", file=f)
    if data.core_energy is not None:
        print(f"{data.core_energy:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}", file=f)
