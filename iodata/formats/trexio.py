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
"""TrexIO file format."""

from __future__ import annotations

from typing import TextIO

import numpy as np

from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..utils import LineIterator, LoadError

try:
    import trexio  # type: ignore[import]
except ImportError:
    trexio = None

__all__ = ()

PATTERNS = ["*.trexio", "*.h5", "*.hdf5"]


@document_load_one(
    "TREXIO",
    ["atcoords", "atnums"],
    ["charge", "nelec", "spinpol"],
)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    filename = lit.filename

    if trexio is None:
        raise LoadError(
            "Reading TREXIO files requires the 'trexio' Python package.",
            filename,
        )

    try:
        with trexio.File(filename, "r", back_end=trexio.TREXIO_AUTO) as tfile:
            n_nuc = trexio.read_nucleus_num(tfile)
            charges = np.asarray(trexio.read_nucleus_charge(tfile), dtype=float)
            coords = np.asarray(trexio.read_nucleus_coord(tfile), dtype=float)

            try:
                nelec = int(trexio.read_electron_num(tfile))
            except trexio.Error:
                nelec = None

            try:
                n_up = int(trexio.read_electron_up_num(tfile))
                n_dn = int(trexio.read_electron_dn_num(tfile))
                spinpol = n_up - n_dn
            except trexio.Error:
                spinpol = None

    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Failed to read TREXIO file: {exc}", filename) from exc

    # Validate data consistency after reading
    if charges.shape[0] != n_nuc or coords.shape[0] != n_nuc:
        raise LoadError(
            "Inconsistent nucleus.* fields in TREXIO file.",
            filename,
        )

    atnums = np.rint(charges).astype(int)

    result: dict = {
        "atcoords": coords,
        "atnums": atnums,
    }

    if nelec is not None:
        result["nelec"] = nelec
        result["charge"] = float(charges.sum() - nelec)
    if spinpol is not None:
        result["spinpol"] = spinpol

    return result


@document_dump_one(
    "TREXIO",
    ["atcoords", "atnums"],
    ["charge", "nelec", "spinpol"],
)
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    if trexio is None:
        raise RuntimeError("Writing TREXIO files requires the 'trexio' Python package.")

    if data.atcoords is None or data.atnums is None:
        raise RuntimeError("TREXIO writer needs atcoords and atnums.")
    if data.atcoords.shape[0] != data.atnums.shape[0]:
        raise RuntimeError("Inconsistent number of atoms in atcoords and atnums.")

    try:
        filename = f.name
    except AttributeError as exc:
        raise RuntimeError(
            "TREXIO writer expects a real file object with a .name attribute."
        ) from exc

    atcoords = np.asarray(data.atcoords, dtype=float)
    atnums = np.asarray(data.atnums, dtype=float)
    nelec = int(data.nelec) if data.nelec is not None else None
    spinpol = int(data.spinpol) if data.spinpol is not None else None

    with trexio.File(filename, "w", back_end=trexio.TREXIO_HDF5) as tfile:
        trexio.write_nucleus_num(tfile, len(atnums))
        trexio.write_nucleus_charge(tfile, atnums)
        trexio.write_nucleus_coord(tfile, atcoords)

        if nelec is not None:
            trexio.write_electron_num(tfile, nelec)
            if spinpol is not None:
                n_up = (nelec + spinpol) // 2
                n_dn = nelec - n_up
                trexio.write_electron_up_num(tfile, n_up)
                trexio.write_electron_dn_num(tfile, n_dn)
