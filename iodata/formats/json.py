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
"""JSON file format."""


from typing import List, Tuple, Union, TextIO
import json

import numpy as np

from ..basis import CCA_CONVENTIONS, Shell, MolecularBasis
from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..orbitals import MolecularOrbitals
from ..utils import angstrom, FileFormatError, FileFormatWarning, LineIterator
from warnings import warn

__all__ = []


PATTERNS = ["*.json"]


@document_load_one("JSON", ["atcoords", "atnums", "atcorenums", "mo", "obasis"], ["title"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    json_in = json.load(lit._f)
    result = _parse_json(json_in, lit)
    return result


def _parse_json(json_in: dict, lit: LineIterator) -> dict:
    """Parse data from QCSchema JSON input file.

    QCSchema supports four different schema types: `qcschema_molecule`, specifying one or more
    molecules in a single system; `qcschema_basis`, specifying a basis set for a molecular system,
    `qcschema_input`, specifying input to a QC program for a specific system; and `qcschema_output`,
    specifying results of a QC program calculation for a specific system along with the input
    information.

    Parameters
    ----------
    lit
        The line iterator holding the file data.
    json_in
        The JSON dict loaded from file.

    Returns
    -------
    out
        Output dictionary containing ...

    """
    # Remove all null entries and empty dicts in json
    # QCEngine seems to add null entries and empty dicts even for optional and empty keys
    fix_keys = {k: v for k, v in json_in.items() if v is not None}
    fix_subkeys = dict()
    for key in fix_keys:
        if isinstance(fix_keys[key], dict):
            fix_subkeys[key] = {k: v for k, v in fix_keys[key].items() if v is not None}
    result = {**fix_keys, **fix_subkeys}
    # Remove empty dicts
    keys = list(result.keys())
    for key in keys:
        if isinstance(result[key], dict) and not bool(result[key]):
            del result[key]

    # Determine schema type
    if "schema_name" in result:
        if result["schema_name"] not in {
            "qcschema_molecule",
            "qcschema_basis",
            "qcschema_input",
            "qcschema_output"
        }:
            del result["schema_name"]
    if "schema_name" not in result:
        # Attempt to determine schema type, since some QCElemental files omit this
        warn(
            "{}: QCSchema files should have a `schema_name` key."
            "Attempting to determine schema type...".format(lit.filename),
            FileFormatWarning,
            2,
        )
        # Geometry is required in any molecule schema
        if "geometry" in result:
            schema_name = "qcschema_molecule"
        # Check if BSE file, which is too different
        elif "molssi_bse_schema" in result:
            raise FileFormatError(
                "{}: IOData does not currently support MolSSI BSE Basis JSON.".format(lit.filename)
            )
        # Center_data is required in any basis schema
        elif "center_data" in result:
            schema_name = "qcschema_basis"
        elif "driver" in result:
            if "return_result" in result:
                schema_name = "qcschema_output"
            else:
                schema_name = "qcschema_input"
        else:
            raise FileFormatError("{}: Could not determine `schema_name`.".format(lit.filename))
    else:
        schema_name = result["schema_name"]
    if "schema_version" not in result:
        warn(
            "{}: QCSchema files should have a `schema_version` key."
            "Attempting to load without version number.".format(lit.filename),
            FileFormatWarning,
            2,
        )

    if schema_name == "qcschema_molecule":
        return _load_qcschema_molecule(result, lit)
    elif schema_name == "qcschema_basis":
        # FIXME add _load_qcschema_basis
        return _load_qcschema_basis(result, lit)
    elif schema_name == "qcschema_input":
        return _load_qcschema_input(result, lit)
    elif schema_name == "qcschema_output":
        return _load_qcschema_output(result, lit)
    else:
        raise FileFormatError(
            "{}: Invalid QCSchema type {}, should be one of `qcschema_molecule`, `qcschema_basis`,"
            "`qcschema_input`, or `qcschema_output".format(lit.filename, result["schema_name"])
        )


def _load_qcschema_molecule(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_molecule properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file, same as the `molecule` key in QCSchema input/output files.
    lit
        The line iterator holding the file data.

    Returns
    -------
    molecule_dict
        Output dictionary containing ``atcoords``, ``atnums`` & ``spinpol`` keys and
        corresponding values.
        It may contain ``atcorenums``, ``atmasses``, ``bonds``, ``charge`` & ``extra`` keys
        and corresponding values as well.

    """
    molecule_dict = dict()

    return molecule_dict


def _load_qcschema_basis(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_basis properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    basis_dict
        ...
    """
    basis_dict = dict()

    return basis_dict


def _load_qcschema_input(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_input properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    input_dict
        ...
    """
    input_dict = dict()

    return input_dict


def _load_qcschema_output(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_output properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    output_dict
        ...
    """
    output_dict = dict()

    return output_dict