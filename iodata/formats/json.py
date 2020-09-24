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


import json
from typing import List, Union
from warnings import warn

import numpy as np

from ..docstrings import document_load_one
from ..periodic import sym2num
from ..utils import FileFormatError, FileFormatWarning, LineIterator

try:
    from iodata.version import __version__
except ImportError:
    __version__ = "0.0.0.post0"

__all__ = []


PATTERNS = ["*.json"]


# FIXME: check this
@document_load_one("QCSchema", ["atcoords", "atnums", "atcorenums", "mo", "obasis"], ["title"])
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
            "qcschema_output",
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
        raise NotImplementedError("{} is not yet implemented in IOData.".format(schema_name))
    elif schema_name == "qcschema_input":
        raise NotImplementedError("{} is not yet implemented in IOData.".format(schema_name))
    elif schema_name == "qcschema_output":
        raise NotImplementedError("{} is not yet implemented in IOData.".format(schema_name))
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
    # All Topology properties are found in the "molecule" key
    molecule_dict = _parse_topology_keys(result, lit)

    return molecule_dict


def _parse_topology_keys(mol: dict, lit: LineIterator) -> dict:
    """Load topology properties from old QCSchema Molecule specifications.

    The qcschema_molecule v2 specification requires a topology for every file, specified in the
    `molecule` key, containing at least the keys `schema_name`, `schema_version`, `symbols`,
    `geometry`, `molecular_charge`, `molecular_multiplicity`, and `provenance`. This schema is
    currently used in QCElemental (and thus the QCArchive ecosystem).

    qcschema_molecule v1 only exists as the specification on the QCSchema website, and seems never
    to have been implemented in QCArchive. It is possible to accept v1 input, since all required
    keys for v2 exist as keys in v1, but it is preferable to convert those files to v2 explicitly.

    Parameters
    ----------
    mol
        The 'molecule' key from the QCSchema file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    topology_dict
        Output dictionary containing ``atcoords`` & ``atnums`` keys and corresponding values.
        It may contain ``atcorenums``, ``atmasses``, ``bonds``, ``charge`` & ``extra`` keys
        and corresponding values as well.

    """
    # Make sure required topology properties are present
    # NOTE: currently molecular_charge and molecular_multiplicity have default values,
    # so some programs may not provide those required keys
    # QCEngineRecords files follow Molecule v1 for some reason, despite being uploaded in 2019
    should_be_required_keys = {"schema_name", "schema_version", "provenance"}
    topology_keys = {
        "symbols",
        "geometry",
    }
    for key in should_be_required_keys:
        if key not in mol:
            warn(
                "{}: QCSchema files should have a '{}' key.".format(lit.filename, key),
                FileFormatWarning,
                2,
            )
    for key in topology_keys:
        if key not in mol:
            raise FileFormatError(
                "{}: QCSchema topology requires '{}' key".format(lit.filename, key)
            )

    topology_dict = dict()
    extra_dict = dict()

    # Save schema name & version
    extra_dict["schema_name"] = "qcschema_molecule"
    try:
        version = mol["schema_version"]
    except KeyError:
        version = -1
    if float(version) < 0 or float(version) > 2:
        warn(
            "{}: Unknown `qcschema_molecule` version {}, "
            "loading may produce invalid results".format(lit.filename, version),
            FileFormatWarning,
            2,
        )
    extra_dict["schema_version"] = version

    atnums = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
    atcorenums = atnums.copy()
    topology_dict["atnums"] = atnums
    # Geometry is in a flattened list, convert to N x 3
    topology_dict["atcoords"] = np.array(
        [mol["geometry"][3 * i : (3 * i) + 3] for i in range(0, len(mol["geometry"]) // 3)]
    )
    # Check for missing charge, warn that this is a required field
    if "molecular_charge" not in mol:
        warn(
            "{}: Missing 'molecular_charge' key."
            "Some QCSchema writers omit this key for default value 0.0,"
            "Ensure this value is correct.",
            FileFormatWarning,
            2,
        )
        formal_charge = 0.0
    else:
        formal_charge = mol["molecular_charge"]
    # Determine number of electrons from atomic numbers and formal charge
    topology_dict["charge"] = formal_charge
    topology_dict["nelec"] = np.sum(atnums) - formal_charge
    # Check for missing mult, warn that this is a required field
    if "molecular_multiplicity" not in mol:
        warn(
            "{}: Missing 'molecular_multiplicity' key."
            "Some QCSchema writers omit this key for default value 1,"
            "Ensure this value is correct.",
            FileFormatWarning,
            2,
        )
        topology_dict["spinpol"] = 0
    else:
        mult = mol["molecular_multiplicity"]
        topology_dict["spinpol"] = mult - 1

    # Provenance is optional in v1
    if "provenance" in mol:
        extra_dict["provenance"] = _parse_provenance(
            mol["provenance"], lit, "qcschema_molecule", False
        )

    # Check for optional keys
    # Load ghost atoms as atoms with zero effective core charge
    if "real" in mol:
        ghosts = mol["real"]
        extra_dict["real"] = ghosts
        atcorenums[ghosts is False] = 0
    # Load atom masses to array, canonical weights assumed if masses not given
    if "masses" in mol and "mass_numbers" in mol:
        warn(
            "{}: Both `masses` and `mass_numbers` given. "
            "Both values will be written to `extra` dict.",
            FileFormatWarning,
            2,
        )
        extra_dict["mass_numbers"] = np.array(mol["mass_numbers"])
        extra_dict["masses"] = np.array(mol["masses"])
    elif "masses" in mol:
        topology_dict["atmasses"] = np.array(mol["masses"])
    elif "mass_numbers" in mol:
        topology_dict["atmasses"] = np.array(mol["mass_numbers"])
    # Load bonds: list of tuple (atom1, atom2, bond_order)
    # Note: The QCSchema spec allows for non-integer bond_orders, these are forced to integers here
    # in accordance with current IOData specification
    if "connectivity" in mol:
        topology_dict["bonds"] = np.array(mol["connectivity"], dtype=int)
    # Check for fragment keys
    # List fragment indices in nested list (likely is a jagged array)
    if "fragments" in mol:
        fragments = mol["fragments"]
        extra_dict["fragments"] = {"indices": fragments}
        if "fragment_charges" in mol:
            extra_dict["fragments"]["charges"] = mol["fragment_charges"]
        if "fragment_multiplicities" in mol:
            extra_dict["fragments"]["multiplicities"] = mol["fragment_multiplicities"]
    if "fix_symmetry" in mol:
        topology_dict["g_rot"] = mol["fix_symmetry"]
    if "fix_orientation" in mol:
        extra_dict["fix_orientation"] = mol["fix_orientation"]

    # Check for other extras
    # name, comment, fix_com, fix_symmetry, fix_orientation
    if "name" in mol:
        topology_dict["title"] = mol["name"]
    if "comment" in mol:
        extra_dict["comment"] = mol["comment"]
    if "fix_com" in mol:
        extra_dict["fix_com"] = mol["fix_com"]
    if "identifiers" in mol:
        extra_dict["identifiers"] = mol["identifiers"]
    if "validated" in mol:
        extra_dict["qcel_validated"] = mol["validated"]
    if "atom_labels" in mol:
        extra_dict["atom_labels"] = mol["atom_labels"]
    if "atomic_numbers" in mol:
        extra_dict["atomic_numbers"] = mol["atomic_numbers"]
    if "id" in mol:
        extra_dict["id"] = mol["id"]
    if "extras" in mol:
        extra_dict["extras"] = mol["extras"]

    if extra_dict:
        topology_dict["extra"] = extra_dict

    molecule_keys = {
        "schema_name",
        "schema_version",
        "validated",
        "symbols",
        "geometry",
        "name",
        "identifiers",
        "comment",
        "molecular_charge",
        "molecular_multiplicity",
        "masses",
        "real",
        "atom_labels",
        "atomic_numbers",
        "mass_numbers",
        "connectivity",
        "fragments",
        "fragment_charges",
        "fragment_multiplicities",
        "fix_com",
        "fix_orientation",
        "fix_symmetry",
        "provenance",
        "id",
        "extras",
    }
    parsed_keys = molecule_keys.intersection(mol.keys())
    for key in parsed_keys:
        del mol[key]
    if mol:
        topology_dict["extra"]["unparsed"] = mol

    return topology_dict


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


def _parse_provenance(
    provenance: Union[List[dict], dict], lit: LineIterator, source: str, append=True
) -> Union[List[dict], dict]:
    """Load provenance properties from QCSchema.

    Parameters
    ----------
    provenance

    lit
        The line iterator holding the file data.
    source
        The schema type {`qcschema_molecule`, `qcschema_input`, `qcschema_output`} associated
        with this provenance data.
    append
        Append IOData provenance entry to provenance list?

    """
    if isinstance(provenance, dict):
        if "creator" not in provenance:
            raise FileFormatError(
                "{}: `{}` provenance requires `creator` key".format(lit.filename, source)
            )
        else:
            if append:
                base_provenance = [provenance]
            else:
                return provenance
    elif isinstance(provenance, list):
        for prov in provenance:
            if "creator" not in prov:
                raise FileFormatError("{}: `{}` provenance requires `creator` key")
        base_provenance = provenance
    else:
        raise FileFormatError("{}: Invalid `{}` provenance type".format(lit.filename, source))
    if append:
        base_provenance.append(
            {"creator": "IOData", "version": __version__, "routine": "iodata.formats.json"}
        )
    return base_provenance
