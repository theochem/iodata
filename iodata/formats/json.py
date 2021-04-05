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
# pylint: disable=too-many-branches, too-many-statements
"""QCSchema JSON file format.

QCSchema defines four different subschema:
- Molecule: specifying a molecular system
- Input: specifying QC program input for a specific Molecule
- Output: specifying QC program output for a specific Molecule
- Basis: specifying a basis set for a specific Molecule

The QCSchema subschema are in various levels of maturity, and are subject to change at any time
without warning, as they are also used as the internal data representation for the QCElemental
program. IOData currently supports the Molecule subschema for both ``load_one`` and ``dump_one``.

"""


import json
from typing import List, TextIO, Union
from warnings import warn

import numpy as np

from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..periodic import num2sym, sym2num
from ..utils import FileFormatError, FileFormatWarning, LineIterator
from .. import __version__


__all__ = []


PATTERNS = ["*.json"]


@document_load_one(
    "QCSchema",
    ["atnums", "atcorenums", "atcoords", "charge", "nelec", "spinpol"],
    ["atmasses", "bonds", "g_rot", "title", "extra"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    json_in = json.load(lit.f)
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
        Dictionary with IOData attributes.

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
        # Correct for qc_schema vs. qcschema, due to inconsistencies in prior versions
        schema_name = result["schema_name"].replace("qc_schema", "qcschema")
        if schema_name not in {
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
    if "schema_version" not in result:
        warn(
            "{}: QCSchema files should have a `schema_version` key."
            "Attempting to load without version number.".format(lit.filename),
            FileFormatWarning,
            2,
        )

    if schema_name == "qcschema_molecule":
        return _load_qcschema_molecule(result, lit)
    if schema_name == "qcschema_basis":
        return _load_qcschema_basis(result, lit)
    if schema_name == "qcschema_input":
        return _load_qcschema_input(result, lit)
    if schema_name == "qcschema_output":
        return _load_qcschema_output(result, lit)
    raise FileFormatError(
        "{}: Invalid QCSchema type {}, should be one of `qcschema_molecule`, `qcschema_basis`,"
        "`qcschema_input`, or `qcschema_output".format(lit.filename, result["schema_name"])
    )


def _load_qcschema_molecule(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_molecule properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    molecule_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``,
        ``nelec`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``g_rot``, ``title`` & ``extra``
        keys and corresponding values as well.

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
        The 'molecule' key from the QCSchema input or output file, or the full result for a QCSchema
        Molecule file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    topology_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``,
        ``nelec`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``g_rot``, ``title`` & ``extra``
        keys and corresponding values as well.

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

    # Geometry is in a flattened list, convert to N x 3
    topology_dict["atcoords"] = np.array(mol["geometry"]).reshape(-1, 3)
    atnums = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
    topology_dict["atnums"] = atnums
    atcorenums = atnums.astype(float)
    topology_dict["atcorenums"] = atcorenums
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
        atcorenums[~np.array(mol["real"])] = 0.0
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
        extra_dict["fragments"] = {"indices": [np.array(fragment) for fragment in fragments]}
        if "fragment_charges" in mol:
            extra_dict["fragments"]["charges"] = np.array(mol["fragment_charges"])
        if "fragment_multiplicities" in mol:
            extra_dict["fragments"]["multiplicities"] = np.array(mol["fragment_multiplicities"])
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
        extra_dict["atomic_numbers"] = np.array(mol["atomic_numbers"])
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
    passthrough_keys = _find_passthrough_keys(mol, molecule_keys)
    if passthrough_keys:
        topology_dict["extra"]["unparsed"] = passthrough_keys

    return topology_dict


def _find_passthrough_keys(result: dict, keys: set) -> dict:
    """Find all keys not specified for a given schema.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    keys
        The set of expected keys for a given schema type.

    Returns
    -------
    passthrough_dict
        All unparsed keys remaining in the parsed dict.
    """
    # Avoid altering original dict
    result = result.copy()

    passthrough_dict = dict()
    parsed_keys = keys.intersection(result.keys())
    for key in parsed_keys:
        del result[key]
    if len(result) > 0:
        passthrough_dict = result

    return passthrough_dict


# pylint: disable=unused-argument
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
    # basis_dict = dict()
    # return basis_dict
    raise NotImplementedError("qcschema_basis is not yet implemented in IOData.")


def _parse_basis_keys(basis: dict, lit: LineIterator) -> dict:
    """Parse basis keys for a QCSchema input, output, or basis file.

    Parameters
    ----------
    basis
        The basis dictionary from a QCSchema basis file or QCSchema input or output 'method' key.
    lit
        The line iterator holding the file data.

    Returns
    -------
    basis_dict
        Dictionary containing ...
    """
    raise NotImplementedError("")


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
    extra_dict = dict()
    input_dict = _parse_input_keys(result, lit)
    extra_dict["input"] = input_dict["extra"]

    if "molecule" not in result:
        raise FileFormatError("{}: QCSchema Input requires 'molecule' key".format(lit.filename))
    molecule_dict = _parse_topology_keys(result["molecule"], lit)
    input_dict.update(molecule_dict)
    extra_dict["molecule"] = molecule_dict["extra"]
    input_dict["extra"] = extra_dict
    input_dict["extra"]["schema_name"] = "qcschema_input"

    return input_dict


def _parse_input_keys(result: dict, lit: LineIterator) -> dict:
    """Parse input keys for QCSchema input or output files.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    input_dict
        Output dictionary containing ``lot`` and ``obasis_name`` keys and corresponding values.
        It may contain ``obasis`` and ``extra`` keys and corresponding values as well.

    """
    # QCEngineRecords input files don't actually specify a name or version
    should_be_required_keys = {"schema_name", "schema_version"}
    input_keys = {"molecule", "driver", "model"}
    for key in should_be_required_keys:
        if key not in result:
            warn(
                "{}: QCSchema files should have a '{}' key.".format(lit.filename, key),
                FileFormatWarning,
                2,
            )
    for key in input_keys:
        if key not in result:
            raise FileFormatError(
                "{}: QCSchema `qcschema_input` file requires '{}' key".format(lit.filename, key)
            )
    # Store all extra keys in extra_dict and gather at end
    input_dict = dict()
    extra_dict = dict()

    # Save schema name & version
    extra_dict["schema_name"] = "qcschema_input"
    try:
        version = result["schema_version"]
    except KeyError:
        version = -1
    if float(version) < 1.0:
        warn(
            "{}: Unknown `qcschema_input` version {}, loading may produce invalid results".format(
                lit.filename, version
            ),
            FileFormatWarning,
            2,
        )
    extra_dict["schema_version"] = version

    # Load driver
    extra_dict["driver"] = _parse_driver(result["driver"], lit)

    # Load model & call basis helper if needed
    model = _parse_model(result["model"], lit)
    input_dict.update(model)
    extra_dict["model"] = model["extra"]

    # Load keywords & store
    # Currently, only the IOData run_type attribute is specifically parsed from keywords, but this
    # is a good space for passing additional IOData-specific keywords, given that the official spec
    # treats this as program-specific territory.
    if "keywords" in result:
        keywords_dict = result["keywords"]
        if "run_type" in keywords_dict:
            input_dict["run_type"] = keywords_dict["run_type"]
        extra_dict["keywords"] = keywords_dict
    # Check for extras
    if "extras" in result:
        extra_dict["extras"] = result["extras"]
    # Check for ID
    if "id" in result:
        extra_dict["id"] = result["id"]
    # Load protocols
    if "protocols" in result:
        extra_dict["protocols"] = _parse_protocols(result["protocols"], lit)
    # Check for provenance
    if "provenance" in result:
        extra_dict["provenance"] = _parse_provenance(result["provenance"], lit, "qcschema_input")

    input_dict["extra"] = extra_dict

    input_keys = {
        "schema_name",
        "schema_version",
        "molecule",
        "driver",
        "model",
        "extras",
        "id",
        "keywords",
        "protocols",
        "provenance",
    }
    passthrough_keys = _find_passthrough_keys(result, input_keys)
    if passthrough_keys:
        input_dict["extra"]["unparsed"] = passthrough_keys

    return input_dict


def _parse_driver(driver: str, lit: LineIterator) -> str:
    """Load driver properties from QCSchema.

    Parameters
    ----------
    driver
        The `driver` key from the QCSchema input.
    lit
        The line iterator holding the file data.

    Returns
    -------
    driver_dict
        The driver for the QCSchema file, specifying what type of calculation is being performed.

    Raises
    ------
    FileFormatError
        If driver is not one of {"energy", "gradient", "hessian", "properties"}.

    Notes
    -----
    This keyword is similar to, but not really interchangeable with, the `run_type` IOData
    attribute. In order to specify the `run_type`, add it to the `keywords` dictionary.

    """
    if driver not in ["energy", "gradient", "hessian", "properties"]:
        raise FileFormatError(
            "{}: QCSchema driver must be one of `energy`, `gradient`, `hessian`, "
            "or `properties`".format(
                lit.filename
            )
        )
    else:
        return driver


def _parse_model(model: dict, lit: LineIterator) -> dict:
    """Load model properties from QCSchema.

    Parameters
    ----------
    model
        The dictionary corresponding to the 'model' key for a QCSchema input or output file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    model_dict
        Output dictionary containing ``lot`` and ``obasis_name`` keys and corresponding values.
        It may contain ``obasis`` key and corresponding values as well.

    """
    model_dict = dict()
    extra_dict = dict()

    if "method" not in model:
        raise FileFormatError("{}: QCSchema `model` requires a `method`".format(lit.filename))
    else:
        model_dict["lot"] = model["method"]
    # QCEngineRecords doesn't give an empty string for basis-free methods, omits req'd key instead
    if "basis" not in model:
        warn(
            "{}: Model `basis` key should be given. Assuming basis-free method.".format(
                lit.filename
            )
        )
    elif isinstance(model["basis"], str):
        if model["basis"] == "":
            warn(
                "{}: QCSchema `basis` could not be read and will be omitted."
                "Unless model is for a basis-free method, check input file.".format(lit.filename),
                FileFormatWarning,
                2,
            )
        else:
            model_dict["obasis_name"] = model["basis"]
    elif isinstance(model["basis"], dict):
        basis = _parse_basis_keys(model["basis"], lit)
        model_dict.update(basis)
        extra_dict["basis"] = basis["extra"]
        pass

    model_dict["extra"] = extra_dict
    return model_dict


def _parse_protocols(protocols: dict, lit: LineIterator) -> dict:
    """Load protocols properties from QCSchema.

    Parameters
    ----------
    protocols
        Protocols key from a QCSchema input or output file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    protocols_dict
        Protocols dictionary containing instructions for the manipulation of output generated from
        this input.

    """
    if "wavefunction" not in protocols:
        warn(
            "{}: Protocols `wavefunction` key not specified, no properties will be kept.",
            FileFormatWarning,
            2
        )
        wavefunction = "none"
    else:
        wavefunction = protocols["wavefunction"]
    if "stdout" not in protocols:
        warn(
            "{}: Protocols `stdout` key not specified, stdout will be kept.",
            FileFormatWarning,
            2
        )
        keep_stdout = True
    else:
        keep_stdout = protocols["stdout"]
    protocols_dict = dict()
    if wavefunction not in {"all", "orbitals_and_eigenvalues", "return_results", "none"}:
        raise FileFormatError(
            "{}: Invalid `protocols` `wavefunction` keyword.".format(lit.filename)
        )
    else:
        protocols_dict["keep_wavefunction"] = wavefunction
    if not isinstance(keep_stdout, bool):
        raise FileFormatError("{}: `protocols` `stdout` option must be a boolean.")
    else:
        protocols_dict["keep_stdout"] = keep_stdout
    return protocols_dict


# pylint: disable=unused-argument
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
    # basis_dict = dict()
    # return basis_dict
    raise NotImplementedError("qcschema_output is not yet implemented in IOData.")


def _parse_provenance(
    provenance: Union[List[dict], dict], lit: LineIterator, source: str, append=True
) -> Union[List[dict], dict]:
    """Load provenance properties from QCSchema.

    Parameters
    ----------
    provenance
        QCSchema JSON provenance dictionary.
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


@document_dump_one(
    "QCSchema",
    ["atnums", "atcoords", "charge", "spinpol"],
    ["title", "atcorenums", "atmasses", "bonds", "g_rot", "extra"])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    if "schema_name" not in data.extra:
        raise FileFormatError("Cannot write qcschema file without 'schema_name' defined.")
    schema_name = data.extra["schema_name"]

    if schema_name == "qcschema_molecule":
        return_dict = _dump_qcschema_molecule(data)
    elif schema_name == "qcschema_basis":
        raise NotImplementedError("{} not yet implemented in IOData.".format(schema_name))
        # return_dict = _dump_qcschema_basis(data)
    elif schema_name == "qcschema_input":
        raise NotImplementedError("{} not yet implemented in IOData.".format(schema_name))
        # return_dict = _dump_qcschema_input(data)
    elif schema_name == "qcschema_input":
        raise NotImplementedError("{} not yet implemented in IOData.".format(schema_name))
        # return_dict = _dump_qcschema_output(data)
    else:
        raise FileFormatError(
            "'schema_name' must be one of 'qcschema_molecule', 'qcschema_basis'"
            "'qcschema_input' or 'qcschema_output'."
        )
    json.dump(return_dict, f, indent=4)


def _dump_qcschema_molecule(data: IOData) -> dict:
    """Dump relevant attributes from IOData to qcschema_molecule.

    Parameters
    ----------
    data
        The IOData instance to dump to file.

    Returns
    -------
    molecule_dict
        The dict that will produce the QCSchema JSON file.

    """
    molecule_dict = {"schema_name": "qcschema_molecule", "schema_version": 2}

    # Gather required field data
    if data.atnums is None or data.atcoords is None:
        raise FileFormatError("qcschema_molecule requires `atnums` and `atcoords` fields.")
    molecule_dict["symbols"] = [num2sym[num] for num in data.atnums]
    molecule_dict["geometry"] = list(data.atcoords.flatten())

    # Should be required field data
    if data.charge is None or data.spinpol is None:
        warn(
            "`charge` and `spinpol` should be given to write qcschema_molecule file:"
            "QCSchema defaults to charge = 0 and multiplicity = 1 if no values given.",
            FileFormatWarning,
            2,
        )
    if data.charge is not None:
        molecule_dict["molecular_charge"] = data.charge
    if data.spinpol is not None:
        molecule_dict["molecular_multiplicity"] = data.spinpol + 1

    # Check for other QCSchema keys from IOData keys
    if data.title:
        molecule_dict["name"] = data.title
    molecule_dict["real"] = [bool(atcorenum != 0) for atcorenum in data.atcorenums]
    # "masses" could be overwritten below (for QCSchema passthrough)
    if data.atmasses is not None:
        molecule_dict["masses"] = data.atmasses.tolist()
    if data.bonds is not None:
        molecule_dict["connectivity"] = [[int(i) for i in bond] for bond in data.bonds]
    if data.g_rot:
        molecule_dict["fix_symmetry"] = data.g_rot

    # Check for other QCSchema keys from IOData extra dict
    if "qcel_validated" in data.extra:
        molecule_dict["validated"] = data.extra["qcel_validated"]
    if "identifiers" in data.extra:
        molecule_dict["identifiers"] = data.extra["identifiers"]
    if "comment" in data.extra:
        molecule_dict["comment"] = data.extra["comment"]
    if "atom_labels" in data.extra:
        molecule_dict["atom_labels"] = data.extra["atom_labels"]
    if "atomic_numbers" in data.extra:
        molecule_dict["atomic_numbers"] = data.extra["atomic_numbers"].tolist()
    if "masses" in data.extra:
        molecule_dict["masses"] = data.extra["masses"].tolist()
    if "mass_numbers" in data.extra:
        molecule_dict["mass_numbers"] = data.extra["mass_numbers"].tolist()
    if "fragments" in data.extra:
        if "indices" in data.extra["fragments"]:
            molecule_dict["fragments"] = [
                fragment.tolist() for fragment in data.extra["fragments"]["indices"]
            ]
        if "indices" in data.extra["fragments"]:
            molecule_dict["fragment_charges"] = data.extra["fragments"]["charges"].tolist()
        if "indices" in data.extra["fragments"]:
            molecule_dict["fragment_multiplicities"] = \
                data.extra["fragments"]["multiplicities"].tolist()
    if "fix_com" in data.extra:
        molecule_dict["fix_com"] = data.extra["fix_com"]
    if "fix_orientation" in data.extra:
        molecule_dict["fix_orientation"] = data.extra["fix_orientation"]
    if "provenance" in data.extra:
        molecule_dict["provenance"] = data.extra["provenance"]
    else:
        molecule_dict["provenance"] = {
            "creator": "IOData",
            "version": __version__,
            "routine": "iodata.formats.json",
        }
    if "id" in data.extra:
        molecule_dict["id"] = data.extra["id"]
    if "extras" in data.extra:
        molecule_dict["extras"] = data.extra["extras"]
    if "unparsed" in data.extra:
        for k in data.extra["unparsed"]:
            molecule_dict[k] = data.extra["unparsed"][k]

    return molecule_dict
