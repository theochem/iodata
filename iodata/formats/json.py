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


from typing import Tuple, Union, TextIO
import json

import numpy as np

from ..basis import Shell, MolecularBasis
from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..orbitals import MolecularOrbitals
from ..utils import angstrom, FileFormatError, FileFormatWarning, LineIterator
import warnings

__all__ = []


PATTERNS = ["*.json"]


# First dict: guaranteed to be loaded; second dict, loaded if present
@document_load_one("JSON", ["atcoords", "atnums", "atcorenums", "mo", "obasis"], ["title"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    result = json.load(lit._f)
    result = _parse_json(lit, result)
    return result


def _parse_json(lit: LineIterator, json_in: dict):
    """Parse data from QCSchema JSON input file.

    QCSchema supports three different schema types: `topology`, specifying one or more molecules;
    `input`, specifying input to a QC program in addition to `topology`; and `output`, specifying
    results of a QC program calculation in addition to `topology` and `input`.

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
    # Remove all null entries in json
    # QCEngine seems to add null entries even for optional and empty keys
    fix_keys = {k: v for k, v in json_in.items() if v is not None}
    fix_subkeys = dict()
    for key in fix_keys:
        if type(fix_keys[key]) == dict:
            fix_subkeys[key] = {k: v for k, v in fix_keys[key].items() if v is not None}
    result = {**fix_keys, **fix_subkeys}
    # Put all extras into a separate dict and combine at end
    parsed_result = dict()
    extras_dict = dict()
    # Load topology (required), input (optional), output (optional)
    topology_dict = _parse_topology_keys(lit, result)
    parsed_result.update(topology_dict)
    if "extra" in topology_dict:
        extras_dict.update(topology_dict["extra"])
    input_dict = _parse_input_keys(lit, result)
    parsed_result.update(input_dict)
    if "extra" in input_dict:
        extras_dict.update(input_dict["extra"])
    # All input keys are required for output
    if input_dict:
        output_dict = _parse_output_keys(lit, result, input_dict["extra"]["driver"])
        parsed_result.update(output_dict)
        if "extra" in output_dict:
            extras_dict.update(output_dict["extra"])
    parsed_result["extra"] = extras_dict

    return parsed_result


def _parse_topology_keys(lit: LineIterator, result: dict) -> dict:
    """Load topology properties from QCSchema.

    The QCSchema specification requires a topology for every file, specified in the `molecule` key,
    containing at least the keys `symbols` (``atnums`` in `IOData`) and `geometry` (``atcoords`` in
    `IOData`).

    Parameters
    ----------
    lit
        The line iterator holding the file data.
    result
        The JSON dict loaded from file.

    Returns
    -------
    topology_dict
        Output dictionary containing ``atcoords`` & ``atnums`` keys and corresponding values.
        It may contain ``atcorenums``, ``atmasses``, ``bonds``, ``charge`` & ``extra`` keys
        and corresponding values as well.

    """
    # All Topology properties are found in the "molecule" key
    if "molecule" in result:
        mol = result["molecule"]
    else:
        raise FileFormatError("{}: QCSchema file requires 'molecule' key".format(lit.filename))

    # TODO: move these
    # Make sure required topology properties are present
    if "symbols" not in mol:
        raise FileFormatError("{}: QCSchema topology requires 'symbols' key".format(lit.filename))
    elif "geometry" not in mol:
        raise FileFormatError("{}: QCSchema topology requires 'geometry' key".format(lit.filename))
    else:
        atnums = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
        # Geometry is in a flattened list, convert to N x 3
        atcoords = np.array(
            [mol["geometry"][3 * i : (3 * i) + 3] for i in range(0, len(mol["geometry"]) // 3)]
        )

    topology_dict = {
        "atnums": atnums,
        "atcoords": atcoords,
    }

    # Check for optional keys
    # Load atom masses to array, canonical weights assumed if masses not given
    # FIXME: add mass_numbers and deal with -1 case
    if "masses" in mol:
        topology_dict["atmasses"] = np.array(mol["masses"])
    elif "mass_numbers" in mol:
        mass_numbers = mol["mass_numbers"]
    # FIXME ^^^^^
    # Load molecular charge (required in v2.0)
    if "molecular_charge" in mol:
        topology_dict["charge"] = mol["molecular_charge"]
    # Load molecular multiplicity (required in v2.0)
    # FIXME: deal with multiplicity
    if "molecular_multiplicity" in mol:
        pass
    # FIXME ^^^^^
    # Load bonds: list of tuple (atom1, atom2, bond_order)
    # Note: The QCSchema spec allows for non-integer bond_orders, these are forced to integers here
    if "connectivity" in mol:
        topology_dict["bonds"] = np.array(mol["connectivity"], dtype=int)
    # Load ghost atoms
    # FIXME: do something with ghost atoms
    if "real" in mol:
        ghosts = mol["real"]
    # FIXME ^^^^^
    # Add extra keys
    extra = dict()
    # Check for fragment keys
    # List fragment indices in nested list (likely is a jagged array)
    if "fragments" in mol:
        fragments = mol["fragments"]
        extra["fragments"] = {"indices": fragments}
        if "fragment_charges" in mol:
            extra["fragments"]["charges"] = np.array(mol["fragment_charges"])
        if "fragment_multiplicities" in mol:
            extra["fragments"]["multiplicities"] = np.array(mol["fragment_multiplicities"])

    # Check for other extras
    # FIXME: Deal with extras
    # name, comment, fix_com, fix_symmetry, fix_orientation, provenance which should be mandatory
    if "name" in mol:
        extra["name"] = mol["name"]
    # FIXME ^^^^^
    if extra:
        topology_dict["extra"] = extra

    return topology_dict


def _parse_input_keys(lit: LineIterator, result: dict) -> dict:
    """Load input properties from QCSchema.

    Parameters
    ----------
    lit
        The line iterator holding the file data.
    result
        The JSON dict loaded from file.

    Returns
    -------
    input_dict
        Output dictionary containing ... keys and corresponding values.
        It may contain ... keys
        and corresponding values as well.

    Notes
    -----
    It might be necessary to pass the ghost array to this function for validation.

    """
    # Make sure required input keys are present
    # TODO: move these
    for input_key in ["schema_name", "schema_version", "driver", "keywords", "model"]:
        if input_key not in result:
            return dict()
    input_dict = dict()
    extra = dict()
    # Load schema_name, schema_version
    # NOTE: QCElemental's default version is 2, but QCSchema v2 is not documented elsewhere
    # TODO: make sure this doesn't overwrite the basis schema_name and _version
    extra["schema_name"] = result["schema_name"]
    extra["schema_version"] = result["schema_version"]
    # Check if driver is valid
    driver = result["driver"]
    if driver not in ["energy", "gradient", "hessian"]:
        raise FileFormatError(
            "{}: QCSchema driver must be one of `energy`, `gradient`, or `hessian`".format(
                lit.filename
            )
        )
    else:
        extra["driver"] = driver
    # Load any keywords
    # NOTE: there is no actual specification for how keywords are defined in the schema
    # TODO: deal with this crap
    extra["keywords"] = result["keywords"]
    # Load basis set name or basis specification
    # For the DFTD3 program, QCEngine may not define the basis
    model = result["model"]
    if "method" not in model:
        raise FileFormatError("{}: QCSchema `model` requires a `method`".format(lit.filename))
    else:
        method = model["method"]
        input_dict["lot"] = method
    if "basis" not in model:
        if method.upper().find("-D3") > -1:
            lit.warn(
                "You may be trying to load results from a `DFTD3` program calculation, which is"
                "not currently supported in `IOData`."
            )
        raise FileFormatError(
            "{}: QCSchema `model` must be defined using either `basis` or `basis_spec`".format(
                lit.filename
            )
        )
    elif isinstance(model["basis"], str):
        input_dict["obasis_name"] = model["basis"]
    elif isinstance(model["basis"], dict):
        # TODO: split this into a separate function to load the qcschema_basis
        # Load basis, consistent with v1 of `basis` spec found here:
        # https://github.com/MolSSI/QCSchema/blob/master/qcschema/dev/basis.py
        # and BasisSet in QCElemental
        basis = model["basis"]
        # Check for required properties:
        # NOTE: description is optional in QCElemental, required in v1.dev
        for key in [
            "schema_name",
            "schema_version",
            "name",
            "center_data",
            "atom_map",
            "function_type",
        ]:
            if key not in basis:
                raise FileFormatWarning(
                    "{}: QCSchema `basis` requires '{}' key".format(lit.filename, key)
                )
        # No STOs in iodata
        if basis["function_type"].lower() == "sto":
            raise FileFormatError(
                "{}: Slater-type orbitals are not supported by IOData".format(lit.filename)
            )
        extra["basis_schema_name"] = basis["schema_name"]
        extra["basis_schema_version"] = basis["schema_version"]
        input_dict["obasis_name"] = basis["name"]
        # Load basis data
        center_data = basis["center_data"]
        atom_map = basis["atom_map"]
        center_shells = dict()
        # Center_data is composed of basis_center, each has req'd electron_shells, ecp_electrons,
        # and optional ecp_potentials
        for center in center_data:
            # Initiate lists for building basis
            center_shells[center] = list()
            # QCElemental example omits ecp_electrons for cases with default value (0)
            # TODO: find ECP in iodata if exists
            if "electron_shells" not in center:
                raise FileFormatError(
                    "{}: Basis center {} requires `electron_shells` key".format(
                        lit.filename, center
                    )
                )
            if "ecp_electrons" in center:
                ecp_electrons = center["ecp_electrons"]
            else:
                ecp_electrons = 0
            shells = center["electron_shells"]
            for shell in shells:
                # electron_shell requires angular_momentum, harmonic_type, exponents, coefficients
                for key in ["angular_momentum", "harmonic_type", "exponents", "coefficients"]:
                    if key not in shell:
                        raise FileFormatError(
                            "{}: Basis center {} contains a shell missing '{}' key".format(
                                lit.filename, center, key
                            )
                        )
                # Load shell data
                # Convert exps and coeffs to float, BSE qcschema_basis v1 gives strings instead
                # TODO: test that these come out to the right thing
                angmoms = shell["angular_momentum"]
                exps = np.array([float(x) for x in shell["exponents"]])
                coeffs = np.array(
                    [[float(x) for x in segment] for segment in shell["coefficients"]]
                )
                # Check for single harmonic_type value (BSE qcschema_basis v1)
                if isinstance(shell["harmonic_type"], str):
                    if shell["harmonic_type"] not in ["cartesian", "spherical"]:
                        raise FileFormatError(
                            "{}: Basis center {} contains a shell with invalid `harmonic_type`".format(
                                lit.filename, center, key
                            )
                        )
                    else:
                        kinds = [shell["harmonic_type"][0] for _ in range(len(angmoms))]
                else:
                    if set(shell["harmonic_type"]) != {"cartesian", "spherical"}:
                        raise FileFormatError(
                            "{}: Basis center {} contains a shell with invalid `harmonic_type`".format(
                                lit.filename, center, key
                            )
                        )
                    else:
                        kinds = [shell["harmonic_type"][0] for kind in shell["harmonic_type"]]
                # Gather shell components
                center_shells[center].append(
                    {"angmoms": angmoms, "kinds": kinds, "exponents": exps, "coeffs": coeffs}
                )
        # Build obasis shells using the atom_map
        # Each atom in atom_map corresponds to a key in center_shells
        obasis_shells = list()
        for i, atom in enumerate(atom_map):
            for shell in center_shells[atom]:
                # Unpack angmoms, kinds, exponents, coeffs into obasis
                obasis_shells.append(Shell(icenter=i, **shell))
        # FIXME: need conventions, prim_norm
        conventions = {}
        prim_norm = "L2"
        # FIXME ^^^^^
        input_dict["obasis"] = MolecularBasis(
            shells=obasis_shells, conventions=conventions, primitive_normalization=prim_norm
        )

    else:
        raise FileFormatError("{}: QCSchema basis could not be read.".format(lit.filename))

    input_dict["extra"] = extra
    return input_dict


def _parse_output_keys(lit: LineIterator, result: dict, driver: str) -> dict:
    """Load output properties from QCSchema.

    Parameters
    ----------
    lit
        The line iterator holding the file data.
    result
        The JSON dict loaded from file.
    driver
        FIXME: doc

    Returns
    -------
    output_dict
        Output dictionary containing ... keys and corresponding values.
        It may contain ... keys
        and corresponding values as well.

    """
    # TODO: move these
    for output_key in ["provenance", "properties", "success", "return_result"]:
        if output_key not in result:
            return dict()
    output_dict = dict()
    extra = dict()
    # Load provenance data, can be either dict or list of dict, convert to list
    if isinstance(result["provenance"], dict):
        extra["provenance"] = list(result["provenance"])
    else:
        extra["provenance"] = result["provenance"]
    # Load properties (all calculation results)
    # TODO: load properties
    properties = result["properties"]

    # Check success
    extra["calc_success"] = bool(result["success"])

    # Load error data
    if "error" in result:
        pass

    # Load return result for driver
    if driver == "energy":
        extra["driver_result"] = float(result["return_result"])
    # Gradient and hessian are saved as flat lists, not nested
    elif driver == "gradient":
        extra["driver_result"] = np.array(
            [
                result["return_result"][3 * i : (3 * i) + 3]
                for i in range(0, len(result["return_result"]) // 3)
            ]
        )
    # Driver is already checked in input_dict, must be hessian here
    else:
        extra["driver_result"] = np.array(
            [
                result["return_result"][9 * i : (9 * i) + 9]
                for i in range(0, len(result["return_result"]) // 9)
            ]
        )

    output_dict["extra"] = extra
    return output_dict
