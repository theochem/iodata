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
from warnings import warn

__all__ = []


PATTERNS = ["*.json"]


CCA_CONVENTIONS = {
    (0, "c"): ["1"],
    (1, "c"): ["x", "y", "z"],
    (2, "c"): ["xx", "xy", "xz", "yy", "yz", "zz"],
    (2, "p"): ["s2", "s1", "c0", "c1", "c2"],
    (3, "c"): ["xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"],
    (3, "p"): ["s3", "s2", "s1", "c0", "c1", "c2", "c3"],
    (4, "c"): ["xxxx", "xxxy", "xxxz", "xxyy", "xxyz", "xxzz", "xyyy", "xyyz",
               "xyzz", "xzzz", "yyyy", "yyyz", "yyzz", "yzzz", "zzzz"],
    (4, "p"): ["s4", "s3", "s2", "s1", "c0", "c1", "c2", "c3", "c4"],
    (5, "c"): ["xxxxx", "xxxxy", "xxxxz", "xxxyy", "xxxyz", "xxxzz", "xxyyy", "xxyyz",
               "xxyzz", "xxzzz", "xyyyy", "xyyyz", "xyyzz", "xyzzz", "xzzzz", "yyyyy",
               "yyyyz", "yyyzz", "yyzzz", "yzzzz", "zzzzz"],
    (5, "p"): ["s5", "s4", "s3", "s2", "s1", "c0", "c1", "c2", "c3", "c4", "c5"],
    (6, "c"): ["xxxxxx", "xxxxxy", "xxxxxz", "xxxxyy", "xxxxyz", "xxxxzz", "xxxyyy", "xxxyyz",
               "xxxyzz", "xxxzzz", "xxyyyy", "xxyyyz", "xxyyzz", "xxyzzz", "xxzzzz", "xyyyyy",
               "xyyyyz", "xyyyzz", "xyyzzz", "xyzzzz", "xzzzzz", "yyyyyy", "yyyyyz", "yyyyzz",
               "yyyzzz", "yyzzzz", "yzzzzz", "zzzzzz"],
    (6, "p"): ["s6", "s5", "s4", "s3", "s2", "s1", "c0", "c1", "c2", "c3", "c4", "c5", "c6"],
}


@document_load_one("JSON", ["atcoords", "atnums", "atcorenums", "mo", "obasis"], ["title"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    json_in = json.load(lit._f)
    result = _parse_json(json_in, lit)
    return result


def _parse_json(json_in: dict, lit: LineIterator) -> dict:
    """Parse data from QCSchema JSON input file.

    QCSchema supports three different schema types: `qcschema_molecule`, specifying one or more
    molecules in a single system; `qcschema_input`, specifying input to a QC program for a specific
    system; and `qcschema_output`, specifying results of a QC program calculation for a specific
    system along with the input information.

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

    # Determine schema type
    if "schema_name" not in result:
        raise FileFormatError("{}: QCSchema file requires `schema_name` key".format(lit.filename))
    elif "schema_version" not in result:
        raise FileFormatError(
            "{}: QCSchema file requires `schema_version` key".format(lit.filename)
        )
    if result["schema_name"] == "qcschema_molecule":
        return _load_qcschema_molecule(result, lit)
    elif result["schema_name"] == "qcschema_input":
        return _load_qcschema_input(result, lit)
    elif result["schema_name"] == "qcschema_output":
        return _load_qcschema_output(result, lit)
    else:
        raise FileFormatError(
            "{}: Invalid QCSchema type {}, should be one of `qcschema_molecule`, "
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
    topology_dict
        Output dictionary containing ``atcoords`` & ``atnums`` keys and corresponding values.
        It may contain ``atcorenums``, ``atmasses``, ``bonds``, ``charge`` & ``extra`` keys
        and corresponding values as well.
    """
    # All Topology properties are found in the "molecule" key
    if "molecule" not in result:
        raise FileFormatError(
            "{}: QCSchema `qcschema_molecule` file requires 'molecule' key".format(lit.filename)
        )
    topology_dict = _parse_topology_keys(lit, result["molecule"])

    return topology_dict


def _load_qcschema_input(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_input properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    """
    input_keys = ["schema_name", "schema_version", "molecule", "driver", "keywords", "model"]
    for key in input_keys:
        if key not in result:
            raise FileFormatError(
                "{}: QCSchema `qcschema_input` file requires '{}' key".format(lit.filename, key)
            )
    # Store all extra keys in extra_dict and gather at end
    input_dict = dict()
    extra_dict = dict()

    # Load molecule schema
    topology_dict = _parse_topology_keys(result["molecule"], lit)
    input_dict.update(topology_dict)
    extra_dict.update(topology_dict["extra"])

    # Save schema name & version
    extra_dict["schema_name"] = result["schema_name"]
    version = result["schema_version"]
    # TODO: check this once QCSchema is fully specified
    if float(version) != 1.0:
        warn(
            "{}: Unknown `qcschema_input` version {}, loading may produce invalid results".format(
                lit.filename, version
            ),
            FileFormatWarning,
            2,
        )
    extra_dict["schema_version"] = version

    # Load driver
    driver = result["driver"]
    if driver not in ["energy", "gradient", "hessian", "properties"]:
        raise FileFormatError(
            "{}: QCSchema driver must be one of `energy`, `gradient`, `hessian`, or `properties`".format(
                lit.filename
            )
        )
    else:
        extra_dict["driver"] = driver

    # Load keywords
    extra_dict["keywords"] = result["keywords"]

    # Load model & call basis helper if needed
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
            "{}: QCSchema `model` requires a `basis`, either as a string or a `qcschema_basis` instance".format(
                lit.filename
            )
        )
    elif isinstance(model["basis"], str):
        input_dict["obasis_name"] = model["basis"]
    elif isinstance(model["basis"], dict):
        basis = _parse_basis_keys(model["basis"], lit)
        input_dict.update(basis)
        extra_dict["basis"] = basis["extra"]

    else:
        warn(
            "{}: QCSchema `basis` could not be read. "
            "Unless model is for a basis-free method, check input file.".format(lit.filename),
            FileFormatWarning,
            2,
        )

    input_dict["extra"] = extra_dict
    return input_dict


def _load_qcschema_output(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_output properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    """
    input_keys = [
        "schema_name",
        "schema_version",
        "molecule",
        "driver",
        "keywords",
        "model",
        "provenance",
        "properties",
        "success",
        "return_result",
    ]
    for key in input_keys:
        if key not in result:
            raise FileFormatError(
                "{}: QCSchema `qcschema_input` file requires '{}' key".format(lit.filename, key)
            )
    # Store all extra keys in extra_dict and gather at end
    output_dict = dict()
    extra_dict = dict()
    # Load molecule schema
    topology_dict = _parse_topology_keys(lit, result["molecule"])
    output_dict.update(topology_dict)
    extra_dict.update(topology_dict["extra"])
    # Validate schema name & version

    # Load driver

    # Load keywords

    # Load model & call basis helper if needed

    # Save provenance

    # Check success

    # Load return_result

    output_dict["extra"] = extra_dict
    return output_dict


def _parse_topology_keys(mol: dict, lit: LineIterator) -> dict:
    """Load topology properties from QCSchema.

    The qcschema_molecule v2 specification requires a topology for every file, specified in the
    `molecule` key, containing at least the keys `schema_name`, `schema_version`, `symbols`,
    `geometry`, `molecular_charge`, `molecular_multiplicity`, and `provenance`. This schema is
    currently used in QCElemental (and thus the QCArchive ecosystem).

    qcschema_molecule v1 only exists as the specification on the QCSchema website, and seems never
    to have been implemented in QCArchive. It is possible to accept v1 input, since all required
    keys for v2 exist in v1, but it is preferable to convert those files to v2 explicitly.

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
    topology_keys = [
        "schema_name",
        "schema_version",
        "symbols",
        "geometry",
        "molecular_charge",
        "molecular_multiplicity",
    ]
    for key in topology_keys:
        if key not in mol:
            raise FileFormatError(
                "{}: QCSchema topology requires '{}' key".format(lit.filename, key)
            )

    topology_dict = dict()
    topology_dict["atnums"] = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
    # Geometry is in a flattened list, convert to N x 3
    topology_dict["atcoords"] = np.array(
        [mol["geometry"][3 * i : (3 * i) + 3] for i in range(0, len(mol["geometry"]) // 3)]
    )
    topology_dict["charge"] = mol["molecular_charge"]
    # FIXME: deal with multiplicity
    mult = mol["molecular_multiplicity"]
    # FIXME ^^^^^

    # Check for optional keys
    # Load atom masses to array, canonical weights assumed if masses not given
    # FIXME: add mass_numbers and deal with -1 case
    if "masses" in mol:
        topology_dict["atmasses"] = np.array(mol["masses"])
    elif "mass_numbers" in mol:
        mass_numbers = mol["mass_numbers"]
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


def _parse_basis_keys(basis: dict, lit: LineIterator) -> dict:
    """Load input properties from QCSchema.

    Parameters
    ----------
    basis
        The JSON dict loaded from file. FIXME: doc
    lit
        The line iterator holding the file data.

    Returns
    -------
    basis_dict
        Output dictionary containing ... keys and corresponding values.
        It may contain ... keys
        and corresponding values as well.

    Notes
    -----
    It might be necessary to pass the ghost array to this function for validation.

    """
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
                "{}: QCSchema `qcschema_basis` requires '{}' key".format(lit.filename, key)
            )
    basis_dict = dict()
    extra_dict = dict()
    # No STOs in iodata
    if basis["function_type"].lower() == "sto":
        raise FileFormatError(
            "{}: Slater-type orbitals are not supported by IOData".format(lit.filename)
        )
    extra_dict["schema_name"] = basis["schema_name"]
    extra_dict["schema_version"] = basis["schema_version"]
    basis_dict["obasis_name"] = basis["name"]
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
        if "electron_shells" not in center:
            raise FileFormatError(
                "{}: Basis center {} requires `electron_shells` key".format(lit.filename, center)
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
            coeffs = np.array([[float(x) for x in segment] for segment in shell["coefficients"]])
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

    # These are assumed within QCSchema
    conventions = CCA_CONVENTIONS
    prim_norm = "L2"

    basis_dict["obasis"] = MolecularBasis(
        shells=tuple(obasis_shells), conventions=conventions, primitive_normalization=prim_norm
    )

    return basis_dict


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
    properties = _parse_properties(result["properties"])

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


def _parse_properties(properties: dict) -> dict:
    """Load all property keys.

    Parameters
    ----------
    properties
        All QC program calculation results stored in a QCSchema instance.

    Returns
    -------
    properties_dict
        Parsed QC program calculation results.

    Notes
    -----
    CalcInfo properties:
    calcinfo_nbasis
        The number of basis functions for the computation.

    calcinfo_nmo
        The number of molecular orbitals for the computation.

    calcinfo_nalpha
        The number of alpha electrons in the computation.

    calcinfo_nbeta
        The number of beta electrons in the computation.

    calcinfo_natom
        The number of atoms in the computation.

    return_energy
        The energy of the requested method, identical to `return_value` for energy computations.

    SCF properties:
    scf_one_electron_energy
        The one-electron (core Hamiltonian) energy contribution to the total SCF energy.,

    scf_two_electron_energy
        The two-electron energy contribution to the total SCF energy.

    nuclear_repulsion_energy
        The nuclear repulsion energy contribution to the total SCF energy.

    scf_vv10_energy
        The VV10 functional energy contribution to the total SCF energy.

    scf_xc_energy
        The functional energy contribution to the total SCF energy.

    scf_dispersion_correction_energy
        The dispersion correction appended to an underlying functional
        when a DFT-D method is requested.

    scf_dipole_moment
        The X, Y, and Z dipole components.,

    scf_total_energy
        The total electronic energy of the SCF stage of the calculation.
        This is represented as the sum of the ... quantities.

    scf_iterations
        The number of SCF iterations taken before convergence.

    MP2 properties
    mp2_same_spin_correlation_energy
        The portion of MP2 doubles correlation energy from same-spin (i.e. triplet) correlations,
        without any user scaling.

    mp2_opposite_spin_correlation_energy
        The portion of MP2 doubles correlation energy from opposite-spin (i.e. singlet)
        correlations, without any user scaling.

    mp2_singles_energy
        The singles portion of the MP2 correlation energy. Zero except in ROHF.

    mp2_doubles_energy
        The doubles portion of the MP2 correlation energy including same-spin
        and opposite-spin correlations.

    mp2_correlation_energy
        The MP2 correlation energy.

    mp2_total_energy
        The total MP2 energy (MP2 correlation energy + HF energy).

    mp2_dipole_moment
        The MP2 X, Y, and Z dipole components.

    CC properties
    ccsd_same_spin_correlation_energy
        The portion of CCSD doubles correlation energy from same-spin (i.e. triplet)
        correlations, without any user scaling.

    ccsd_opposite_spin_correlation_energy
        The portion of CCSD doubles correlation energy from opposite-spin (i.e. singlet)
        correlations, without any user scaling.

    ccsd_singles_energy
        The singles portion of the CCSD correlation energy. Zero except in ROHF.

    ccsd_doubles_energy
        The doubles portion of the CCSD correlation energy including same-spin and
        opposite-spin correlations.

    ccsd_correlation_energy
        The CCSD correlation energy.

    ccsd_total_energy
        The total CCSD energy (CCSD correlation energy + HF energy).

    ccsd_prt_pr_correlation_energy
        The CCSD(T) correlation energy.

    ccsd_prt_pr_total_energy
        The total CCSD(T) energy (CCSD(T) correlation energy + HF energy).

    ccsdt_correlation_energy
        The CCSDT correlation energy.

    ccsdt_total_energy
        The total CCSDT energy (CCSDT correlation energy + HF energy).

    ccsdtq_correlation_energy
        The CCSDTQ correlation energy.

    ccsdtq_total_energy
        The total CCSDTQ energy (CCSDTQ correlation energy + HF energy).

    ccsd_dipole_moment
        The CCSD X, Y, and Z dipole components.,

    ccsd_prt_pr_dipole_moment
        The CCSD(T) X, Y, and Z dipole components.,

    ccsdt_dipole_moment
        The CCSDT X, Y, and Z dipole components.,

    ccsdtq_dipole_moment
        The CCSDTQ X, Y, and Z dipole components.,

    ccsd_iterations
        The number of CCSD iterations taken before convergence.

    ccsdt_iterations
        The number of CCSDT iterations taken before convergence.

    ccsdtq_iterations
        The number of CCSDTQ iterations taken before convergence.

    """
    properties_dict = dict()
    properties_dict["extra"] = dict()

    # Load any calcinfo properties
    # TODO: place these
    if "calcinfo_nbasis" in properties:
        nbasis = int(properties["calcinfo_nbasis"])
    if "calcinfo_nmo" in properties:
        nmo = int(properties["calcinfo_nmo"])
    if "calcinfo_nalpha" in properties:
        nalpha = int(properties["calcinfo_nalpha"])
    if "calcinfo_nbeta" in properties:
        nbeta = int(properties["calcinfo_nbeta"])
    if "calcinfo_natom" in properties:
        natom = int(properties["calcinfo_natom"])
    if "calcinfo_return_energy" in properties:
        properties_dict["energy"] = float(properties["return_energy"])

    # Load SCF properties
    scf_keys = {
        "scf_one_electron_energy",
        "scf_two_electron_energy",
        "nuclear_repulsion_energy",
        "scf_vv10_energy",
        "scf_xc_energy",
        "scf_dispersion_correction_energy",
        "scf_dipole_moment",
        "scf_total_energy",
        "scf_iterations",
    }
    scf_dict = dict()
    for scf_key in scf_keys.intersection(properties):
        value = properties[scf_key]
        # Ensure correct output type
        if scf_key == "scf_iterations":
            scf_dict["iterations"] = int(value)
        elif scf_key == "scf_dipole_moment":
            scf_dict["dipole_moment"] = [float(x) for x in value]
        else:
            # Slice out the "scf_" part
            if scf_key[:4] == "scf_":
                scf_key = scf_key[4:]
            scf_dict[scf_key] = float(value)

    # Load MP properties (apparently only MP2 right now)
    mp_keys = {
        "mp2_same_spin_correlation_energy",
        "mp2_opposite_spin_correlation_energy",
        "mp2_singles_energy",
        "mp2_doubles_energy",
        "mp2_correlation_energy",
        "mp2_total_energy",
        "mp2_dipole_moment",
    }
    mp_dict = dict()
    for mp_key in mp_keys.intersection(properties):
        value = properties[mp_key]
        if mp_key == "mp2_dipole_moment":
            mp_dict["dipole_moment"] = [float(x) for x in value]
        else:
            # Slice out the "mp_" part
            if mp_key[:4] == "mp2_":
                mp_key = mp_key[4:]
            mp_dict[mp_key] = float(value)

    # Load CC properties
    cc_keys = {
        "ccsd_same_spin_correlation_energy",
        "ccsd_opposite_spin_correlation_energy",
        "ccsd_singles_energy",
        "ccsd_doubles_energy",
        "ccsd_correlation_energy",
        "ccsd_total_energy",
        "ccsd_prt_pr_correlation_energy",
        "ccsd_prt_pr_total_energy",
        "ccsdt_correlation_energy",
        "ccsdt_total_energy",
        "ccsdtq_correlation_energy",
        "ccsdtq_total_energy",
        "ccsd_dipole_moment",
        "ccsd_prt_pr_dipole_moment",
        "ccsdt_dipole_moment",
        "ccsdtq_dipole_moment",
        "ccsd_iterations",
        "ccsdt_iterations",
        "ccsdtq_iterations",
    }
    cc_dict = dict()
    for cc_key in cc_keys.intersection(properties):
        value = properties[cc_key]
        # Slice out the "cc_" part
        if cc_key[:3] == "cc_":
            cc_key = cc_key[3:]
        if cc_key[-10:] == "iterations":
            cc_dict[cc_key] = int(value)
        elif cc_key[-13:] == "dipole_moment":
            cc_dict[cc_key] = [float(x) for x in value]
        else:
            cc_dict[cc_key] = float(value)

    # Bundle all dicts
    # TODO: deal with moments
    if scf_dict:
        properties_dict["extra"]["scf"] = scf_dict
    if mp_dict:
        properties_dict["extra"]["mp"] = mp_dict
    if cc_dict:
        properties_dict["extra"]["cc"] = cc_dict
    return properties_dict
