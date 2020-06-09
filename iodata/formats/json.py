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

from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..orbitals import MolecularOrbitals
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.json']


# First dict: guaranteed to be loaded; second dict, loaded if present
@document_load_one("JSON", ['atcoords', 'atnums', 'atcorenums', 'mo', 'obasis'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    result = json.load(lit._f)
    result = _parse_json(result)
    return result


def _parse_json(result: dict):
    """Parse data from QCSchema JSON input file.

    Parameters
    ----------
    result
        The JSON dict loaded from file.

    Returns
    -------
    out
        Output dictionary containing ...

    """
    topology = _parse_topology_keys(result)
    input = _parse_input_keys(result)
    output = _parse_output_keys(result)

    result = {**topology, **input, **output}

    return result


def _parse_topology_keys(result: dict) -> dict:
    """Load topology properties from QCSchema

    Parameters
    ----------
    result
        The JSON dict loaded from file.

    Returns
    -------
    out
        Output dictionary containing ``atcoords`` & ``atnums`` keys and corresponding values.
        It may contain ``atcorenums``, ``atmasses``, ``bonds`` & ``charge`` keys and corresponding
        values as well.

    Required keys: symbols (atnums), geometry (atcoords)
    Optional keys:
    comment                   extra
    real                      * Labels ghost atoms -> atnums and atcorenums (as 0)
    molecular_multiplicity    ????
    name                      * Could go in title
    mass_numbers              - Depreciated or not implemented
    molecular_charge          charge
    masses                    atmasses
    provenance                * Could go in title
    fix_com                   extra
    fix_orientation           extra
    atom_labels               extra
    connectivity              bonds
    atomic_numbers            - Depreciated / not implemented
    fragment_multiplicities   extra
    fragments                 extra
    fragment_charges          extra
    fix_symmetry              extra

    """
    # All Topology properties are found in the "molecule" key
    mol = result["molecule"]
    atnums = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
    # Geometry is in a flattened list, convert to N x 3
    atcoords = np.array([mol["geometry"][3*i:(3*i)+3] for i in range(0, len(mol["geometry"])//3)])

    result = {
        "atnums": atnums,
        "atcoords": atcoords,
    }

    # Check for optional keys (lots of try except statements due to JSON loading as a dict)
    # Load atom masses to array, canonical weights assumed if masses not given
    try:
        result["atmasses"] = np.array(mol["masses"])
    except KeyError:
        pass
    # Load molecular charge
    try:
        result["charge"] = mol["molecular_charge"]
    except KeyError:
        pass
    # Load bonds: list of tuple (atom1, atom2, bond_order)
    # Note: The QCSchema spec allows for non-integer bond_orders, these are forced to integers here
    try:
        result["bonds"] = np.array(mol["connectivity"], dtype=int)
    except KeyError:
        pass
    # Load ghost atoms
    # FIXME: do something with ghost atoms
    try:
        ghosts = mol["real"]
    except KeyError:
        pass
    # Add extra keys
    extra = dict()
    # Check for fragment keys
    # List fragment indices in nested list (likely is a jagged array)
    try:
        fragments = mol["fragments"]
        extra["fragments"] = {"indices": fragments}
    except KeyError:
        pass
    else:
        try:
            fragment_charges = mol["fragment_charges"]
            extra["fragments"]["charges"] = np.array(fragment_charges)
        except KeyError:
            pass
        try:
            fragment_multiplicities = mol["fragment_multiplicities"]
            extra["fragments"]["multiplicities"] = np.array(fragment_multiplicities)
        except KeyError:
            pass

    return result


def _parse_input_keys(result: dict):
    """Load Input (optional)
    Required keys:
    schema_name
    -------------------------------------
    schema_version
    -------------------------------------
    molecule - covered above
    -------------------------------------
    driver (type of computation requested)
    enum                      extra
    -------------------------------------
    keywords
    -------------------------------------
    model
    basis_spec (required)     OBasis
    basis_set_description
    basis_function_type (gto/sto)
    basis_harmonic_type
    basis_set_elements
    basis_set_atoms
    -------------------------------------
    """
    return dict()


def _parse_output_keys(result: dict):
    """Load Output

    """
    return dict()
