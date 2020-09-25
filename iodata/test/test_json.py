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
"""Test iodata.formats.json module."""

import numpy as np
import pytest

from ..api import load_one
from ..utils import FileFormatWarning


try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path

# Tests for qcschema_molecule
# geoms: dict of str: np.ndarray(N, 3)
geoms = {
    "LiCl": np.array([[0.000000, 0.000000, -1.631761], [0.000000, 0.000000, 0.287958]]),
    "OHr": np.array([[0.0, 0.0, -0.12947694], [0.0, -1.49418734, 1.02744651]]),
    "CuSCN": np.array(
        [
            [1.469987, -0.328195, 0.052136],
            [3.593873, -0.020962, 0.010402],
            [3.968446, -1.653292, 0.232148],
            [4.253724, -2.762010, 0.382764],
        ]
    ),
}
# These molecule examples were manually generated for testing
# mol_files: (filename, atnums, charge, spinpol, geometry)
mol_files = [
    ("LiCl_molecule.json", [3, 17], 0, 0, geoms["LiCl"]),
    # Manual validation of molpro_uks_hydroxyl_radical_gradient_output.json
    ("Hydroxyl_radical_molecule.json", [8, 1], 0, 1, geoms["OHr"]),
    # Warnings:
    #   has both masses and mass numbers
    ("CuSCN_molecule.json", [29, 16, 6, 7], 0, 0, geoms["CuSCN"]),
]


@pytest.mark.parametrize("filename, atnums, charge, spinpol, geometry", mol_files)
@pytest.mark.filterwarnings("ignore")
def test_qcschema_molecule(filename, atnums, charge, spinpol, geometry):
    """Test qcschema_molecule parsing using manually generated files."""
    with path("iodata.test.data", filename) as qcschema_molecule:
        mol = load_one(str(qcschema_molecule))

    np.testing.assert_equal(mol.atnums, atnums)
    assert mol.charge == charge
    assert mol.spinpol == spinpol
    np.testing.assert_allclose(mol.atcoords, geometry)


# Not a single valid example of qcschema_molecule is easily found for anything but water
# These molecule examples are sourced from the QCEngineRecords repo or from the QCSchema site
# molssi_mol_files: (filename, atnums, charge, spinpol, warnings)
molssi_mol_files = [
    # Extracted from qchem_logonly_rimp2_watercluster_gradient_output.json
    # Warnings:
    #   has both masses and mass numbers
    ("water_cluster.json", np.array([8, 1, 1, 8, 1, 1, 8, 1, 1]), 0, 0, 1),
    # Extracted from qchem_hf_water_energy_input.json
    # Warnings:
    #   has both masses and mass numbers
    ("water_full.json", np.array([8, 1, 1]), 0, 0, 1),
    # Copied from QCSchema RTD site
    # Warnings:
    #   no schema_name (warned in load_one and parsing molecule keys)
    #   no schema_version (warned in load_one & parsing molecule keys & unknown version)
    #   missing molecular_charge key
    #   missing molecular_multiplicity key
    ("incomplete_water.json", np.array([8, 1, 1]), 0, 0, 7),
    # Copied from QCSchema RTD site
    # Warnings:
    #   missing molecular_charge key
    #   missing molecular_multiplicity key
    ("old_water.json", np.array([8, 1, 1]), 0, 0, 2),
]


@pytest.mark.parametrize("filename, atnums, charge, spinpol, warnings", molssi_mol_files)
def test_molssi_qcschema_molecule(filename, atnums, charge, spinpol, warnings):
    """Test qcschema_molecule parsing using MolSSI-sourced files."""
    with path("iodata.test.data", filename) as qcschema_molecule:
        with pytest.warns(FileFormatWarning) as record:
            mol = load_one(str(qcschema_molecule))

    np.testing.assert_equal(mol.atnums, atnums)
    assert mol.charge == charge
    assert mol.spinpol == spinpol
    assert len(record) == warnings


# Unparsed dicts for test files
unparsed = {
    "extra": {"another_field": True},
    "nested_extra": {
        "related_projects": {"HSAB": {"id": "HSAB_2019_LALB"}, "4PB3": {"id": "4PB3_2020_Group1"}}
    },
}
# Test passthrough for molecule files using modified versions of CuSCN_molecule.json
# passthrough_mol_files: {filename, unparsed_dict}
passthrough_mol_files = [
    ("CuSCN_molecule_extra.json", unparsed["extra"]),
    ("CuSCN_molecule_nested_extra.json", unparsed["nested_extra"]),
]


@pytest.mark.parametrize("filename, unparsed_dict", passthrough_mol_files)
@pytest.mark.filterwarnings("ignore")
def test_passthrough_qcschema_molecule(filename, unparsed_dict):
    """Test qcschema_molecule parsing for passthrough of unparsed keys."""
    with path("iodata.test.data", filename) as qcschema_molecule:
        mol = load_one(str(qcschema_molecule))

    assert mol.extra["unparsed"] == unparsed_dict
