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

import os

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from .common import compute_mulliken_charges, compare_mols, check_orthonormal
from ..api import load_one, dump_one
from ..basis import convert_conventions
from ..formats.molden import _load_low
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS
from ..utils import LineIterator, angstrom, FileFormatWarning


try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water_hf_gradient():
    """Test load_one for water HF gradient example.

    Source: "https://molssi-qc-schema.readthedocs.io/en/latest/examples.html"

    """
    with path("iodata.test.data", "water_hf_gradient.json") as json_in:
        mol = load_one(str(json_in))

    assert_equal(mol.atnums, [8, 1, 1])
    assert_allclose(
        mol.atcoords, np.array([[0, 0, -0.1294], [0, -1.4941, 1.0274], [0, 1.4941, 1.0274]])
    )


def test_load_water_mp2_energy():
    """Test load_one for water MP2 example.

    Source: https://molssi-qc-schema.readthedocs.io/en/latest/examples.html

    """
    pass


def test_big_fileset_for_crashes():
    """Try loading lots of JSON to look for errors.

    Notes
    -----
    _file_data.txt Contains the keys for each file, to narrow them down soon.
    qchem_logonly_dft_watercluster_gradient_input.json is some sort of dummy file.
    qchem_logonly_dft_watercluster_gradient_output.json is the first file where I noticed that
        QCEngine will put in null values for optional fields.
    dftd3_water_energy_output.json is a DFTD3 calculation result, which isn't supported
    dftd3_water_gradient_output.json is a DFTD3 calculation result, which isn't supported

    """
    import os

    invalid = [
        "_file_data.txt",
        "qchem_logonly_dft_watercluster_gradient_input.json",
        "qchem_logonly_rimp2_watercluster_gradient_input.json",
        "dftd3_water_energy_output.json",
        "dftd3_water_gradient_output.json",
    ]
    for f in os.listdir("C:/Users/wilha/iodata/iodata/test/data/json_tests_qcenginerecords"):
        if f not in invalid:
            print(f)
            with path("iodata.test.data.json_tests_qcenginerecords", f) as json_in:
                mol = load_one(
                    os.path.join(
                        "C:/Users/wilha/iodata/iodata/test/data/json_tests_qcenginerecords", f
                    )
                )


def test_qcschema_dev_for_crashes():
    """Try loading lots of JSON to look for errors.

    Notes
    -----
    basis_waterZr_energy_B3LYP_STO3G_DEF2_input.json is missing schema_name in basis

    """
    import os

    invalid = []
    for f in os.listdir("C:/Users/wilha/iodata/iodata/test/data/json_tests_qcschemadev"):
        if f not in invalid:
            print(f)
            with path("iodata.test.data.json_tests_qcschemadev", f) as json_in:
                mol = load_one(
                    os.path.join(
                        "C:/Users/wilha/iodata/iodata/test/data/json_tests_qcschemadev", f
                    )
                )


def test_basis():
    """Test basis_spec using the only available test.

    https://github.com/MolSSI/QCElemental/blob/master/qcelemental/tests/test_model_results.py

    random_data is from basissetexchange.com

    """
    center_data = {
        "bs_sto3g_h": {
            "electron_shells": [
                {
                    "harmonic_type": "spherical",
                    "angular_momentum": [0],
                    "exponents": [3.42525091, 0.62391373, 0.16885540],
                    "coefficients": [[0.15432897, 0.53532814, 0.44463454]],
                }
            ]
        },
        "bs_sto3g_o": {
            "electron_shells": [
                {
                    "harmonic_type": "spherical",
                    "angular_momentum": [0],
                    "exponents": [130.70939, 23.808861, 6.4436089],
                    "coefficients": [[0.15432899, 0.53532814, 0.44463454]],
                },
                {
                    "harmonic_type": "cartesian",
                    "angular_momentum": [0, 1],
                    "exponents": [5.0331513, 1.1695961, 0.3803890],
                    "coefficients": [
                        [-0.09996723, 0.39951283, 0.70011547],
                        [0.15591629, 0.60768379, 0.39195739],
                    ],
                },
                {
                    "harmonic_type": "cartesian",
                    "angular_momentum": [0],
                    "exponents": [5.0331513, 1.1695961, 0.3803890],
                    "coefficients": [
                        [-5.09996723, 0.39951283, 0.70011547],
                        [0.15591629, 0.60768379, 0.39195739],
                    ],
                },
            ]
        },
        "bs_def2tzvp_zr": {
            "electron_shells": [
                {
                    "harmonic_type": "spherical",
                    "angular_momentum": [0],
                    "exponents": [11.000000000, 9.5000000000, 3.6383667759, 0.76822026698],
                    "coefficients": [
                        [-0.19075595259, 0.33895588754, 0.0000000, 0.0000000],
                        [0.0000000, 0.0000000, 1.0000000000, 0.0000000],
                    ],
                },
                {
                    "harmonic_type": "spherical",
                    "angular_momentum": [2],
                    "exponents": [4.5567957795, 1.2904939799, 0.51646987229],
                    "coefficients": [
                        [-0.96190569023e-09, 0.20569990155, 0.41831381851],
                        [0.0000000, 0.0000000, 0.0000000],
                        [0.0000000, 0.0000000, 0.0000000],
                    ],
                },
                {
                    "harmonic_type": "spherical",
                    "angular_momentum": [3],
                    "exponents": [0.3926100],
                    "coefficients": [[1.0000000]],
                },
            ],
            "ecp_electrons": 28,
            "ecp_potentials": [
                {
                    "ecp_type": "scalar",
                    "angular_momentum": [0],
                    "r_exponents": [2, 2, 2, 2],
                    "gaussian_exponents": [7.4880494, 3.7440249, 6.5842120, 3.2921060],
                    "coefficients": [[135.15384419, 15.55244130, 19.12219811, 2.43637549]],
                },
                {
                    "ecp_type": "spinorbit",
                    "angular_momentum": [1],
                    "r_exponents": [2, 2, 2, 2],
                    "gaussian_exponents": [6.4453779, 3.2226886, 6.5842120, 3.2921060],
                    "coefficients": [[87.78499169, 11.56406599, 19.12219811, 2.43637549]],
                },
            ],
        },
    }
    random_data = {
        "schema_name": "qcschema_basis",
        "schema_version": 1,
        "name": "Ahlrichs pVDZ",
        "description": "VDZP    Valence Double Zeta + Polarization on All Atoms",
        "center_data": {
            "h_Ahlrichs pVDZ": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["13.010701", "1.9622572", "0.44453796"],
                        "coefficients": [["0.19682158E-01", "0.13796524", "0.47831935"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.12194962"],
                        "coefficients": [["1.0000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.800000"],
                        "coefficients": [["1.000000"]],
                        "harmonic_type": "cartesian",
                    },
                ]
            },
            "ne_Ahlrichs pVDZ": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": [
                            "3598.9736625",
                            "541.32073112",
                            "122.90450062",
                            "34.216617022",
                            "10.650584124",
                        ],
                        "coefficients": [
                            [
                                "-0.53259297003E-02",
                                "-0.39817417969E-01",
                                "-0.17914358188",
                                "-0.46893582977",
                                "-0.44782537577",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["1.3545953960"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.41919362639"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["28.424053785", "6.2822510953", "1.6978715079"],
                        "coefficients": [
                            ["-0.46031944795E-01", "-0.23993183041", "-0.50871724964"]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.43300700172"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": ["1.888000"],
                        "coefficients": [["1.000000"]],
                        "harmonic_type": "spherical",
                    },
                ]
            },
            "na_Ahlrichs pVDZ": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": [
                            "4098.2003908",
                            "616.49374031",
                            "139.96644001",
                            "39.073441051",
                            "11.929847205",
                        ],
                        "coefficients": [
                            [
                                "-0.58535911879E-02",
                                "-0.43647161872E-01",
                                "-0.19431465884",
                                "-0.48685065731",
                                "-0.41881705137",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["20.659966030", "1.9838860978", "0.64836323942"],
                        "coefficients": [["0.85949689854E-01", "-0.56359144041", "-0.51954009048"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.52443967404E-01"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.28048160742E-01"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": [
                            "75.401862017",
                            "17.274818978",
                            "5.1842347425",
                            "1.6601211973",
                            "0.51232528958",
                        ],
                        "coefficients": [
                            [
                                "0.154353625324E-01",
                                "0.997382931840E-01",
                                "0.312095939659",
                                "0.492956748074",
                                "0.324203983180",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.052000"],
                        "coefficients": [["1.000000"]],
                        "harmonic_type": "cartesian",
                    },
                ]
            },
            "cr_Ahlrichs pVDZ": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": [
                            "51528.086349",
                            "7737.2103487",
                            "1760.3748470",
                            "496.87706544",
                            "161.46520598",
                            "55.466352268",
                        ],
                        "coefficients": [
                            [
                                "0.14405823106E-02",
                                "0.11036202287E-01",
                                "0.54676651806E-01",
                                "0.18965038103",
                                "0.38295412850",
                                "0.29090050668",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["107.54732999", "12.408671897", "5.0423628826"],
                        "coefficients": [["-0.10932281100", "0.64472599471", "0.46262712560"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["8.5461640165", "1.3900441221", "0.56066602876"],
                        "coefficients": [["-0.22711013286", "0.73301527591", "0.44225565433"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.71483705972E-01"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.28250687604E-01"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": [
                            "640.48536096",
                            "150.69711194",
                            "47.503755296",
                            "16.934120165",
                            "6.2409680590",
                        ],
                        "coefficients": [
                            [
                                "0.96126715203E-02",
                                "0.70889834655E-01",
                                "0.27065258990",
                                "0.52437343414",
                                "0.34107994714",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["3.0885463206", "1.1791047769", "0.43369774432"],
                        "coefficients": [["0.33973986903", "0.57272062927", "0.24582728206"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.120675"],
                        "coefficients": [["1.000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": [
                            "27.559479426",
                            "7.4687020327",
                            "2.4345903574",
                            "0.78244754808",
                        ],
                        "coefficients": [
                            ["0.30612488044E-01", "0.15593270944", "0.36984421276", "0.47071118077"]
                        ],
                        "harmonic_type": "spherical",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": ["0.21995774311"],
                        "coefficients": [["0.33941649889"]],
                        "harmonic_type": "spherical",
                    },
                ]
            },
            "as_Ahlrichs pVDZ": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": [
                            "100146.52554",
                            "15036.861711",
                            "3421.2902833",
                            "966.16965717",
                            "314.87394026",
                            "108.70823790",
                        ],
                        "coefficients": [
                            [
                                "0.14258349617E-02",
                                "0.10930176963E-01",
                                "0.54294174610E-01",
                                "0.18976078153",
                                "0.38775195453",
                                "0.30402812040",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["209.54238950", "25.038221139", "10.390964343"],
                        "coefficients": [["-0.11162094204", "0.64697607762", "0.44223608673"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["18.555090093", "3.1281217449", "1.3884885073"],
                        "coefficients": [["-0.22994190569", "0.73319107613", "0.45533653943"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.24714362141"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [0],
                        "exponents": ["0.91429428670E-01"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": [
                            "1355.6443507",
                            "319.99929270",
                            "101.67734092",
                            "36.886323845",
                            "13.861115909",
                        ],
                        "coefficients": [
                            [
                                "0.89182507898E-02",
                                "0.67454750717E-01",
                                "0.26759772110",
                                "0.53776844520",
                                "0.35992570244",
                            ]
                        ],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["7.4260666912", "3.0316247187", "1.2783078340"],
                        "coefficients": [["0.34036849637", "0.57030149334", "0.26606170238"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.37568503356"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["0.11394805454"],
                        "coefficients": [["1.0000000000"]],
                        "harmonic_type": "cartesian",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": [
                            "84.445514539",
                            "24.190416102",
                            "8.4045015119",
                            "2.9808970748",
                        ],
                        "coefficients": [
                            ["0.24518402724E-01", "0.14107454677", "0.36875228915", "0.48409561362"]
                        ],
                        "harmonic_type": "spherical",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": ["0.97909243359"],
                        "coefficients": [["0.28250268781"]],
                        "harmonic_type": "spherical",
                    },
                    {
                        "angular_momentum": [2],
                        "exponents": ["0.2930"],
                        "coefficients": [["1.000000"]],
                        "harmonic_type": "spherical",
                    },
                ]
            },
        },
    }
