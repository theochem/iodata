# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --
# pragma pylint: disable=invalid-name,no-member
"""Test iodata.wfn module."""

import pytest

# import warnings

import numpy as np

from ..wfx import load_wfx_low

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_wfx_low_h2():
    """Test load_wfx_low with h2_ub3lyp_ccpvtz.wfx."""
    with path('iodata.test.data', 'h2_ub3lyp_ccpvtz.wfx') as fn_wfx:
        data = load_wfx_low(str(fn_wfx))
    # unpack data
    title, keywords, model_name, atom_names, num_atoms, num_primitives, \
        num_occ_mo, num_perturbations, num_electrons, num_alpha_electron, \
        num_beta_electron, num_spin_multi, charge, energy, \
        virial_ratio, nuclear_virial, full_virial_ratio, mo_count, \
        atom_numbers, mo_spin_type, coordinates, centers, \
        primitives_types, exponent, mo_occ, mo_energy, gradient_atoms, \
        gradient, mo_coefficients = data

    assert title == 'h2 ub3lyp/cc-pvtz opt-stable-freq'
    assert keywords == 'GTO'
    assert model_name is None
    assert num_atoms == [2]
    assert num_primitives == [34]
    assert num_occ_mo == [56]
    assert charge == [0]
    assert num_perturbations == [0]
    assert num_electrons == [2]
    assert num_alpha_electron == [1]
    assert num_beta_electron == [1]
    assert num_spin_multi == [1]
    assert energy == [-1.179998789924e+00]
    assert virial_ratio == [2.036441983763e+00]
    assert nuclear_virial == [1.008787649881e-08]
    assert full_virial_ratio == [2.036441992623e+00]
    assert (atom_names == ['H1', 'H2']).all()
    assert (atom_numbers == [1, 1]).all()
    assert (mo_spin_type == np.array(
        [['Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha',
          'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha',
          'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha',
          'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha', 'Alpha',
          'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta',
          'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta',
          'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta',
          'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta', 'Beta', ]]).T).all()
    assert (coordinates[0] == [0., 0., 0.7019452462164]).all()
    assert (coordinates[1] == [0., 0., -0.7019452462164]).all()
    assert (centers == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2]).all()
    assert (primitives_types == [1, 1, 1, 1, 1, 2,
                                 3, 4, 2, 3, 4, 5,
                                 6, 7, 8, 9, 10, 1,
                                 1, 1, 1, 1, 2, 3,
                                 4, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10]).all()
    assert (exponent == [3.387000000000e+01, 5.095000000000e+00,
                         1.159000000000e+00, 3.258000000000e-01,
                         1.027000000000e-01, 1.407000000000e+00,
                         1.407000000000e+00, 1.407000000000e+00,
                         3.880000000000e-01, 3.880000000000e-01,
                         3.880000000000e-01, 1.057000000000e+00,
                         1.057000000000e+00, 1.057000000000e+00,
                         1.057000000000e+00, 1.057000000000e+00,
                         1.057000000000e+00, 3.387000000000e+01,
                         5.095000000000e+00, 1.159000000000e+00,
                         3.258000000000e-01, 1.027000000000e-01,
                         1.407000000000e+00, 1.407000000000e+00,
                         1.407000000000e+00, 3.880000000000e-01,
                         3.880000000000e-01, 3.880000000000e-01,
                         1.057000000000e+00, 1.057000000000e+00,
                         1.057000000000e+00, 1.057000000000e+00,
                         1.057000000000e+00, 1.057000000000e+00]).all()
    assert (mo_occ == [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0.]).all()
    assert (mo_energy == [-4.340830854172e-01, 5.810590098068e-02,
                          1.957476339319e-01, 4.705943952631e-01,
                          5.116003517961e-01, 5.116003517961e-01,
                          9.109680450208e-01, 9.372078887497e-01,
                          9.372078887497e-01, 1.367198523024e+00,
                          2.035656924620e+00, 2.093459617091e+00,
                          2.882582109554e+00, 2.882582109559e+00,
                          3.079758295551e+00, 3.079758295551e+00,
                          3.356387932344e+00, 3.600856684661e+00,
                          3.600856684661e+00, 3.793185027287e+00,
                          3.793185027400e+00, 3.807665977092e+00,
                          3.807665977092e+00, 4.345665616275e+00,
                          5.386560784523e+00, 5.386560784523e+00,
                          5.448122593462e+00, 6.522366660004e+00,
                          -4.340830854172e-01, 5.810590098068e-02,
                          1.957476339319e-01, 4.705943952631e-01,
                          5.116003517961e-01, 5.116003517961e-01,
                          9.109680450208e-01, 9.372078887497e-01,
                          9.372078887497e-01, 1.367198523024e+00,
                          2.035656924620e+00, 2.093459617091e+00,
                          2.882582109554e+00, 2.882582109559e+00,
                          3.079758295551e+00, 3.079758295551e+00,
                          3.356387932344e+00, 3.600856684661e+00,
                          3.600856684661e+00, 3.793185027287e+00,
                          3.793185027400e+00, 3.807665977092e+00,
                          3.807665977092e+00, 4.345665616275e+00,
                          5.386560784523e+00, 5.386560784523e+00,
                          5.448122593462e+00, 6.522366660004e+00]).all()
    assert (gradient_atoms == ['H1', 'H2']).all()
    assert (gradient[0, :] == [9.744384163503e-17,
                               -2.088844408785e-16,
                               -7.185657679987e-09]).all()
    assert (gradient[1, :] == [-9.744384163503e-17,
                               2.088844408785e-16,
                               7.185657679987e-09]).all()
    assert (mo_count == np.arange(1, 57)).all()
    assert mo_coefficients.shape == (34, 56)
    exponent_expected = np.array([
        5.054717669172e-02, 9.116391481072e-02,
        1.344211391235e-01, 8.321037376208e-02,
        1.854203733451e-02, -5.552096650015e-17,
        1.685043781907e-17, -2.493514848195e-02,
        5.367769875676e-18, -8.640401342563e-21,
        -4.805966923740e-03, -3.124765025063e-04,
        -3.124765025063e-04, 6.249530050126e-04,
        6.560467295881e-16, -8.389003686496e-17,
        1.457172009403e-16, 5.054717669172e-02,
        9.116391481072e-02, 1.344211391235e-01,
        8.321037376215e-02, 1.854203733451e-02,
        1.377812848830e-16, -5.365229184139e-18,
        2.493514848197e-02, -2.522774106094e-17,
        2.213188439119e-17, 4.805966923784e-03,
        -3.124765025186e-04, -3.124765025186e-04,
        6.249530050373e-04, -6.548275062740e-16,
        4.865003740982e-17, -1.099855647247e-16], dtype=np.float)
    assert (mo_coefficients[:, 0] == exponent_expected).all()
    assert mo_coefficients[2, 9] == 1.779549601504e-02
    assert mo_coefficients[19, 14] == -1.027984391469e-15
    assert mo_coefficients[26, 36] == -5.700424557682e-01


def test_load_wfx_low_water():
    """Test load_wfx_low with water_sto3g_hf.wfx."""
    with path('iodata.test.data', 'water_sto3g_hf.wfx') as fn_wfx:
        data = load_wfx_low(str(fn_wfx))
    # unpack data
    title, keywords, model_name, atom_names, num_atoms, num_primitives, \
        num_occ_mo, num_perturbations, num_electrons, num_alpha_electron, \
        num_beta_electron, num_spin_multi, charge, energy, \
        virial_ratio, nuclear_virial, full_virial_ratio, mo_count, \
        atom_numbers, mo_spin_type, coordinates, centers, \
        primitives_types, exponent, mo_occ, mo_energy, gradient_atoms, \
        gradient, mo_coefficients = data

    assert title == 'H2O HF/STO-3G//HF/STO-3G'
    assert keywords == 'GTO'
    assert model_name == 'Restricted HF'
    assert num_atoms == [3]
    assert num_primitives == [21]
    assert num_occ_mo == [5]
    assert charge == [0.00000000000000E+000]
    assert num_perturbations == [0]
    assert num_electrons == [10]
    assert num_alpha_electron == [5]
    assert num_beta_electron == [5]
    assert num_spin_multi == np.array(None)
    assert energy == [-7.49659011707870E+001]
    assert virial_ratio == [2.00599838291596E+000]
    assert nuclear_virial == np.array(None)
    assert full_virial_ratio == [2.00600662884992E+000]
    assert (atom_names == ['O1', 'H2', 'H3']).all()
    assert (atom_numbers == [8, 1, 1]).all()
    assert (mo_spin_type == np.array([['Alpha',
                                       'Beta',
                                       'Alpha',
                                       'Beta',
                                       'Alpha',
                                       'Beta',
                                       'Alpha',
                                       'Beta',
                                       'Alpha',
                                       'Beta']]).T).all()
    assert (coordinates[0] == [0.00000000000000E+000,
                               0.00000000000000E+000,
                               2.40242907000000E-001]).all()
    assert (coordinates[1] == [0.00000000000000E+000,
                               1.43244242000000E+000,
                               -9.60971627000000E-001]).all()
    assert (coordinates[2] == [-1.75417809000000E-016,
                               -1.43244242000000E+000,
                               -9.60971627000000E-001]).all()
    assert (centers == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 2, 2, 2, 3, 3, 3]).all()
    assert (primitives_types == [1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3,
                                 4, 2, 3, 4, 1, 1, 1, 1, 1, 1]).all()

    assert (exponent == [1.30709321000000E+002, 2.38088661000000E+001,
                         6.44360831000000E+000, 5.03315132000000E+000,
                         1.16959612000000E+000, 3.80388960000000E-001,
                         5.03315132000000E+000, 5.03315132000000E+000,
                         5.03315132000000E+000, 1.16959612000000E+000,
                         1.16959612000000E+000, 1.16959612000000E+000,
                         3.80388960000000E-001, 3.80388960000000E-001,
                         3.80388960000000E-001, 3.42525091000000E+000,
                         6.23913730000000E-001, 1.68855404000000E-001,
                         3.42525091000000E+000, 6.23913730000000E-001,
                         1.68855404000000E-001]).all()

    assert (mo_occ == [2.00000000000000E+000, 2.00000000000000E+000,
                       2.00000000000000E+000, 2.00000000000000E+000,
                       2.00000000000000E+000]).all()
    assert (mo_energy == [-2.02515479000000E+001, -1.25760928000000E+000,
                          -5.93941119000000E-001, -4.59728723000000E-001,
                          -3.92618460000000E-001]).all()
    assert (gradient_atoms == ['O1', 'H2', 'H3']).all()
    assert (gradient[0, :] == [6.09070231000000E-016,
                               -5.55187875000000E-016,
                               -2.29270172000000E-004]).all()
    assert (gradient[1, :] == [-2.46849911000000E-016,
                               -1.18355659000000E-004,
                               1.14635086000000E-004]).all()
    assert (mo_count == np.arange(1, 6)).all()
    assert mo_coefficients.shape == (21, 5)
    exponent_expected = [4.22735025664585E+000,
                         4.08850914632625E+000,
                         1.27420971692421E+000,
                         -6.18883321546465E-003,
                         8.27806436882009E-003,
                         6.24757868903820E-003,
                         0.00000000000000E+000,
                         0.00000000000000E+000,
                         -6.97905144921135E-003,
                         0.00000000000000E+000,
                         0.00000000000000E+000,
                         -4.38861481239680E-003,
                         0.00000000000000E+000,
                         0.00000000000000E+000,
                         -6.95230322147800E-004,
                         -1.54680714141406E-003,
                         -1.49600452906993E-003,
                         -4.66239267760156E-004,
                         -1.54680714141406E-003,
                         -1.49600452906993E-003,
                         -4.66239267760156E-004]
    assert (mo_coefficients[:, 0] == exponent_expected).all()
    assert mo_coefficients[1, 3] == -4.27845789719456E-001


def test_load_wfx_low_missing_tag_h2o():
    """Test load_wfx_low with h2o_error.wfx with missing tag."""
    with pytest.raises(IOError):
        load_wfx_low(filename='iodata/test/data/h2o_error.wfx')


# def test_load_wfx_low_missing_tag_h2o():
#     """test load_wfx_low with missing title in h2o_error2.wfx."""
#     with warnings.catch_warnings(record=True) as w:
#         # Ensure all warnings to always to be triggered
#         warnings.simplefilter("always")
#         # Trigger the warning
#         load_wfx_low(filename='iodata/test/data/h2o_error2.wfx')
#         assert 'are/is required but not present in the WFX file.' \
#                in str(w[-1].message)
