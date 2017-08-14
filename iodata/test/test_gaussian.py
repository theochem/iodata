# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import numpy as np

from nose.tools import assert_raises

from . common import get_fn
from .. gaussian import load_operators_g09, load_fchk
from .. iodata import IOData
from .. utils import shells_to_nbasis, check_dm
from .. overlap import compute_overlap

#TODO: shells_to_nbasis(obasis["shell_types"]) replacement test


def test_load_operators_water_sto3g_hf_g03():
    eps = 1e-5
    fn = get_fn('water_sto3g_hf_g03.log')
    result = load_operators_g09(fn)

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    assert overlap.shape == (7, 7)
    assert kinetic.shape == (7, 7)
    assert nuclear_attraction.shape == (7, 7)
    assert electronic_repulsion.shape == (7, 7, 7, 7)

    assert abs(overlap[0, 0] - 1.0) < eps
    assert abs(overlap[0, 1] - 0.236704) < eps
    assert abs(overlap[0, 2] - 0.0) < eps
    assert abs(overlap[-1, -3] - (-0.13198)) < eps

    assert abs(kinetic[2, 0] - 0.0) < eps
    assert abs(kinetic[4, 4] - 2.52873) < eps
    assert abs(kinetic[-1, 5] - 0.00563279) < eps
    assert abs(kinetic[-1, -3] - (-0.0966161)) < eps

    assert abs(nuclear_attraction[3, 3] - 9.99259) < eps
    assert abs(nuclear_attraction[-2, -1] - 1.55014) < eps
    assert abs(nuclear_attraction[2, 6] - 2.76941) < eps
    assert abs(nuclear_attraction[0, 3] - 0.0) < eps

    assert abs(electronic_repulsion[0, 0, 0, 0] - 4.78506575204) < eps
    assert abs(electronic_repulsion[6, 6, 6, 6] - 0.774605944194) < eps
    assert abs(electronic_repulsion[6, 5, 0, 5] - 0.0289424337101) < eps
    assert abs(electronic_repulsion[5, 4, 0, 1] - 0.00274145291476) < eps


def test_load_operators_water_ccpvdz_pure_hf_g03():
    eps = 1e-5
    fn = get_fn('water_ccpvdz_pure_hf_g03.log')
    result = load_operators_g09(fn)

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    assert overlap.shape == (24, 24)
    assert kinetic.shape == (24, 24)
    assert nuclear_attraction.shape == (24, 24)
    assert electronic_repulsion.shape == (24, 24, 24, 24)

    assert abs(overlap[0, 0] - 1.0) < eps
    assert abs(overlap[0, 1] - 0.214476) < eps
    assert abs(overlap[0, 2] - 0.183817) < eps
    assert abs(overlap[10, 16] - 0.380024) < eps
    assert abs(overlap[-1, -3] - 0.000000) < eps

    assert abs(kinetic[2, 0] - 0.160648) < eps
    assert abs(kinetic[11, 11] - 4.14750) < eps
    assert abs(kinetic[-1, 5] - (-0.0244025)) < eps
    assert abs(kinetic[-1, -6] - (-0.0614899)) < eps

    assert abs(nuclear_attraction[3, 3] - 12.8806) < eps
    assert abs(nuclear_attraction[-2, -1] - 0.0533113) < eps
    assert abs(nuclear_attraction[2, 6] - 0.173282) < eps
    assert abs(nuclear_attraction[-1, 0] - 1.24131) < eps

    assert abs(electronic_repulsion[0, 0, 0, 0] - 4.77005841522) < eps
    assert abs(electronic_repulsion[23, 23, 23, 23] - 0.785718708997) < eps
    assert abs(electronic_repulsion[23, 8, 23, 2] - (-0.0400337571969)) < eps
    assert abs(electronic_repulsion[15, 2, 12, 0] - (-0.0000308196281033)) < eps


def test_load_fchk_nonexistent():
    with assert_raises(IOError):
        load_fchk(get_fn('fubar_crap.fchk'))


def test_load_fchk_hf_sto3g_num():
    fields = load_fchk(get_fn('hf_sto3g.fchk'))
    assert fields['title'] == 'hf_sto3g'
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    assert len(obasis["shell_types"]) == 4
    assert shells_to_nbasis(obasis["shell_types"]) == 6
    assert (obasis['nprims'] == 3).all()
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -9.856961609951867E+01
    assert (fields['mulliken_charges'] == [0.45000000E+00, 4.22300000E+00]).all()
    assert (fields['npa_charges'] == [3.50000000E+00, 1.32000000E+00]).all()
    assert (fields['esp_charges'] == [0.77700000E+00, 0.66600000E+00]).all()


def test_load_fchk_h_sto3g_num():
    fields = load_fchk(get_fn('h_sto3g.fchk'))
    assert fields['title'] == 'h_sto3g'
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    assert len(obasis["shell_types"]) == 1
    assert shells_to_nbasis(obasis["shell_types"]) == 1
    assert (obasis['nprims'] == 3).all()
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 1
    assert energy == -4.665818503844346E-01


def test_load_fchk_o2_cc_pvtz_pure_num():
    fields = load_fchk(get_fn('o2_cc_pvtz_pure.fchk'))
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    assert len(obasis["shell_types"]) == 20
    assert shells_to_nbasis(obasis["shell_types"]) == 60
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -1.495944878699246E+02


def test_load_fchk_o2_cc_pvtz_cart_num():
    fields = load_fchk(get_fn('o2_cc_pvtz_cart.fchk'))
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    assert len(obasis["shell_types"]) == 20
    assert shells_to_nbasis(obasis["shell_types"]) == 70
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -1.495953594545721E+02


def test_load_fchk_water_sto3g_hf():
    fields = load_fchk(get_fn('water_sto3g_hf_g03.fchk'))
    obasis = fields['obasis']
    assert len(obasis["shell_types"]) == 5
    assert shells_to_nbasis(obasis["shell_types"]) == 7
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 3
    orb_alpha = fields['orb_alpha']
    orb_alpha_coeffs = fields['orb_alpha_coeffs']
    orb_alpha_energies = fields['orb_alpha_energies']
    orb_alpha_occs = fields['orb_alpha_occs']
    assert orb_alpha[0] == 7
    assert abs(orb_alpha_energies[0] - (-2.02333942E+01)) < 1e-7
    assert abs(orb_alpha_energies[-1] - 7.66134805E-01) < 1e-7
    assert abs(orb_alpha_coeffs[0, 0] - 0.99410) < 1e-4
    assert abs(orb_alpha_coeffs[1, 0] - 0.02678) < 1e-4
    assert abs(orb_alpha_coeffs[-1, 2] - (-0.44154)) < 1e-4
    assert abs(orb_alpha_coeffs[3, -1]) < 1e-4
    assert abs(orb_alpha_coeffs[4, -1] - (-0.82381)) < 1e-4
    assert abs(orb_alpha_occs.sum() - 5) == 0.0
    assert orb_alpha_occs.min() == 0.0
    assert orb_alpha_occs.max() == 1.0
    energy = fields['energy']
    assert energy == -7.495929232844363E+01


def test_load_fchk_lih_321g_hf():
    fields = load_fchk(get_fn('li_h_3-21G_hf_g09.fchk'))
    obasis = fields['obasis']
    assert len(obasis["shell_types"]) == 7
    assert shells_to_nbasis(obasis["shell_types"]) == 11
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    orb_alpha = fields['orb_alpha']
    orb_alpha_coeffs = fields['orb_alpha_coeffs']
    orb_alpha_energies = fields['orb_alpha_energies']
    orb_alpha_occs = fields['orb_alpha_occs']
    assert orb_alpha[0] == 11
    assert abs(orb_alpha_energies[0] - (-2.76117)) < 1e-4
    assert abs(orb_alpha_energies[-1] - 0.97089) < 1e-4
    assert abs(orb_alpha_coeffs[0, 0] - 0.99105) < 1e-4
    assert abs(orb_alpha_coeffs[1, 0] - 0.06311) < 1e-4
    assert abs(orb_alpha_coeffs[3, 2]) < 1e-4
    assert abs(orb_alpha_coeffs[-1, 9] - 0.13666) < 1e-4
    assert abs(orb_alpha_coeffs[4, -1] - 0.17828) < 1e-4
    assert abs(orb_alpha_occs.sum() - 2) == 0.0
    assert orb_alpha_occs.min() == 0.0
    assert orb_alpha_occs.max() == 1.0
    orb_beta = fields['orb_beta']
    orb_beta_coeffs = fields['orb_beta_coeffs']
    orb_beta_energies = fields['orb_beta_energies']
    orb_beta_occs = fields['orb_beta_occs']
    assert orb_beta[0] == 11
    assert abs(orb_beta_energies[0] - (-2.76031)) < 1e-4
    assert abs(orb_beta_energies[-1] - 1.13197) < 1e-4
    assert abs(orb_beta_coeffs[0, 0] - 0.99108) < 1e-4
    assert abs(orb_beta_coeffs[1, 0] - 0.06295) < 1e-4
    assert abs(orb_beta_coeffs[3, 2]) < 1e-4
    assert abs(orb_beta_coeffs[-1, 9] - 0.80875) < 1e-4
    assert abs(orb_beta_coeffs[4, -1] - (-0.15503)) < 1e-4
    assert abs(orb_beta_occs.sum() - 1) == 0.0
    assert orb_beta_occs.min() == 0.0
    assert orb_beta_occs.max() == 1.0
    energy = fields['energy']
    assert energy == -7.687331212191968E+00


def test_load_fchk_ghost_atoms():
    # Load fchk file with ghost atoms
    fields = load_fchk(get_fn('water_dimer_ghost.fchk'))
    numbers = fields['numbers']
    coordinates = fields['coordinates']
    mulliken_charges = fields['mulliken_charges']
    obasis = fields['obasis']
    # There should be 3 real atoms and 3 ghost atoms
    natom = 3
    nghost = 3
    assert numbers.shape[0] == natom
    assert coordinates.shape[0] == natom
    assert mulliken_charges.shape[0] == natom
    assert obasis['centers'].shape[0] == (natom + nghost)


def test_load_fchk_ch3_rohf_g03():
    fields = load_fchk(get_fn('ch3_rohf_sto3g_g03.fchk'))
    orb_alpha = fields['orb_alpha']
    orb_alpha_coeffs = fields['orb_alpha_coeffs']
    orb_alpha_occs = fields['orb_alpha_occs']
    orb_beta_coeffs = fields['orb_beta_coeffs']
    orb_beta_occs = fields['orb_beta_occs']
    assert orb_alpha_occs.sum() == 5
    orb_beta = fields['orb_beta']
    assert orb_beta_occs.sum() == 4
    assert (orb_alpha_coeffs == orb_beta_coeffs).all()
    assert not (orb_alpha is orb_beta)
    assert 'dm_full_scf' not in fields


def check_load_azirine(key, numbers):
    fields = load_fchk(get_fn('2h-azirine-%s.fchk') % key)
    obasis = fields['obasis']
    assert shells_to_nbasis(obasis["shell_types"]) == 33
    dm_full = fields['dm_full_%s' % key]
    assert dm_full[0, 0] == numbers[0]
    assert dm_full[32, 32] == numbers[1]


def test_load_azirine_cc():
    check_load_azirine('cc', [2.08221382E+00, 1.03516466E-01])


def test_load_azirine_ci():
    check_load_azirine('ci', [2.08058265E+00, 6.12011064E-02])


def test_load_azirine_mp2():
    check_load_azirine('mp2', [2.08253448E+00, 1.09305208E-01])


def test_load_azirine_mp3():
    check_load_azirine('mp3', [2.08243417E+00, 1.02590815E-01])


def check_load_nitrogen(key, numbers_full, numbers_spin):
    fields = load_fchk(get_fn('nitrogen-%s.fchk') % key)
    obasis = fields['obasis']
    assert shells_to_nbasis(obasis["shell_types"]) == 9
    dm_full = fields['dm_full_%s' % key]
    assert dm_full[0, 0] == numbers_full[0]
    assert dm_full[8, 8] == numbers_full[1]
    dm_spin = fields['dm_spin_%s' % key]
    assert dm_spin[0, 0] == numbers_spin[0]
    assert dm_spin[8, 8] == numbers_spin[1]


def test_load_nitrogen_cc():
    check_load_nitrogen('cc', [2.08709209E+00, 3.74723580E-01], [7.25882619E-04, -1.38368575E-02])


def test_load_nitrogen_ci():
    check_load_nitrogen('ci', [2.08741410E+00, 2.09292886E-01], [7.41998558E-04, -6.67582215E-03])


def test_load_nitrogen_mp2():
    check_load_nitrogen('mp2', [2.08710027E+00, 4.86472609E-01], [7.31802950E-04, -2.00028488E-02])


def test_load_nitrogen_mp3():
    check_load_nitrogen('mp3', [2.08674302E+00, 4.91149023E-01], [7.06941101E-04, -1.96276763E-02])


def check_normalization_dm_full_azirine(key):
    #TODO: replace with cached data
    mol = IOData.from_file(get_fn('2h-azirine-%s.fchk') % key)
    olp = compute_overlap(**mol.obasis)
    dm = getattr(mol, 'dm_full_%s' % key)
    check_dm(dm, olp, eps=1e-2, occ_max=2)
    assert abs(np.einsum('ab,ba', olp, dm) - 22.0) < 1e-3


def test_normalization_dm_full_azirine_cc():
    check_normalization_dm_full_azirine('cc')


def test_normalization_dm_full_azirine_ci():
    check_normalization_dm_full_azirine('ci')


def test_normalization_dm_full_azirine_mp2():
    check_normalization_dm_full_azirine('mp2')


def test_normalization_dm_full_azirine_mp3():
    check_normalization_dm_full_azirine('mp3')


def test_load_water_hfs_321g():
    mol = IOData.from_file(get_fn('water_hfs_321g.fchk'))
    assert mol.polar[0, 0] == 7.23806684E+00
    assert mol.polar[1, 1] == 8.04213953E+00
    assert mol.polar[1, 2] == 1.20021770E-10
    np.testing.assert_allclose(mol.dipole_moment, [
        -5.82654324E-17, 0.00000000E+00, -8.60777067E-01])
    np.testing.assert_allclose(mol.quadrupole_moment, [
        -8.89536026E-01,  # xx
        8.28408371E-17,  # xy
        4.89353090E-17,  # xz
        1.14114241E+00,  # yy
        -5.47382213E-48,  # yz
        -2.51606382E-01])  # zz


def test_load_monosilicic_acid_hf_lan():
    mol = IOData.from_file(get_fn('monosilicic_acid_hf_lan.fchk'))
    np.testing.assert_allclose(mol.dipole_moment, [
        -6.05823053E-01, -9.39656399E-03, 4.18948869E-01])
    np.testing.assert_allclose(mol.quadrupole_moment, [
        2.73609152E+00,  # xx
        -6.65787832E-02,  # xy
        2.11973730E-01,  # xz
        8.97029351E-01,  # yy
        -1.38159653E-02,  # yz
        -3.63312087E+00])  # zz
