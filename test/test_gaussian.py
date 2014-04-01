# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
#pylint: skip-file


from nose.tools import assert_raises

from horton import *


def test_load_operators_water_sto3g_hf_g03():
    lf = DenseLinalgFactory()
    eps = 1e-5
    fn = context.get_fn('test/water_sto3g_hf_g03.log')
    result = load_operators_g09(fn, lf)

    for op in result.itervalues():
        assert op is not None
        assert op.nbasis == 7
        op.check_symmetry()

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    assert abs(overlap.get_element(0,0) - 1.0) < eps
    assert abs(overlap.get_element(0,1) - 0.236704) < eps
    assert abs(overlap.get_element(0,2) - 0.0) < eps
    assert abs(overlap.get_element(-1,-3) - (-0.13198)) < eps

    assert abs(kinetic.get_element(2,0) - 0.0) < eps
    assert abs(kinetic.get_element(4,4) - 2.52873) < eps
    assert abs(kinetic.get_element(-1,5) - 0.00563279) < eps
    assert abs(kinetic.get_element(-1,-3) - (-0.0966161)) < eps

    assert abs(nuclear_attraction.get_element(3,3) - 9.99259) < eps
    assert abs(nuclear_attraction.get_element(-2,-1) - 1.55014) < eps
    assert abs(nuclear_attraction.get_element(2,6) - 2.76941) < eps
    assert abs(nuclear_attraction.get_element(0,3) - 0.0) < eps

    assert abs(electronic_repulsion.get_element(0,0,0,0) - 4.78506575204) < eps
    assert abs(electronic_repulsion.get_element(6,6,6,6) - 0.774605944194) < eps
    assert abs(electronic_repulsion.get_element(6,5,0,5) - 0.0289424337101) < eps
    assert abs(electronic_repulsion.get_element(5,4,0,1) - 0.00274145291476) < eps


def test_load_operators_water_ccpvdz_pure_hf_g03():
    lf = DenseLinalgFactory()
    eps = 1e-5
    fn = context.get_fn('test/water_ccpvdz_pure_hf_g03.log')
    result = load_operators_g09(fn, lf)

    overlap = result['olp']
    kinetic = result['kin']
    nuclear_attraction = result['na']
    electronic_repulsion = result['er']

    for op in overlap, kinetic, nuclear_attraction, electronic_repulsion:
        assert op is not None
        assert op.nbasis == 24
        op.check_symmetry()

    assert abs(overlap.get_element(0,0) - 1.0) < eps
    assert abs(overlap.get_element(0,1) - 0.214476) < eps
    assert abs(overlap.get_element(0,2) - 0.183817) < eps
    assert abs(overlap.get_element(10,16) - 0.380024) < eps
    assert abs(overlap.get_element(-1,-3) - 0.000000) < eps

    assert abs(kinetic.get_element(2,0) - 0.160648) < eps
    assert abs(kinetic.get_element(11,11) - 4.14750) < eps
    assert abs(kinetic.get_element(-1,5) - (-0.0244025)) < eps
    assert abs(kinetic.get_element(-1,-6) - (-0.0614899)) < eps

    assert abs(nuclear_attraction.get_element(3,3) - 12.8806) < eps
    assert abs(nuclear_attraction.get_element(-2,-1) - 0.0533113) < eps
    assert abs(nuclear_attraction.get_element(2,6) - 0.173282) < eps
    assert abs(nuclear_attraction.get_element(-1,0) - 1.24131) < eps

    assert abs(electronic_repulsion.get_element(0,0,0,0) - 4.77005841522) < eps
    assert abs(electronic_repulsion.get_element(23,23,23,23) - 0.785718708997) < eps
    assert abs(electronic_repulsion.get_element(23,8,23,2) - (-0.0400337571969)) < eps
    assert abs(electronic_repulsion.get_element(15,2,12,0) - (-0.0000308196281033)) < eps


def test_load_fchk_nonexistent():
    lf = DenseLinalgFactory()
    with assert_raises(IOError):
        load_fchk(context.get_fn('test/fubar_crap.fchk'), lf)


def test_load_fchk_hf_sto3g_num():
    lf = DenseLinalgFactory()
    fields = load_fchk(context.get_fn('test/hf_sto3g.fchk'), lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    wfn = fields['wfn']
    assert obasis.nshell == 4
    assert obasis.nbasis == 6
    assert (obasis.nprims == 3).all()
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -9.856961609951867E+01
    assert (fields['mulliken_charges'] == [0.45000000E+00 , 4.22300000E+00]).all()
    assert (fields['npa_charges']== [3.50000000E+00,  1.32000000E+00]).all()
    assert (fields['esp_charges']==[ 0.77700000E+00,  0.66600000E+00]).all()
    wfn.dm_full.check_symmetry()


def test_load_fchk_h_sto3g_num():
    lf = DenseLinalgFactory()
    fields = load_fchk(context.get_fn('test/h_sto3g.fchk'), lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    wfn = fields['wfn']
    assert obasis.nshell == 1
    assert obasis.nbasis == 1
    assert (obasis.nprims == 3).all()
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 1
    assert energy == -4.665818503844346E-01
    wfn.dm_full.check_symmetry()
    wfn.dm_spin.check_symmetry()


def test_load_fchk_o2_cc_pvtz_pure_num():
    lf = DenseLinalgFactory()
    fields = load_fchk(context.get_fn('test/o2_cc_pvtz_pure.fchk'), lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    wfn = fields['wfn']
    assert obasis.nshell == 20
    assert obasis.nbasis == 60
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -1.495944878699246E+02
    wfn.dm_full.check_symmetry()


def test_load_fchk_o2_cc_pvtz_cart_num():
    lf = DenseLinalgFactory()
    fields = load_fchk(context.get_fn('test/o2_cc_pvtz_cart.fchk'), lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    wfn = fields['wfn']
    energy = fields['energy']
    assert obasis.nshell == 20
    assert obasis.nbasis == 70
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert energy == -1.495953594545721E+02
    wfn.dm_full.check_symmetry()


def test_load_fchk_water_sto3g_hf():
    lf = DenseLinalgFactory()
    fields = load_fchk(context.get_fn('test/water_sto3g_hf_g03.fchk'), lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    wfn = fields['wfn']
    assert obasis.nshell == 5
    assert obasis.nbasis == 7
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 3
    assert wfn.nbasis == 7
    assert wfn.occ_model.nalpha == 5
    assert wfn.occ_model.nbeta == 5
    assert abs(wfn.exp_alpha.energies[0] - (-2.02333942E+01)) < 1e-7
    assert abs(wfn.exp_alpha.energies[-1] - 7.66134805E-01) < 1e-7
    assert abs(wfn.exp_alpha.coeffs[0,0] - 0.99410) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[1,0] - 0.02678) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[-1,2] - (-0.44154)) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[3,-1]) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[4,-1] - (-0.82381)) < 1e-4
    assert energy == -7.495929232844363E+01
    wfn.dm_full.check_symmetry()


def test_load_fchk_lih_321g_hf():
    lf = DenseLinalgFactory()
    fn_fchk = context.get_fn('test/li_h_3-21G_hf_g09.fchk')
    fields = load_fchk(fn_fchk, lf)
    obasis = fields['obasis']
    coordinates = fields['coordinates']
    numbers = fields['numbers']
    energy = fields['energy']
    wfn = fields['wfn']
    assert obasis.nshell == 7
    assert obasis.nbasis == 11
    assert len(coordinates) == len(numbers)
    assert coordinates.shape[1] == 3
    assert len(numbers) == 2
    assert wfn.nbasis == 11
    assert wfn.occ_model.nalpha == 2
    assert wfn.occ_model.nbeta == 1
    assert abs(wfn.exp_alpha.energies[0] - (-2.76117)) < 1e-4
    assert abs(wfn.exp_alpha.energies[-1] - 0.97089) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[0,0] - 0.99105) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[1,0] - 0.06311) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[3,2]) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[-1,9] - 0.13666) < 1e-4
    assert abs(wfn.exp_alpha.coeffs[4,-1] - 0.17828) < 1e-4
    assert abs(wfn.exp_beta.energies[0] - (-2.76031)) < 1e-4
    assert abs(wfn.exp_beta.energies[-1] - 1.13197) < 1e-4
    assert abs(wfn.exp_beta.coeffs[0,0] - 0.99108) < 1e-4
    assert abs(wfn.exp_beta.coeffs[1,0] - 0.06295) < 1e-4
    assert abs(wfn.exp_beta.coeffs[3,2]) < 1e-4
    assert abs(wfn.exp_beta.coeffs[-1,9] - 0.80875) < 1e-4
    assert abs(wfn.exp_beta.coeffs[4,-1] - (-0.15503)) < 1e-4
    wfn.dm_full.check_symmetry()
    wfn.dm_spin.check_symmetry()
    assert energy == -7.687331212191968E+00


def test_load_fchk_ghost_atoms():
    # Load fchk file with ghost atoms
    lf = DenseLinalgFactory()
    fn_fchk = context.get_fn('test/water_dimer_ghost.fchk')
    fields = load_fchk(fn_fchk, lf)
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
    assert obasis.centers.shape[0] == ( natom + nghost )


def check_load_azirine(key, numbers):
    lf = DenseLinalgFactory()
    fn_fchk = context.get_fn('test/2h-azirine-%s.fchk' % key)
    fields = load_fchk(fn_fchk, lf)
    wfn = fields['wfn']
    assert 'dm_full' in wfn._cache
    assert wfn.nbasis == 33
    assert wfn.dm_full._array[0, 0] == numbers[0]
    assert wfn.dm_full._array[32, 32] == numbers[1]


def test_load_azirine_ccd():
    check_load_azirine('ccd', [2.08221382E+00, 1.03516466E-01])


def test_load_azirine_cis():
    check_load_azirine('cis', [2.08058265E+00, 6.12011064E-02])


def test_load_azirine_mp2():
    check_load_azirine('mp2', [2.08253448E+00, 1.09305208E-01])


def test_load_azirine_mp3():
    check_load_azirine('mp3', [2.08243417E+00, 1.02590815E-01])


def check_load_nitrogen(key, numbers_full, numbers_spin):
    lf = DenseLinalgFactory()
    fn_fchk = context.get_fn('test/nitrogen-%s.fchk' % key)
    fields = load_fchk(fn_fchk, lf)
    wfn = fields['wfn']
    assert 'dm_full' in wfn._cache
    assert 'dm_spin' in wfn._cache
    assert wfn.nbasis == 9
    assert wfn.dm_full._array[0, 0] == numbers_full[0]
    assert wfn.dm_full._array[8, 8] == numbers_full[1]
    assert wfn.dm_spin._array[0, 0] == numbers_spin[0]
    assert wfn.dm_spin._array[8, 8] == numbers_spin[1]


def test_load_nitrogen_ccd():
    check_load_nitrogen('ccd', [2.08709209E+00, 3.74723580E-01], [7.25882619E-04, -1.38368575E-02])


def test_load_nitrogen_cis():
    check_load_nitrogen('cis', [2.08741410E+00, 2.09292886E-01], [7.41998558E-04, -6.67582215E-03])


def test_load_nitrogen_mp2():
    check_load_nitrogen('mp2', [2.08710027E+00, 4.86472609E-01], [7.31802950E-04, -2.00028488E-02])


def test_load_nitrogen_mp3():
    check_load_nitrogen('mp3', [2.08674302E+00, 4.91149023E-01], [7.06941101E-04, -1.96276763E-02])


def check_normalization_dm_full_azirine(fn_fchk):
    mol = Molecule.from_file(fn_fchk)
    olp = mol.obasis.compute_overlap(mol.lf)
    dm = mol.wfn.dm_full
    check_dm(dm, olp, mol.lf, 'full', eps=1e-2, occ_max=2)
    assert abs(olp.expectation_value(dm) - 22.0) < 1e-3


def test_normalization_dm_full_azirine_ccd():
    check_normalization_dm_full_azirine(context.get_fn('test/2h-azirine-ccd.fchk'))


def test_normalization_dm_full_azirine_cis():
    check_normalization_dm_full_azirine(context.get_fn('test/2h-azirine-cis.fchk'))


def test_normalization_dm_full_azirine_mp2():
    check_normalization_dm_full_azirine(context.get_fn('test/2h-azirine-mp2.fchk'))


def test_normalization_dm_full_azirine_mp3():
    check_normalization_dm_full_azirine(context.get_fn('test/2h-azirine-mp3.fchk'))


def test_nucnuc():
    fn_fchk = context.get_fn('test/hf_sto3g.fchk')
    mol = Molecule.from_file(fn_fchk)
    assert abs(compute_nucnuc(mol.coordinates, mol.pseudo_numbers) - 4.7247965053) < 1e-5
