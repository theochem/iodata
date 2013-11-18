# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
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

import numpy as np

import os
from horton import *
from horton.io.wfn import *
from horton.io.test.common import compute_mulliken_charges


def test_load_wfn_low_he_s():
    fn_wfn = context.get_fn('test/he_s_orbital.wfn')
    numbers, coordinates, centers, type_assignment, exponents, mo_count, occ_num, mo_energy, coefficients = load_wfn_low(fn_wfn)
    assert numbers.shape == (1,)
    assert numbers == [2]
    assert coordinates.shape == (1,3)
    assert (coordinates == [0.00, 0.00, 0.00]).all()
    assert centers.shape == (4,)
    assert (centers == [0, 0, 0, 0]).all()
    assert type_assignment.shape == (4,)
    assert (type_assignment == [1, 1, 1, 1]).all()
    assert exponents.shape == (4,)
    assert (exponents == [0.3842163E+02, 0.5778030E+01, 0.1241774E+01, 0.2979640E+00]).all()
    assert mo_count.shape == (1,)
    assert mo_count == [1]
    assert occ_num.shape == (1,)
    assert occ_num == [2.0]
    assert mo_energy.shape == (1,)
    assert mo_energy == [-0.914127]
    assert coefficients.shape == (4, 1)
    assert (coefficients == np.array([0.26139500E+00, 0.41084277E+00, 0.39372947E+00, 0.14762025E+00]).reshape(4,1)).all()


def test_load_wfn_low_h2O():
    fn_wfn = context.get_fn('test/h2o_sto3g.wfn')
    numbers, coordinates, centers, type_assignment, exponents, mo_count, occ_num, mo_energy, coefficients = load_wfn_low(fn_wfn)
    assert numbers.shape == (3,)
    assert (numbers == np.array([8, 1, 1])).all()
    assert coordinates.shape == (3, 3)
    assert (coordinates[0] == [-4.44734101, 3.39697999, 0.00000000]).all()
    assert (coordinates[1] == [-2.58401495, 3.55136194, 0.00000000]).all()
    assert (coordinates[2] == [-4.92380519, 5.20496220, 0.00000000]).all()
    assert centers.shape == (21,)
    assert (centers[:15] == np.zeros(15, int)).all()
    assert (centers[15:] == np.array([1, 1, 1, 2, 2, 2])).all()
    assert type_assignment.shape == (21,)
    assert (type_assignment[:6] == np.ones(6)).all()
    assert (type_assignment[6:15] == np.array([2, 2, 2, 3, 3, 3, 4, 4, 4])).all()
    assert (type_assignment[15:] == np.ones(6)).all()
    assert exponents.shape == (21,)
    assert (exponents[ :3] == [0.1307093E+03, 0.2380887E+02, 0.6443608E+01]).all()
    assert (exponents[5:8] == [0.3803890E+00, 0.5033151E+01, 0.1169596E+01]).all()
    assert (exponents[13:16] == [0.1169596E+01, 0.3803890E+00, 0.3425251E+01]).all()
    assert exponents[-1] == 0.1688554E+00
    assert mo_count.shape == (5,)
    assert (mo_count == [1, 2, 3, 4, 5]).all()
    assert occ_num.shape == (5,)
    assert np.sum(occ_num) == 10.0
    assert (occ_num == [2.0, 2.0, 2.0, 2.0, 2.0]).all()
    assert mo_energy.shape == (5,)
    assert (mo_energy == np.sort(mo_energy)).all()
    assert (mo_energy[:3] == [-20.251576, -1.257549, -0.593857]).all()
    assert (mo_energy[3:] == [ -0.459729, -0.392617]).all()
    assert coefficients.shape == (21, 5)
    assert (coefficients[0] == [0.42273517E+01, -0.99395832E+00, 0.19183487E-11, 0.44235381E+00, -0.57941668E-14]).all()
    assert coefficients[ 6, 2] ==  0.83831599E+00
    assert coefficients[10, 3] ==  0.65034846E+00
    assert coefficients[17, 1] ==  0.12988055E-01
    assert coefficients[-1, 0] == -0.46610858E-03
    assert coefficients[-1,-1] == -0.33277355E-15


def test_setup_permutation1():
    assert (setup_permutation1(np.array([1, 1, 1])) == [0, 1, 2]).all()
    assert (setup_permutation1(np.array([1, 1, 2, 3, 4])) == [0, 1, 2, 3, 4]).all()
    assert (setup_permutation1(np.array([2, 3, 4])) == [0, 1, 2]).all()
    assert (setup_permutation1(np.array([2, 2, 3, 3, 4, 4])) == [0, 2, 4, 1, 3, 5]).all()
    assert (setup_permutation1(np.array([1, 1, 2, 2, 3, 3, 4, 4, 1])) == [0, 1, 2, 4, 6, 3, 5, 7, 8]).all()
    assert (setup_permutation1(np.array([1, 5, 6, 7, 8, 9, 10, 1])) == [0, 1, 2, 3, 4, 5, 6, 7]).all()
    assert (setup_permutation1(np.array([5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])) == [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]).all()
    assert (setup_permutation1(np.array([1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10])) == [0, 1, 3, 5, 2, 4, 6, 7, 8, 9, 10, 11, 12]).all()
    assert (setup_permutation1(np.array([11, 12, 13, 17, 14, 15, 18, 19, 16, 20])) == range(10)).all() # f orbitals
    assert (setup_permutation1(np.array([23, 29, 32, 27, 22, 28, 35, 34, 26, 31, 33, 30, 25, 24, 21])) == range(15)).all() # g orbitals
    assert (setup_permutation1(np.array([23, 29, 32, 27, 22, 28, 35, 34, 26, 31, 33, 30, 25, 24, 21])) == range(15)).all() # g orbitals
    assert (setup_permutation1(np.arange(36, 57)) == range(21)).all() # h orbitals
    assert (setup_permutation1(np.array([1, 1, 11, 12, 13, 17, 14, 15, 18, 19, 16, 20])) == range(12)).all()
    assert (setup_permutation1(np.array([2, 3, 4, 11, 12, 13, 17, 14, 15, 18, 19, 16, 20, 1, 1])) == range(15)).all()


def test_setup_permutation2():
    assert (setup_permutation2(np.array([1, 1, 1])) == [0, 1, 2]).all()
    assert (setup_permutation2(np.array([2, 2, 3, 3, 4, 4])) == [0, 2, 4, 1, 3, 5]).all()
    assert (setup_permutation2(np.array([1, 2, 3, 4, 1])) == [0, 1, 2, 3, 4]).all()
    assert (setup_permutation2(np.array([5, 6, 7, 8, 9, 10])) == [0, 3, 4, 1, 5, 2]).all()
    assert (setup_permutation2(np.repeat([5, 6, 7, 8, 9, 10], 2)) == [0, 6, 8, 2, 10, 4, 1, 7, 9, 3, 11, 5]).all()
    assert (setup_permutation2(np.arange(1, 11)) == [0, 1, 2, 3, 4, 7, 8, 5, 9, 6]).all()
    assert (setup_permutation2(np.array([1, 5, 6, 7, 8, 9, 10, 1])) == [0, 1, 4, 5, 2, 6, 3, 7]).all()
    assert (setup_permutation2(np.array([11, 12, 13, 17, 14, 15, 18, 19, 16, 20])) == [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]).all()
    assert (setup_permutation2(np.array([1, 11, 12, 13, 17, 14, 15, 18, 19, 16, 20, 1])) == [0, 1, 5, 6, 4, 10, 7, 2, 9, 8, 3, 11]).all()
    assert (setup_permutation2(np.array([1, 11, 12, 13, 17, 14, 15, 18, 19, 16, 20, 2, 2, 3, 3, 4, 4])) == [0, 1, 5, 6, 4, 10, 7, 2, 9, 8, 3, 11, 13, 15, 12, 14, 16]).all()
    assert (setup_permutation2(np.array([1, 11, 12, 13, 17, 14, 15, 18, 19, 16, 20, 2, 3, 4, 5, 6, 7, 8, 9, 10])) == [0, 1, 5, 6, 4, 10, 7, 2, 9, 8, 3, 11, 12, 13, 14, 17, 18, 15, 19, 16]).all()
    assert (setup_permutation2(np.arange(36, 57)) == np.arange(21)[::-1]).all()
    assert (setup_permutation2(np.array([23, 29, 32, 27, 22, 28, 35, 34, 26, 31, 33, 30, 25, 24, 21])) == [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]).all()
    assert (setup_permutation2(np.arange(36, 57)) == range(21)[::-1]).all()


def test_setup_mask():
    assert (setup_mask(np.array([2, 3, 4])) == [True, False, False]).all()
    assert (setup_mask(np.array([1, 2, 3, 4, 1, 2, 3, 4])) == [True, True, False, False, True, True, False, False]).all()
    assert (setup_mask(np.array([5, 6, 7, 8, 9, 10])) == [True, False, False, False, False, False]).all()
    assert (setup_mask(np.array([2, 3, 4, 1, 5, 6, 7, 8, 9, 10])) == [True, False, False, True, True, False, False, False, False, False]).all()
    assert (setup_mask(np.arange(11, 21)) == [True, False, False, False, False, False, False, False, False, False]).all()
    assert (setup_mask(np.array([21, 24, 25])) == [True, False, False]).all()
    assert (setup_mask(np.array([11, 21, 36, 1])) == [True, True, True, True]).all()


def check_load_wfn(name):
    #system out of *.wfn file
    fn_wfn = context.get_fn('test/%s.wfn' % name)
    sys1 = System.from_file(fn_wfn)
    ham1= Hamiltonian(sys1, [HartreeFockExchange()])
    energy1 = ham1.compute()
    #system out of *.log and *.fchk files
    sys2 = System.from_file(context.get_fn('test/%s.log' % name), context.get_fn('test/%s.fchk' % name))
    ham2 = Hamiltonian(sys2, [HartreeFockExchange()])
    energy2 = ham2.compute()
    #System check:
    assert sys1.natom == sys2.natom
    assert (abs(sys1.coordinates - sys2.coordinates) < 1e-6).all()
    assert (sys1.numbers == sys2.numbers).all()
    #Basis Set check:
    assert sys1.obasis.nbasis == sys2.obasis.nbasis
    assert (sys1.obasis.shell_map == sys2.obasis.shell_map).all()
    assert (sys1.obasis.shell_types == sys2.obasis.shell_types).all()
    assert (sys1.obasis.nprims == sys2.obasis.nprims).all()
    assert (abs(sys1.obasis.alphas - sys2.obasis.alphas) < 1.e-4).all()
    #Comparing MOs (*.wfn might not contain virtual orbitals):
    n_mo = sys1.wfn.exp_alpha.nfn
    assert (abs(sys1.wfn.exp_alpha.energies - sys2.wfn.exp_alpha.energies[:n_mo]) < 1.e-5).all()
    assert (sys1.wfn.exp_alpha.occupations == sys2.wfn.exp_alpha.occupations[:n_mo]).all()
    assert (abs(sys1.wfn.exp_alpha.coeffs   - sys2.wfn.exp_alpha.coeffs[:,:n_mo]) < 1.e-7).all()
    assert (abs(sys1.get_overlap()._array[:] - sys2.get_overlap()._array[:]) < 1e-6).all()
    assert abs(energy1 - energy2) < 1e-5
    # Check normalization
    sys1.wfn.exp_alpha.check_normalization(sys1.get_overlap(), 1e-5)
    return sys1, energy1


def test_load_wfn_he_s_virtual():
    sys, energy = check_load_wfn('he_s_virtual')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-2.855160426155)) < 1.e-6     #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                     #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_s():
    sys, energy = check_load_wfn('he_s_orbital')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-2.855160426155)) < 1.e-6     #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                    #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_sp():
    sys, energy = check_load_wfn('he_sp_orbital')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-2.859895424589)) < 1.e-6     #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                    #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_spd():
    sys, energy = check_load_wfn('he_spd_orbital')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-2.855319016184)) < 1.e-6     #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                    #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_spdf():
    sys, energy = check_load_wfn('he_spdf_orbital')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-1.100269433080)) < 1.e-6   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                  #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_spdfgh():
    sys, energy = check_load_wfn('he_spdfgh_orbital')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-1.048675168346)) < 1.e-6   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                  #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_he_spdfgh_virtual():
    sys, energy = check_load_wfn('he_spdfgh_virtual')
    assert isinstance(sys.wfn, RestrictedWFN)
    assert abs(energy - (-1.048675168346)) < 1.e-6   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)                  #Check charges
    expected_charge = np.array([0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_h2o_sto3g_decontracted():
    fn_wfn = context.get_fn('test/h2o_sto3g_decontracted.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, RestrictedWFN)
    assert sys.obasis.nbasis == 21
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5)  #Chech normalization
    ham = Hamiltonian(sys, [HartreeFockExchange()])  #Compare to the energy printed in wfn file
    energy = ham.compute()
    assert abs(energy - (-75.162231674351)) < 1.e-5
    charges = compute_mulliken_charges(sys)                  #Check charges
    expected_charge = np.array([-0.546656, 0.273328, 0.273328])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_h2_ccpvqz_virtual():
    fn_wfn = context.get_fn('test/h2_ccpvqz.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, RestrictedWFN)
    assert sys.obasis.nbasis == 74
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5)  #Chech normalization
    assert (abs(sys.obasis.alphas[:5] - [82.64000, 12.41000, 2.824000, 0.7977000, 0.2581000]) < 1.e-5).all()
    assert (sys.wfn.exp_alpha.energies[:5] == [-0.596838, 0.144565, 0.209605, 0.460401, 0.460401]).all()
    assert (sys.wfn.exp_alpha.energies[-5:] == [12.859067, 13.017471, 16.405834, 25.824716, 26.100443]).all()
    assert (sys.wfn.exp_alpha.occupations[:5] == [1.0, 0.0, 0.0, 0.0, 0.0] ).all()
    assert abs(sys.wfn.exp_alpha.occupations.sum() - 1.0) < 1.e-6
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute()
    assert abs(energy - (-1.133504568400)) < 1.e-5   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)     #Check charges
    expected_charge = np.array([0.0, 0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_h2o_sto3g():
    fn_wfn = context.get_fn('test/h2o_sto3g.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, RestrictedWFN)
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5) #Check normalization
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute()
    assert abs(energy - (-74.965901217080)) < 1.e-5   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)   #Check charges
    expected_charge = np.array([-0.330532, 0.165266, 0.165266])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_li_sp_virtual():
    fn_wfn = context.get_fn('test/li_sp_virtual.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, UnrestrictedWFN)
    assert sys.obasis.nbasis == 8
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5) #Check normalization
    sys.wfn.exp_beta.check_normalization(sys.get_overlap(), 1e-5)  #Check normalization
    assert abs(sys.wfn.exp_alpha.occupations.sum() - 2.0) < 1.e-6
    assert abs(sys.wfn.exp_beta.occupations.sum()  - 1.0) < 1.e-6
    assert (sys.wfn.exp_alpha.occupations == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).all()
    assert (sys.wfn.exp_beta.occupations  == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).all()
    assert (abs(sys.wfn.exp_alpha.energies - [-0.087492, -0.080310, 0.158784, 0.158784, 1.078773, 1.090891, 1.090891, 49.643670]) < 1.e-6).all()
    assert (abs(sys.wfn.exp_beta.energies  - [-0.079905, 0.176681, 0.176681, 0.212494, 1.096631, 1.096631, 1.122821, 49.643827]) < 1.e-6).all()
    assert sys.wfn.exp_alpha.coeffs.shape == (8, 8)
    assert sys.wfn.exp_beta.coeffs.shape  == (8, 8)
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute()
    assert abs(energy - (-3.712905542719)) < 1.e-6   #Compare to the energy printed in wfn file
    charges = compute_mulliken_charges(sys)   #Check charges
    expected_charge = np.array([0.0, 0.0])
    assert (abs(charges - expected_charge) < 1e-5).all()


def test_load_wfn_li_sp():
    fn_wfn = context.get_fn('test/li_sp_orbital.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, UnrestrictedWFN)
    assert sys.wfn.exp_alpha.nfn == 2
    assert sys.wfn.exp_beta.nfn == 1


def test_load_wfn_o2():
    fn_wfn = context.get_fn('test/o2_uhf.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, UnrestrictedWFN)
    assert sys.wfn.exp_alpha.nfn == 9
    assert sys.wfn.exp_beta.nfn == 7
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute()
    assert abs(energy - (-149.664140769678)) < 1.e-6   #Compare to the energy printed in wfn file


def test_load_wfn_o2_virtual():
    fn_wfn = context.get_fn('test/o2_uhf_virtual.wfn')
    sys = System.from_file(fn_wfn)
    assert sys.natom == 2
    assert sys.obasis.nbasis == 72
    assert isinstance(sys.wfn, UnrestrictedWFN)
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5) #Check normalization
    sys.wfn.exp_beta.check_normalization(sys.get_overlap(), 1e-5)  #Check normalization
    assert abs(sys.wfn.exp_alpha.occupations.sum() - 9.0) < 1.e-6
    assert abs(sys.wfn.exp_beta.occupations.sum()  - 7.0) < 1.e-6
    assert sys.wfn.exp_alpha.occupations.shape == (44,)
    assert sys.wfn.exp_beta.occupations.shape  == (44,)
    assert (sys.wfn.exp_alpha.occupations[:9] == np.ones(9)).all()
    assert (sys.wfn.exp_beta.occupations[:7]  == np.ones(7)).all()
    assert (sys.wfn.exp_alpha.occupations[9:] == np.zeros(35)).all()
    assert (sys.wfn.exp_beta.occupations[7:]  == np.zeros(37)).all()
    assert sys.wfn.exp_alpha.energies.shape == (44,)
    assert sys.wfn.exp_beta.energies.shape  == (44,)
    assert sys.wfn.exp_alpha.energies[0]  == -20.752000
    assert sys.wfn.exp_alpha.energies[10] == 0.179578
    assert sys.wfn.exp_alpha.energies[-1] ==  51.503193
    assert sys.wfn.exp_beta.energies[0]  == -20.697027
    assert sys.wfn.exp_beta.energies[15] ==  0.322590
    assert sys.wfn.exp_beta.energies[-1] ==  51.535258
    assert sys.wfn.exp_alpha.coeffs.shape == (72, 44)
    assert sys.wfn.exp_beta.coeffs.shape  == (72, 44)
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute()
    assert abs(energy - (-149.664140769678)) < 1.e-6   #Compare to the energy printed in wfn file


def test_load_wfn_lif_fci():
    fn_wfn = context.get_fn('test/lif_fci.wfn')
    sys = System.from_file(fn_wfn)
    assert isinstance(sys.wfn, RestrictedWFN)
    assert sys.natom == 2
    assert sys.obasis.nbasis == 44
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5) #Check normalization
    assert sys.wfn.exp_alpha.occupations.shape == (18,)
    assert abs(sys.wfn.exp_alpha.occupations.sum() - 6.0) < 1.e-6
    assert sys.wfn.exp_alpha.occupations[0] == 2.00000000/2
    assert sys.wfn.exp_alpha.occupations[10] == 0.00128021/2
    assert sys.wfn.exp_alpha.occupations[-1] == 0.00000054/2
    assert sys.wfn.exp_alpha.energies.shape == (18,)
    assert sys.wfn.exp_alpha.energies[0] == -26.09321253
    assert sys.wfn.exp_alpha.energies[15] == 1.70096290
    assert sys.wfn.exp_alpha.energies[-1] == 2.17434072
    assert sys.wfn.exp_alpha.coeffs.shape == (44, 18)
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute() #cannot be compared!
    kin = sys.extra['energy_kin']
    nn = sys.extra['energy_nn']
    expected_kin = 106.9326884815  #FCI kinetic energy
    expected_nn = 9.1130265227
    assert (kin - expected_kin) < 1.e-6
    assert (nn - expected_nn) < 1.e-6
    charges = compute_mulliken_charges(sys)   #Check charges
    expected_charge = np.array([-0.645282, 0.645282])
    assert (abs(charges - expected_charge) < 1e-5).all()
    points = np.array([[0.0, 0.0,-0.17008], [0.0, 0.0, 0.0], [0.0, 0.0, 0.03779]])
    density = sys.compute_grid_density(points)
    assert (abs(density - [0.492787, 0.784545, 0.867723]) < 1.e-4).all()


def test_load_wfn_lih_cation_fci():
    fn_wfn = context.get_fn('test/lih_cation_fci.wfn')
    sys = System.from_file(fn_wfn)
    assert sys.natom == 2
    assert sys.obasis.nbasis == 26
    assert (sys.numbers == [3, 1]).all()
    sys.wfn.exp_alpha.check_normalization(sys.get_overlap(), 1e-5) #Check normalization
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    energy = ham.compute() #cannot be compared!
    kin = sys.extra['energy_kin']
    nn = sys.extra['energy_nn']
    expected_kin = 7.7989675958  #FCI kinetic energy
    expected_nn = 0.9766607347
    assert (kin - expected_kin) < 1.e-6
    assert (nn - expected_nn) < 1.e-6
    assert sys.wfn.exp_alpha.occupations.shape == (11,)
    assert abs(sys.wfn.exp_alpha.occupations.sum() - 1.5) < 1.e-6
    charges = compute_mulliken_charges(sys)   #Check charges
    expected_charge = np.array([0.913206, 0.086794])
    assert (abs(charges - expected_charge) < 1e-5).all()
    point = np.array([])
