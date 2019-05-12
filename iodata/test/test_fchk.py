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
# pylint: disable=unsubscriptable-object,no-member
"""Test iodata.formats.fchk module."""


import numpy as np
from numpy.testing import assert_equal, assert_allclose

import pytest

from ..api import load_one, load_many
from ..formats.fchk import load_one as load_fchk
from ..orbitals import UnrestrictedOrbitals, RestrictedOrbitals
from ..overlap import compute_overlap
from ..utils import check_dm, LineIterator

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_fchk_nonexistent():
    with pytest.raises(IOError):
        with path('iodata.test.data', 'fubar_crap.fchk') as fn:
            load_one(str(fn))


def load_fchk_helper_internal(fn_fchk):
    """Load a testing fchk file with iodata.formats.fchk.load directly."""
    with path('iodata.test.data', fn_fchk) as fn:
        lit = LineIterator(fn)
        return load_fchk(lit)


def load_fchk_helper(fn_fchk):
    """Load a testing fchk file with iodata.iodata.load_one."""
    with path('iodata.test.data', fn_fchk) as fn:
        return load_one(fn)


def test_load_fchk_hf_sto3g_num():
    mol = load_fchk_helper('hf_sto3g.fchk')
    assert mol.title == 'hf_sto3g'
    assert mol.run_type == 'energy'
    assert mol.lot == 'rhf'
    assert mol.obasis_name == 'sto-3g'
    assert isinstance(mol.mo, RestrictedOrbitals)
    assert mol.spinpol == 0
    assert mol.obasis.nbasis == 6
    assert len(mol.obasis.shells) == 3
    shell0 = mol.obasis.shells[0]
    assert shell0.icenter == 0
    assert shell0.angmoms == [0]
    assert shell0.kinds == ['c']
    assert_allclose(shell0.exponents, np.array([1.66679134E+02, 3.03608123E+01, 8.21682067E+00]))
    assert_allclose(shell0.coeffs,
                    np.array([[1.54328967E-01], [5.35328142E-01], [4.44634542E-01]]))
    assert shell0.nprim == 3
    assert shell0.ncon == 1
    assert shell0.nbasis == 1
    shell1 = mol.obasis.shells[1]
    assert shell1.icenter == 0
    assert shell1.angmoms == [0, 1]
    assert shell1.kinds == ['c', 'c']
    assert_allclose(shell1.exponents, np.array([6.46480325E+00, 1.50228124E+00, 4.88588486E-01]))
    assert_allclose(shell1.coeffs,
                    np.array([[-9.99672292E-02, 1.55916275E-01],
                              [3.99512826E-01, 6.07683719E-01],
                              [7.00115469E-01, 3.91957393E-01]]))
    assert shell1.nprim == 3
    assert shell1.ncon == 2
    assert shell1.nbasis == 4
    shell2 = mol.obasis.shells[2]
    assert shell2.nprim == 3
    assert shell2.ncon == 1
    assert shell2.nbasis == 1
    assert mol.obasis.primitive_normalization == 'L2'
    assert len(mol.atcoords) == len(mol.atnums)
    assert mol.atcoords.shape[1] == 3
    assert len(mol.atnums) == 2
    assert_allclose(mol.energy, -9.856961609951867E+01)
    assert_allclose(mol.atcharges['mulliken'], [0.45000000E+00, 4.22300000E+00])
    assert_allclose(mol.atcharges['npa'], [3.50000000E+00, 1.32000000E+00])
    assert_allclose(mol.atcharges['esp'], [0.77700000E+00, 0.66600000E+00])


def test_load_fchk_h_sto3g_num():
    fields = load_fchk_helper_internal('h_sto3g.fchk')
    assert fields['title'] == 'h_sto3g'
    assert len(fields['obasis'].shells) == 1
    assert fields['obasis'].nbasis == 1
    assert fields['obasis'].shells[0].nprim == 3
    assert len(fields['atcoords']) == len(fields['atnums'])
    assert fields['atcoords'].shape[1] == 3
    assert len(fields['atnums']) == 1
    assert_allclose(fields['energy'], -4.665818503844346E-01)


def test_load_fchk_o2_cc_pvtz_pure_num():
    mol = load_fchk_helper('o2_cc_pvtz_pure.fchk')
    assert mol.run_type == 'energy'
    assert mol.lot == 'rhf'
    assert mol.obasis_name == 'cc-pvtz'
    assert len(mol.obasis.shells) == 20
    assert mol.obasis.nbasis == 60
    assert mol.natom == 2
    assert_allclose(mol.energy, -1.495944878699246E+02)


def test_load_fchk_o2_cc_pvtz_cart_num():
    fields = load_fchk_helper_internal('o2_cc_pvtz_cart.fchk')
    assert len(fields['obasis'].shells) == 20
    assert fields['obasis'].nbasis == 70
    assert len(fields['atcoords']) == len(fields['atnums'])
    assert fields['atcoords'].shape[1] == 3
    assert len(fields['atnums']) == 2
    assert_allclose(fields['energy'], -1.495953594545721E+02)


def test_load_fchk_water_sto3g_hf():
    fields = load_fchk_helper_internal('water_sto3g_hf_g03.fchk')
    assert len(fields['obasis'].shells) == 4
    assert fields['obasis'].nbasis == 7
    assert len(fields['atcoords']) == len(fields['atnums'])
    assert fields['atcoords'].shape[1] == 3
    assert len(fields['atnums']) == 3
    mo = fields['mo']
    assert_allclose(mo.energies[0], -2.02333942E+01, atol=1.e-7)
    assert_allclose(mo.energies[-1], 7.66134805E-01, atol=1.e-7)
    assert_allclose(mo.coeffs[0, 0], 0.99410, atol=1.e-4)
    assert_allclose(mo.coeffs[1, 0], 0.02678, atol=1.e-4)
    assert_allclose(mo.coeffs[-1, 2], -0.44154, atol=1.e-4)
    assert abs(mo.coeffs[3, -1]) < 1e-4
    assert_allclose(mo.coeffs[4, -1], -0.82381, atol=1.e-4)
    assert_equal(mo.occs.sum(), 10)
    assert_equal(mo.occs.min(), 0.0)
    assert_equal(mo.occs.max(), 2.0)
    energy = fields['energy']
    assert_allclose(energy, -7.495929232844363E+01)


def test_load_fchk_lih_321g_hf():
    mol = load_fchk_helper('li_h_3-21G_hf_g09.fchk')
    assert len(mol.obasis.shells) == 5
    assert mol.obasis.nbasis == 11
    assert len(mol.atcoords) == len(mol.atnums)
    assert mol.atcoords.shape[1] == 3
    assert len(mol.atnums) == 2
    assert_allclose(mol.energy, -7.687331212191968E+00)

    assert_allclose(mol.mo.energiesa[0], (-2.76117), atol=1.e-4)
    assert_allclose(mol.mo.energiesa[-1], 0.97089, atol=1.e-4)
    assert_allclose(mol.mo.coeffsa[0, 0], 0.99105, atol=1.e-4)
    assert_allclose(mol.mo.coeffsa[1, 0], 0.06311, atol=1.e-4)
    assert mol.mo.coeffsa[3, 2] < 1.e-4
    assert_allclose(mol.mo.coeffsa[-1, 9], 0.13666, atol=1.e-4)
    assert_allclose(mol.mo.coeffsa[4, -1], 0.17828, atol=1.e-4)
    assert_equal(mol.mo.occsa.sum(), 2)
    assert_equal(mol.mo.occsa.min(), 0.0)
    assert_equal(mol.mo.occsa.max(), 1.0)

    assert_allclose(mol.mo.energiesb[0], -2.76031, atol=1.e-4)
    assert_allclose(mol.mo.energiesb[-1], 1.13197, atol=1.e-4)
    assert_allclose(mol.mo.coeffsb[0, 0], 0.99108, atol=1.e-4)
    assert_allclose(mol.mo.coeffsb[1, 0], 0.06295, atol=1.e-4)
    assert abs(mol.mo.coeffsb[3, 2]) < 1e-4
    assert_allclose(mol.mo.coeffsb[-1, 9], 0.80875, atol=1.e-4)
    assert_allclose(mol.mo.coeffsb[4, -1], -0.15503, atol=1.e-4)
    assert_equal(mol.mo.occsb.sum(), 1)
    assert_equal(mol.mo.occsb.min(), 0.0)
    assert_equal(mol.mo.occsb.max(), 1.0)
    assert_equal(mol.mo.occsa.shape[0], mol.mo.coeffsa.shape[0])
    assert_equal(mol.mo.occsb.shape[0], mol.mo.coeffsb.shape[0])


def test_load_fchk_ghost_atoms():
    # Load fchk file with ghost atoms
    mol = load_fchk_helper('water_dimer_ghost.fchk')
    # There should be 3 real atoms and 3 ghost atoms
    assert mol.natom == 6
    assert_equal(mol.atnums, [1, 8, 1, 1, 8, 1])
    assert_equal(mol.atcorenums, [1.0, 8.0, 1.0, 0.0, 0.0, 0.0])
    assert_equal(mol.atcoords.shape[0], 6)
    assert_equal(mol.atcharges['mulliken'].shape[0], 6)


def test_load_fchk_ch3_rohf_g03():
    mol = load_fchk_helper('ch3_rohf_sto3g_g03.fchk')
    assert_equal(mol.mo.occs.shape[0], mol.mo.coeffs.shape[0])
    assert_equal(mol.mo.occs.sum(), 9.0)
    assert_equal(mol.mo.occs.min(), 0.0)
    assert_equal(mol.mo.occs.max(), 2.0)
    assert 'scf' not in mol.one_rdms


def check_load_azirine(key, numbers):
    """Perform some basic checks on a azirine fchk file."""
    mol = load_fchk_helper('2h-azirine-{}.fchk'.format(key))
    assert mol.obasis.nbasis == 33
    dm = mol.one_rdms['post_scf']
    assert_equal(dm[0, 0], numbers[0])
    assert_equal(dm[32, 32], numbers[1])


def test_load_azirine_cc():
    check_load_azirine('cc', [2.08221382E+00, 1.03516466E-01])


def test_load_azirine_ci():
    check_load_azirine('ci', [2.08058265E+00, 6.12011064E-02])


def test_load_azirine_mp2():
    check_load_azirine('mp2', [2.08253448E+00, 1.09305208E-01])


def test_load_azirine_mp3():
    check_load_azirine('mp3', [2.08243417E+00, 1.02590815E-01])


def check_load_nitrogen(key, numbers, numbers_spin):
    """Perform some basic checks on a nitrogen fchk file."""
    mol = load_fchk_helper('nitrogen-{}.fchk'.format(key))
    assert mol.obasis.nbasis == 9
    dm = mol.one_rdms['post_scf']
    assert_equal(dm[0, 0], numbers[0])
    assert_equal(dm[8, 8], numbers[1])
    dm_spin = mol.one_rdms['post_scf_spin']
    assert_equal(dm_spin[0, 0], numbers_spin[0])
    assert_equal(dm_spin[8, 8], numbers_spin[1])


def test_load_nitrogen_cc():
    check_load_nitrogen('cc', [2.08709209E+00, 3.74723580E-01], [7.25882619E-04, -1.38368575E-02])


def test_load_nitrogen_ci():
    check_load_nitrogen('ci', [2.08741410E+00, 2.09292886E-01], [7.41998558E-04, -6.67582215E-03])


def test_load_nitrogen_mp2():
    check_load_nitrogen('mp2', [2.08710027E+00, 4.86472609E-01], [7.31802950E-04, -2.00028488E-02])


def test_load_nitrogen_mp3():
    check_load_nitrogen('mp3', [2.08674302E+00, 4.91149023E-01], [7.06941101E-04, -1.96276763E-02])


def check_normalization_dm_azirine(key):
    """Perform some basic checks on a 2h-azirine fchk file."""
    mol = load_fchk_helper('2h-azirine-{}.fchk'.format(key))
    olp = compute_overlap(mol.obasis, mol.atcoords)
    dm = mol.one_rdms['post_scf']
    check_dm(dm, olp, eps=1e-2, occ_max=2)
    assert_allclose(np.einsum('ab,ba', olp, dm), 22.0, atol=1.e-3)


def test_normalization_dm_azirine_cc():
    check_normalization_dm_azirine('cc')


def test_normalization_dm_azirine_ci():
    check_normalization_dm_azirine('ci')


def test_normalization_dm_azirine_mp2():
    check_normalization_dm_azirine('mp2')


def test_normalization_dm_azirine_mp3():
    check_normalization_dm_azirine('mp3')


def test_load_water_hfs_321g():
    mol = load_fchk_helper('water_hfs_321g.fchk')
    pol = mol.extra['polarizability_tensor']
    assert_allclose(pol[0, 0], 7.23806684E+00)
    assert_allclose(pol[1, 1], 8.04213953E+00)
    assert_allclose(pol[1, 2], 1.20021770E-10)
    assert_allclose(mol.moments[(1, 'c')],
                    [-5.82654324E-17, 0.00000000E+00, -8.60777067E-01])
    assert_allclose(mol.moments[(2, 'c')],
                    [-8.89536026E-01,  # xx
                     8.28408371E-17,  # xy
                     4.89353090E-17,  # xz
                     1.14114241E+00,  # yy
                     -5.47382213E-48,  # yz
                     -2.51606382E-01])  # zz


def test_load_monosilicic_acid_hf_lan():
    mol = load_fchk_helper('monosilicic_acid_hf_lan.fchk')
    assert_allclose(mol.moments[(1, 'c')],
                    [-6.05823053E-01, -9.39656399E-03, 4.18948869E-01])
    assert_allclose(mol.moments[(2, 'c')],
                    [2.73609152E+00,  # xx
                     -6.65787832E-02,  # xy
                     2.11973730E-01,  # xz
                     8.97029351E-01,  # yy
                     -1.38159653E-02,  # yz
                     -3.63312087E+00])  # zz
    assert_allclose(mol.atgradient[0], [0.0, 0.0, 0.0])


def load_fchk_trj_helper(fn_fchk):
    """Load a trajectory from a testing fchk file with iodata.iodata.load_many."""
    with path('iodata.test.data', fn_fchk) as fn:
        return list(load_many(fn))


def check_trj_basics(trj, nsteps, title, irc):
    """Check sizes of arrays, step and point attributes."""
    # Make a copy of the list, so we can pop items without destroying the original.
    trj = list(trj)
    assert len(trj) == sum(nsteps)
    natom = trj[0].natom
    for ipoint, nstep in enumerate(nsteps):
        for istep in range(nstep):
            mol = trj.pop(0)
            assert mol.extra['ipoint'] == ipoint
            assert mol.extra['npoint'] == len(nsteps)
            assert mol.extra['istep'] == istep
            assert mol.extra['nstep'] == nstep
            assert mol.natom == natom
            assert mol.atnums.shape == (natom, )
            assert mol.atcorenums.shape == (natom, )
            assert mol.atcoords.shape == (natom, 3)
            assert mol.atgradient.shape == (natom, 3)
            assert mol.title == title
            assert hasattr(mol, 'energy')
            assert ('reaction_coordinate' in mol.extra) ^ (not irc)


def test_peroxide_opt():
    trj = load_fchk_trj_helper("peroxide_opt.fchk")
    check_trj_basics(trj, [5], 'opt', False)
    assert_allclose(trj[0].energy, -1.48759755E+02)
    assert_allclose(trj[1].energy, -1.48763504E+02)
    assert_allclose(trj[-1].energy, -1.48764883E+02)
    assert_allclose(trj[0].atcoords[1],
                    [9.02056208E-17, -1.37317707E+00, 0.00000000E+00])
    assert_allclose(trj[-1].atcoords[-1],
                    [-1.85970174E+00, -1.64631025E+00, 0.00000000E+00])
    assert_allclose(trj[2].atgradient[0],
                    [-5.19698814E-03, -1.17503170E-03, -1.06165077E-15])
    assert_allclose(trj[3].atgradient[2],
                    [-8.70435823E-04, 1.44609443E-03, -3.79091290E-16])


def test_peroxide_tsopt():
    trj = load_fchk_trj_helper("peroxide_tsopt.fchk")
    check_trj_basics(trj, [3], 'tsopt', False)
    assert_allclose(trj[0].energy, -1.48741996E+02)
    assert_allclose(trj[1].energy, -1.48750392E+02)
    assert_allclose(trj[2].energy, -1.48750432E+02)
    assert_allclose(trj[0].atcoords[3],
                    [-2.40150648E-01, -1.58431001E+00, 1.61489448E+00])
    assert_allclose(trj[2].atcoords[2],
                    [1.26945011E-03, 1.81554334E+00, 1.62426250E+00])
    assert_allclose(trj[1].atgradient[1],
                    [-8.38752120E-04, 3.46889422E-03, 1.96559245E-03])
    assert_allclose(trj[-1].atgradient[0],
                    [2.77986102E-05, -1.74709101E-05, 2.45875530E-05])


def test_peroxide_relaxed_scan():
    trj = load_fchk_trj_helper("peroxide_relaxed_scan.fchk")
    check_trj_basics(trj, [6, 1, 1, 1, 2, 2], 'relaxed scan', False)
    assert_allclose(trj[0].energy, -1.48759755E+02)
    assert_allclose(trj[10].energy, -1.48764896E+02)
    assert_allclose(trj[-1].energy, -1.48764905E+02)
    assert_allclose(trj[1].atcoords[3],
                    [-1.85942837E+00, -1.70565735E+00, -1.11022302E-16])
    assert_allclose(trj[5].atcoords[0],
                    [-1.21430643E-16, 1.32466211E+00, 3.46944695E-17])
    assert_allclose(trj[8].atgradient[1],
                    [2.46088230E-04, -4.46299289E-04, -3.21529658E-05])
    assert_allclose(trj[9].atgradient[2],
                    [-1.02574260E-04, -3.33214833E-04, 5.27406641E-05])


def test_peroxide_irc():
    trj = load_fchk_trj_helper("peroxide_irc.fchk")
    check_trj_basics(trj, [21], 'irc', True)
    assert_allclose(trj[0].energy, -1.48750432E+02)
    assert_allclose(trj[5].energy, -1.48752713E+02)
    assert_allclose(trj[-1].energy, -1.48757803E+02)
    assert trj[0].extra['reaction_coordinate'] == 0.0
    assert_allclose(trj[1].extra['reaction_coordinate'], 1.05689581E-01)
    assert_allclose(trj[10].extra['reaction_coordinate'], 1.05686037E+00)
    assert_allclose(trj[-1].extra['reaction_coordinate'], -1.05685760E+00)
    assert_allclose(trj[0].atcoords[2],
                    [-1.94749866E+00, -5.22905491E-01, -1.47814774E+00])
    assert_allclose(trj[10].atcoords[1],
                    [1.31447798E+00, 1.55994117E-01, -5.02320861E-02])
    assert_allclose(trj[15].atgradient[3],
                    [4.73066407E-04, -5.36135653E-03, 2.16301508E-04])
    assert_allclose(trj[-1].atgradient[0],
                    [-1.27710420E-03, -6.90543903E-03, 4.49870405E-03])


def test_atgradient():
    mol = load_fchk_helper('peroxide_tsopt.fchk')
    assert_allclose(mol.atgradient[0], [2.77986102E-05, -1.74709101E-05, 2.45875530E-05])
    assert_allclose(mol.atgradient[-1], [2.03469628E-05, 1.49353694E-05, -2.45875530E-05])


def test_athessian():
    mol = load_fchk_helper('peroxide_tsopt.fchk')
    assert mol.run_type == 'freq'
    assert mol.lot == 'rhf'
    assert mol.obasis_name == 'sto-3g'
    assert_allclose(mol.athessian[0, 0], -1.49799052E-02)
    assert_allclose(mol.athessian[-1, -1], 5.83032386E-01)
    assert_allclose(mol.athessian[0, 1], 5.07295215E-05)
    assert_allclose(mol.athessian[1, 0], 5.07295215E-05)
    assert mol.athessian.shape == (3 * mol.natom, 3 * mol.natom)


def test_atfrozen():
    mol = load_fchk_helper('peroxide_tsopt.fchk')
    assert_equal(mol.atfrozen, [False, False, False, True])


def test_atmasses():
    mol = load_fchk_helper('peroxide_tsopt.fchk')
    assert_allclose(mol.atmasses[0], 29156.94, atol=0.1)
    assert_allclose(mol.atmasses[-1], 1837.15, atol=0.1)


def test_spinpol():
    mol1 = load_fchk_helper('ch3_rohf_sto3g_g03.fchk')
    assert isinstance(mol1.mo, RestrictedOrbitals)
    assert mol1.spinpol == 1
    mol2 = load_fchk_helper('li_h_3-21G_hf_g09.fchk')
    assert isinstance(mol2.mo, UnrestrictedOrbitals)
    assert mol2.spinpol == 1
    with pytest.raises(TypeError):
        mol2.spinpol = 2


def test_nelec_charge():
    mol1 = load_fchk_helper('ch3_rohf_sto3g_g03.fchk')
    assert mol1.nelec == 9
    assert mol1.charge == 0
    mol2 = load_fchk_helper('li_h_3-21G_hf_g09.fchk')
    assert mol2.nelec == 3
    assert mol2.charge == 1
    with pytest.raises(TypeError):
        mol2.nelec = 4
    with pytest.raises(TypeError):
        mol2.charge = 0
