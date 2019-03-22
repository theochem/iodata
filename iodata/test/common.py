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
"""Utilities for unit tests."""

from os import path
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..overlap import compute_overlap, get_shell_nbasis

__all__ = ['compute_mulliken_charges', 'compute_1rdm',
           'compare_mols', 'check_orthonormal']


def compute_1rdm(iodata):
    """Compute 1-RDM."""
    if hasattr(iodata, 'mo'):
        coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
        dm = np.dot(coeffs * occs, coeffs.T)
    if hasattr(iodata, 'orb_alpha_coeffs'):
        coeffs, occs = iodata.orb_alpha_coeffs, iodata.orb_alpha_occs
        dm = np.dot(coeffs * occs, coeffs.T)
        if hasattr(iodata, 'orb_beta_coeffs'):
            coeffs, occs = iodata.orb_beta_coeffs, iodata.orb_beta_occs
            dm += np.dot(coeffs * occs, coeffs.T)
        else:
            dm *= 2
    return dm


def compute_mulliken_charges(iodata, pseudo_numbers=None):
    """Compute Mulliken charges."""
    if pseudo_numbers is None:
        pseudo_numbers = iodata.pseudo_numbers
    dm = compute_1rdm(iodata)
    ov = compute_overlap(**iodata.obasis)
    # compute basis function population matrix
    bp = np.sum(np.multiply(dm, ov), axis=1)
    # find basis functions center
    basis_center = []
    for (ci, ti) in zip(iodata.obasis["shell_map"],
                        iodata.obasis["shell_types"]):
        basis_center.extend([ci] * get_shell_nbasis(ti))
    basis_center = np.array(basis_center)
    # compute atomic populations
    populations = np.zeros(len(iodata.obasis["centers"]))
    for index in range(len(iodata.obasis["centers"])):
        populations[index] = np.sum(bp[basis_center == index])
    assert_equal(pseudo_numbers.shape, populations.shape)
    return pseudo_numbers - np.array(populations)


@contextmanager
def truncated_file(fn_orig, nline, nadd, tmpdir):
    """Make a temporary truncated copy of a file.

    Parameters
    ----------
    fn_orig : str
        The file to be truncated.
    nline : int
        The number of lines to retain.
    nadd : int
        The number of empty lines to add.
    tmpdir : str
        A temporary directory where the truncated file is stored.

    """
    fn_truncated = '%s/truncated_%i_%s' % (
        tmpdir, nline, path.basename(fn_orig))
    with open(fn_orig) as f_orig, open(fn_truncated, 'w') as f_truncated:
        for counter, line in enumerate(f_orig):
            if counter >= nline:
                break
            f_truncated.write(line)
        for _ in range(nadd):
            f_truncated.write('\n')
    yield fn_truncated


def compare_mols(mol1, mol2):
    """Compare two IOData objects."""
    assert getattr(mol1, 'title') == getattr(mol2, 'title')
    assert_equal(mol1.numbers, mol2.numbers)
    assert_allclose(mol1.coordinates, mol2.coordinates)
    # orbital basis
    if mol1.obasis is not None:
        # compare dictionaries
        assert_equal(len(mol1.obasis), len(mol2.obasis))
        for k, v in mol1.obasis.items():
            assert_allclose(v, mol2.obasis[k], atol=1.e-8)
    else:
        assert mol2.obasis is None
    # wfn
    assert_allclose(mol1.orb_alpha, mol2.orb_alpha)
    assert_allclose(mol1.orb_alpha_coeffs, mol2.orb_alpha_coeffs)
    assert_allclose(mol1.orb_alpha_energies, mol2.orb_alpha_energies)
    assert_allclose(mol1.orb_alpha_occs, mol2.orb_alpha_occs)
    if hasattr(mol1, "orb_beta"):
        assert_allclose(mol1.orb_beta, mol2.orb_beta)
        assert_allclose(mol1.orb_beta_coeffs, mol2.orb_beta_coeffs)
        assert_allclose(mol1.orb_beta_energies, mol2.orb_beta_energies)
        assert_allclose(mol1.orb_beta_occs, mol2.orb_beta_occs)

    # operators
    for key in 'olp', 'kin', 'na', 'er', 'dm_full_mp2', 'dm_spin_mp2', \
               'dm_full_mp3', 'dm_spin_mp3', 'dm_full_ci', 'dm_spin_ci', \
               'dm_full_cc', 'dm_spin_cc', 'dm_full_scf', 'dm_spin_scf':
        if hasattr(mol1, key):
            assert hasattr(mol2, key)
            np.testing.assert_equal(getattr(mol1, key), getattr(mol2, key))
        else:
            assert not hasattr(mol2, key)


def check_orthonormal(mo_coeffs, ao_overlap, atol=1e-5):
    """Check that molecular orbitals are orthogonal and normalized.

    Parameters
    ----------
    mo_coeffs : np.ndarray, shape=(nbasis, mo_count)
        Molecular orbital coefficients.
    ao_overlap : np.ndarray, shape=(nbasis, nbasis)
        Atomic orbital overlap matrix.
    atol : float
        Absolute tolerance in deviation from identity matrix.

    """
    # compute MO overlap & number of MO orbitals
    mo_overlap = np.dot(mo_coeffs.T, np.dot(ao_overlap, mo_coeffs))
    mo_count = mo_coeffs.shape[1]
    message = 'Molecular orbitals are not orthonormal!'
    assert_allclose(mo_overlap, np.eye(mo_count),
                    rtol=0., atol=atol, err_msg=message)
