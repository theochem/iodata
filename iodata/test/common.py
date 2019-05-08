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
"""Utilities for unit tests."""

from os import path
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..overlap import compute_overlap
from ..basis import convert_conventions

__all__ = ['compute_mulliken_charges', 'compute_1rdm',
           'compare_mols', 'check_orthonormal']


def compute_1rdm(iodata):
    """Compute 1-RDM."""
    coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
    dm = np.dot(coeffs * occs, coeffs.T)
    return dm


def compute_mulliken_charges(iodata, atcorenums=None):
    """Compute Mulliken charges."""
    if atcorenums is None:
        atcorenums = iodata.atcorenums
    dm = compute_1rdm(iodata)
    ov = compute_overlap(iodata.obasis)
    # compute basis function population matrix
    bp = np.sum(np.multiply(dm, ov), axis=1)
    # find basis functions center
    basis_center = []
    for shell in iodata.obasis.shells:
        basis_center.extend([shell.icenter] * shell.nbasis)
    basis_center = np.array(basis_center)
    # compute atomic populations
    populations = np.zeros(len(iodata.obasis.centers))
    for index in range(len(iodata.obasis.centers)):
        populations[index] = np.sum(bp[basis_center == index])
    assert_equal(atcorenums.shape, populations.shape)
    return atcorenums - np.array(populations)


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
    assert_equal(mol1.atnums, mol2.atnums)
    assert_allclose(mol1.coordinates, mol2.coordinates)
    # orbital basis
    if mol1.obasis is not None:
        # compare dictionaries
        assert_allclose(mol1.obasis.centers, mol2.obasis.centers)
        assert len(mol1.obasis.shells) == len(mol2.obasis.shells)
        for shell1, shell2 in zip(mol1.obasis.shells, mol2.obasis.shells):
            assert shell1.icenter == shell2.icenter
            assert_equal(shell1.angmoms, shell2.angmoms)
            assert shell1.kinds == shell2.kinds
            assert_allclose(shell1.exponents, shell2.exponents, atol=1e-8)
            assert_allclose(shell1.coeffs, shell2.coeffs, atol=1e-8)
        assert len(mol1.obasis.conventions) == len(mol2.obasis.conventions)
        for key, conv in mol1.obasis.conventions.items():
            s1 = set(word.lstrip('-') for word in conv)
            s2 = set(word.lstrip('-') for word in mol2.obasis.conventions[key])
            assert s1 == s2, (s1, s2)
        assert mol1.obasis.primitive_normalization == mol2.obasis.primitive_normalization
    else:
        assert mol2.obasis is None
    # wfn
    permutation, signs = convert_conventions(mol1.obasis, mol2.obasis.conventions)
    assert mol1.mo.type == mol2.mo.type
    assert_allclose(mol1.mo.occs, mol2.mo.occs)
    assert_allclose(mol1.mo.energies, mol2.mo.energies)
    assert_allclose(mol1.mo.coeffs[permutation] * signs.reshape(-1, 1), mol2.mo.coeffs, atol=1e-8)
    # operators
    for key in 'olp', 'kin', 'na', 'er', 'dm_full_mp2', 'dm_spin_mp2', \
               'dm_full_mp3', 'dm_spin_mp3', 'dm_full_ci', 'dm_spin_ci', \
               'dm_full_cc', 'dm_spin_cc', 'dm_full_scf', 'dm_spin_scf':
        if hasattr(mol1, key):
            assert hasattr(mol2, key)
            matrix1 = getattr(mol1, key)
            matrix1 = matrix1[permutation] * signs.reshape(-1, 1)
            matrix1 = matrix1[:, permutation] * signs
            matrix2 = getattr(mol2, key)
            np.testing.assert_equal(matrix1, matrix2)
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
