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

import os
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest

from ..api import load_one
from ..overlap import compute_overlap
from ..basis import convert_conventions
from ..utils import FileFormatWarning

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


__all__ = ['compute_mulliken_charges', 'compute_1rdm',
           'compare_mols', 'check_orthonormal', 'load_one_warning']


def compute_1rdm(iodata):
    """Compute 1-RDM."""
    coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
    dm = np.dot(coeffs * occs, coeffs.T)
    return dm


def compute_mulliken_charges(iodata):
    """Compute Mulliken charges."""
    dm = compute_1rdm(iodata)
    ov = compute_overlap(iodata.obasis, iodata.atcoords)
    # compute basis function population matrix
    bp = np.sum(np.multiply(dm, ov), axis=1)
    # find basis functions center
    basis_center = []
    for shell in iodata.obasis.shells:
        basis_center.extend([shell.icenter] * shell.nbasis)
    basis_center = np.array(basis_center)
    # compute atomic populations
    populations = np.array([np.sum(bp[basis_center == index])
                            for index in range(iodata.natom)])
    return iodata.atcorenums - np.array(populations)


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
        tmpdir, nline, os.path.basename(fn_orig))
    with open(fn_orig) as f_orig, open(fn_truncated, 'w') as f_truncated:
        for counter, line in enumerate(f_orig):
            if counter >= nline:
                break
            f_truncated.write(line)
        for _ in range(nadd):
            f_truncated.write('\n')
    yield fn_truncated


def compare_mols(mol1, mol2, atol=1.0e-8, rtol=0.0):
    """Compare two IOData objects."""
    assert mol1.title == mol2.title
    assert_equal(mol1.atnums, mol2.atnums)
    assert_equal(mol1.atcorenums, mol2.atcorenums)
    assert_allclose(mol1.atcoords, mol2.atcoords, atol=1e-10)
    # check energy (mol2 might not have energy stored depending on its format)
    if mol1.energy is not None and mol2.energy is not None:
        assert_allclose(mol1.energy, mol2.energy, atol=1e-7)
    elif mol1.energy is None:
        assert mol1.energy == mol2.energy
    # orbital basis
    if mol1.obasis is not None:
        # compare dictionaries
        assert len(mol1.obasis.shells) == len(mol2.obasis.shells)
        for shell1, shell2 in zip(mol1.obasis.shells, mol2.obasis.shells):
            assert shell1.icenter == shell2.icenter
            assert_equal(shell1.angmoms, shell2.angmoms)
            assert shell1.kinds == shell2.kinds
            assert_allclose(shell1.exponents, shell2.exponents, atol=atol, rtol=rtol)
            assert_allclose(shell1.coeffs, shell2.coeffs, atol=atol, rtol=rtol)
        assert mol1.obasis.primitive_normalization == mol2.obasis.primitive_normalization
        # compute and compare Mulliken charges
        charges1 = compute_mulliken_charges(mol1)
        charges2 = compute_mulliken_charges(mol2)
        assert_allclose(charges1, charges2, atol=atol, rtol=rtol)
    else:
        assert mol2.obasis is None
    # wfn
    perm, sgn = convert_conventions(mol1.obasis, mol2.obasis.conventions)
    assert mol1.mo.kind == mol2.mo.kind
    assert_allclose(mol1.mo.occs, mol2.mo.occs, atol=atol, rtol=rtol)
    assert_allclose(mol1.mo.coeffs[perm] * sgn.reshape(-1, 1), mol2.mo.coeffs, atol=atol, rtol=rtol)
    assert_allclose(mol1.mo.energies, mol2.mo.energies, atol=atol, rtol=rtol)
    assert_equal(mol1.mo.irreps, mol2.mo.irreps)
    # operators and density matrices
    cases = [
        ('one_ints', ['olp', 'kin_ao', 'na_ao']),
        ('two_ints', ['er_ao']),
        ('one_rdms', ['scf', 'scf_spin', 'post_scf', 'post_scf_spin']),
    ]
    for attrname, keys in cases:
        d1 = getattr(mol1, attrname)
        d2 = getattr(mol2, attrname)
        for key in keys:
            if key in d1:
                assert key in d2
                matrix1 = d1[key]
                matrix1 = matrix1[perm] * sgn.reshape(-1, 1)
                matrix1 = matrix1[:, perm] * sgn
                matrix2 = d2[key]
                np.testing.assert_equal(matrix1, matrix2)
            else:
                assert key not in d2


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


def load_one_warning(filename: str, fmt: str = None, match: str = None, **kwargs):
    """Call load_one, catching expected FileFormatWarning.

    Parameters
    ----------
    filename
        The file in the unit test data directory to load.
    fmt
        The name of the file format module to use. When not given, it is guessed
        from the filename.
    match
        When given, loading the file is expected to raise a warning whose
        message string contains match.
    **kwargs
        Keyword arguments are passed on to the format-specific load_one function.

    Returns
    -------
    out
        The instance of IOData with data loaded from the input files.

    """
    with path('iodata.test.data', filename) as fn:
        if match is None:
            return load_one(str(fn), fmt, **kwargs)
        with pytest.warns(FileFormatWarning, match=match):
            return load_one(str(fn), fmt, **kwargs)
