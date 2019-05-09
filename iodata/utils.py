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
"""Utility functions module."""


from typing import Tuple, NamedTuple

import numpy as np
import scipy.constants as spc
from scipy.linalg import eigh


__all__ = ['LineIterator', 'set_four_index_element', 'MolecularOrbitals']


# The unit conversion factors below can be used as follows:
# - Conversion to atomic units: distance = 5*angstrom
# - Conversion from atomic units: print(distance/angstrom)
angstrom: float = spc.angstrom / spc.value(u'atomic unit of length')
electronvolt: float = 1 / spc.value(u'hartree-electron volt relationship')
# atomic mass unit (not atomic unit of mass!)
amu: float = 1e-3 / (spc.value(u'electron mass') * spc.value(u'Avogadro constant'))


class LineIterator:
    """Iterator class for looping over lines and keeping track of the line number."""

    def __init__(self, filename: str):
        """Initialize a LineIterator.

        Parameters
        ----------
        filename
            The file that will be read.

        """
        self.filename = filename
        self._f = open(filename)
        self.lineno = 0
        self.stack = []

    def __del__(self):
        self._f.close()

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next line and increase the lineno attribute by one."""
        if self.stack:
            line = self.stack.pop()
        else:
            line = next(self._f)
        self.lineno += 1
        return line

    def error(self, msg: str):
        """Raise an error while reading a file.

        Parameters
        ----------
        msg
            Message to raise alongside filename and line number.

        """
        raise IOError("{}:{} {}".format(self.filename, self.lineno, msg))

    def back(self, line):
        """Go one line back and decrease the lineno attribute by one."""
        self.stack.append(line)
        self.lineno -= 1


class MolecularOrbitals(NamedTuple):
    """Molecular Orbitals Class.

    Attributes
    ----------
    type : str
        Molecular orbital type; choose from 'restricted', 'unrestricted', or 'generalized'.
    norba : int
        Number of alpha molecular orbitals. None in case of type=='generalized'.
    norbb : int
        Number of beta molecular orbitals. None in case of type=='generalized'.
    occs : np.ndarray
        Molecular orbital occupation numbers. The number of elements equals the
        number of columns of coeffs.
    coeffs : np.ndarray
        Molecular orbital basis coefficients.
        In case of restricted: shape = (nbasis, norb_a) = (nbasis, norb_b).
        In case of unrestricted: shape = (nbasis, norb_a + norb_b).
        In case of generalized: shape = (2*nbasis, norb), where norb is the
        total number of orbitals (not defined by other attributes).
    irreps : np.ndarray
        Irreducible representation. The number of elements equals the
        number of columns of coeffs.
    energies : np.ndarray
        Molecular orbital energies. The number of elements equals the
        number of columns of coeffs.

    """

    type: str
    norba: int
    norbb: int
    occs: np.ndarray
    coeffs: np.ndarray
    irreps: np.ndarray
    energies: np.ndarray

    @property
    def nelec(self):
        """Return the total number of electrons."""
        return self.occs.sum()

    @property
    def spinmult(self):
        """Return the spin multiplicity of the Slater determinant."""
        if self.type == 'restricted':
            nbeta = np.clip(self.occs, 0, 1).sum()
            sq = self.nelec - 2 * nbeta
        elif self.type == 'unrestricted':
            sq = self.occs[:self.norba].sum() - self.occs[self.norba:].sum()
        else:
            # Not sure how to do this in a simply way.
            raise NotImplementedError
        return 1 + abs(sq)


def set_four_index_element(four_index_object: np.ndarray, i: int, j: int, k: int, l: int,
                           value: float):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation.

    Parameters
    ----------
    four_index_object
        The four-index object. It will be written to.
        shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
    i, j, k, l
        The indices to assign to.
    value
        The value of the matrix element to store.

    """
    four_index_object[i, j, k, l] = value
    four_index_object[j, i, l, k] = value
    four_index_object[k, j, i, l] = value
    four_index_object[i, l, k, j] = value
    four_index_object[k, l, i, j] = value
    four_index_object[l, k, j, i] = value
    four_index_object[j, k, l, i] = value
    four_index_object[l, i, j, k] = value


def volume(cellvecs: np.ndarray) -> float:
    """Calculate the (generalized) cell volume.

    Parameters
    ----------
    cellvecs
        A numpy matrix of shape (x,3) where x is in {1,2,3}. Each row is one
        cellvector.

    Returns
    -------
    volume
        In case of 3D, the cell volume. In case of 2D, the cell area. In case of
        1D, the cell length.

    """
    nvecs = cellvecs.shape[0]
    if len(cellvecs.shape) == 1 or nvecs == 1:
        return np.linalg.norm(cellvecs)
    if nvecs == 2:
        return np.linalg.norm(np.cross(cellvecs[0], cellvecs[1]))
    if nvecs == 3:
        return np.linalg.det(cellvecs)
    raise ValueError("Argument cellvecs should be of shape (x, 3), where x is in {1, 2, 3}")


def derive_naturals(dm: np.ndarray, overlap: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive natural orbitals from a given density matrix and assign the result to self.

    Parameters
    ----------
    dm
        The density matrix.
        shape=(nbasis, nbasis)
    overlap
        The overlap matrix
        shape=(nbasis, nbasis)

    Returns
    -------
    coeffs
        Orbital coefficients
        shape=(nbasis, nfn)
    occs
        Orbital occupations
        shape=(nfn, )
    energies
        Orbital energies
        shape=(nfn, )

    """
    # Transform density matrix to Fock-like form
    sds = np.dot(overlap.T, np.dot(dm, overlap))
    # Diagonalize and compute eigenvalues
    evals, evecs = eigh(sds, overlap)
    coeffs = np.zeros_like(overlap)
    coeffs = evecs[:, :coeffs.shape[1]]
    occs = evals
    energies = np.zeros(overlap.shape[0])
    return coeffs, occs, energies


def check_dm(dm: np.ndarray, overlap: np.ndarray, eps: float = 1e-4, occ_max: float = 1.0):
    """Check if the density matrix has eigenvalues in the proper range.

    Parameters
    ----------
    dm
        The density matrix
        shape=(nbasis, nbasis), dtype=float
    overlap
        The overlap matrix
        shape=(nbasis, nbasis), dtype=float
    eps
        The threshold on the eigenvalue inequalities.
    occ_max
        The maximum occupation.

    Raises
    ------
    ValueError
        When the density matrix has wrong eigenvalues.

    """
    # construct natural orbitals
    occupations = derive_naturals(dm, overlap)[1]
    if occupations.min() < -eps:
        raise ValueError('The density matrix has eigenvalues considerably smaller than '
                         'zero. error=%e' % (occupations.min()))
    if occupations.max() > occ_max + eps:
        raise ValueError('The density matrix has eigenvalues considerably larger than '
                         'max. error=%e' % (occupations.max() - 1))
