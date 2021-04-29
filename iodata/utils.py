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


from typing import Tuple
import warnings

import attr
import numpy as np
import scipy.constants as spc
from scipy.linalg import eigh

from .attrutils import validate_shape


__all__ = ['LineIterator', 'Cube', 'set_four_index_element', 'volume',
           'derive_naturals', 'check_dm']


# The unit conversion factors below can be used as follows:
# - Conversion to atomic units: distance = 5*angstrom
# - Conversion from atomic units: print(distance/angstrom)
angstrom: float = spc.angstrom / spc.value(u'atomic unit of length')
electronvolt: float = 1 / spc.value(u'hartree-electron volt relationship')
# Unit conversion for Gromacs gro files
meter: float = 1 / spc.value(u'Bohr radius')
nanometer: float = 1e-9 * meter
second: float = 1 / spc.value(u'atomic unit of time')
picosecond: float = 1e-12 * second
# atomic mass unit (not atomic unit of mass!)
amu: float = 1e-3 / (spc.value(u'electron mass') * spc.value(u'Avogadro constant'))
kcalmol: float = 1e3 * spc.calorie / spc.value('Avogadro constant') / spc.value('Hartree energy')
calmol: float = spc.calorie / spc.value('Avogadro constant') / spc.value('Hartree energy')
kjmol: float = 1e3 / spc.value('Avogadro constant') / spc.value('Hartree energy')


class FileFormatError(IOError):
    """Raised when incorrect content is encountered when loading files."""


class FileFormatWarning(Warning):
    """Raised when incorrect content is encountered and fixed when loading files."""


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
        self.f = open(filename)  # pylint: disable=consider-using-with
        self.lineno = 0
        self.stack = []

    def __del__(self):
        self.f.close()

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next line and increase the lineno attribute by one."""
        if self.stack:
            line = self.stack.pop()
        else:
            line = next(self.f)
        self.lineno += 1
        return line

    def error(self, msg: str):
        """Raise an error while reading a file.

        Parameters
        ----------
        msg
            Message to raise alongside filename and line number.

        """
        raise FileFormatError("{}:{} {}".format(self.filename, self.lineno, msg))

    def warn(self, msg: str):
        """Raise a warning while reading a file.

        Parameters
        ----------
        msg
            Message to raise alongside filename and line number.

        """
        warnings.warn("{}:{} {}".format(self.filename, self.lineno, msg),
                      FileFormatWarning, 2)

    def back(self, line):
        """Go one line back and decrease the lineno attribute by one."""
        self.stack.append(line)
        self.lineno -= 1


@attr.s(auto_attribs=True, slots=True,
        on_setattr=[attr.setters.validate, attr.setters.convert])
class Cube:
    """The volumetric data from a cube (or similar) file.

    Attributes
    ----------
    origin
        A 3D vector with the origin of the axes frame.
    axes
        A (3, 3) array where each row represents the spacing between two
        neighboring grid points along the first, second and third axis,
        respectively.
    data
        A (K, L, M) array of data on a uniform grid

    """

    origin: np.ndarray = attr.ib(validator=validate_shape(3))
    axes: np.ndarray = attr.ib(validator=validate_shape(3, 3))
    data: np.ndarray = attr.ib(validator=validate_shape(None, None, None))

    @property
    def shape(self):
        """Shape of the rectangular grid."""  # noqa: D401
        return self.data.shape


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


def derive_naturals(dm: np.ndarray, overlap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Derive natural orbitals from a given density matrix.

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

    """
    # Transform density matrix to Fock-like form
    sds = np.dot(overlap.T, np.dot(dm, overlap))
    # Diagonalize and compute eigenvalues
    evals, evecs = eigh(sds, overlap)
    coeffs = np.zeros_like(overlap)
    coeffs = evecs[:, :coeffs.shape[1]]
    occs = evals
    return coeffs, occs


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
