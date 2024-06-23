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

from pathlib import Path
from typing import Optional, TextIO, Union

import attrs
import numpy as np
import scipy.constants as spc
from numpy.typing import NDArray
from scipy.linalg import eigh

from .attrutils import validate_shape

__all__ = (
    "LineIterator",
    "FileFormatError",
    "LoadError",
    "DumpError",
    "PrepareDumpError",
    "WriteInputError",
    "LoadWarning",
    "DumpWarning",
    "Cube",
    "set_four_index_element",
    "volume",
    "derive_naturals",
    "check_dm",
    "strtobool",
)


# The unit conversion factors below can be used as follows:
# - Conversion to atomic units: distance = 5*angstrom
# - Conversion from atomic units: print(distance/angstrom)
angstrom: float = spc.angstrom / spc.value("atomic unit of length")
electronvolt: float = 1 / spc.value("hartree-electron volt relationship")
# Unit conversion for Gromacs gro files
meter: float = 1 / spc.value("Bohr radius")
nanometer: float = 1e-9 * meter
second: float = 1 / spc.value("atomic unit of time")
picosecond: float = 1e-12 * second
# atomic mass unit (not atomic unit of mass!)
amu: float = 1e-3 / (spc.value("electron mass") * spc.value("Avogadro constant"))
kcalmol: float = 1e3 * spc.calorie / spc.value("Avogadro constant") / spc.value("Hartree energy")
calmol: float = spc.calorie / spc.value("Avogadro constant") / spc.value("Hartree energy")
kjmol: float = 1e3 / spc.value("Avogadro constant") / spc.value("Hartree energy")


class LineIterator:
    """Iterator class for looping over lines and keeping track of the line number.

    Use this class as a context manager, similar to the built-in ``open`` function:

    .. code-block:: python

        with LineIterator("filename.ext") as lit:
            for line in lit:
                ...

    """

    def __init__(self, filename: str):
        """Initialize a LineIterator.

        Parameters
        ----------
        filename
            The file that will be read.

        """
        self.filename = filename
        self.fh = None
        self.lineno = 0
        self.stack = []

    def __enter__(self):
        self.fh = open(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fh.close()

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next line and increase the lineno attribute by one."""
        self.lineno += 1
        return self.stack.pop() if self.stack else next(self.fh)

    def back(self, line):
        """Go back one line in the file and decrease the lineno attribute by one."""
        self.stack.append(line)
        self.lineno -= 1


def _interpret_file_lineno(
    file: Optional[Union[str, Path, LineIterator, TextIO]] = None, lineno: Optional[int] = None
) -> tuple[Optional[str], Optional[int]]:
    """Interpret the file and lineno arguments given to Error and Warning constructors.

    Parameters
    ----------
    file
        Object to deduce the filename (and optionally line number) from.
    lineno
        Line number, if known and not (correctly) included in the file object.

    Returns
    -------
    filename
        The filename associated with the file object.
    lineno
        The line number.
    """
    if isinstance(file, str):
        return file, lineno
    if isinstance(file, Path):
        return str(file), lineno
    if isinstance(file, LineIterator):
        if lineno is None:
            lineno = file.lineno
        return file.filename, lineno
    if isinstance(file, TextIO):
        return file.name, lineno
    if file is None:
        if lineno is not None:
            raise TypeError("A line number without a file is not supported.")
        return None, None
    raise TypeError(f"Types of file and lineno are not supported: {file}, {lineno}")


def _format_file_message(message: str, filename: Optional[str], lineno: Optional[int]) -> str:
    """Format the message of an exception.

    Parameters
    ----------
    message
        The actual error or warning message, without filename or line number info.
    filename
        The filename to which the error or warning is related.
    lineno
        The line number associated with the error or warning.

    Returns
    -------
    full_message
        The error message formated with filename and line number info.
    """
    if filename is None:
        return message
    if lineno is None:
        return f"{message} ({filename})"
    return f"{message} ({filename}:{lineno})"


class BaseFileError(Exception):
    """Base class for all errors related to loading or dumping files."""

    def __init__(
        self,
        message,
        file: Optional[Union[str, Path, LineIterator, TextIO]] = None,
        lineno: Optional[int] = None,
    ):
        super().__init__(message)
        self.filename, self.lineno = _interpret_file_lineno(file, lineno)

    def __str__(self):
        return _format_file_message(super().__str__(), self.filename, self.lineno)


class FileFormatError(BaseFileError):
    """Raise when a file or input format cannot be identified."""


class LoadError(BaseFileError):
    """Raised when an error is encountered while loading from a file."""


class DumpError(BaseFileError):
    """Raised when an error is encountered while dumping to a file."""


class PrepareDumpError(BaseFileError):
    """Raised when an IOData object is incompatible with a format before dumping to a file."""


class WriteInputError(BaseFileError):
    """Raised when an error is encountered while writing an input file."""


class BaseFileWarning(Warning):
    """Base class for all warnings related to loading or dumping files."""

    def __init__(
        self,
        message,
        file: Optional[Union[str, Path, LineIterator, TextIO]] = None,
        lineno: Optional[int] = None,
    ):
        filename, lineno = _interpret_file_lineno(file, lineno)
        super().__init__(_format_file_message(message, filename, lineno))


class LoadWarning(BaseFileWarning):
    """Raised when incorrect content is encountered and fixed when loading from a file."""


class DumpWarning(BaseFileWarning):
    """Raised when an IOData object is made compatible with a format when dumping to a file."""


@attrs.define
class Cube:
    """The volumetric data from a cube (or similar) file."""

    origin: NDArray[float] = attrs.field(validator=validate_shape(3))
    """A 3D vector with the origin of the axes frame."""

    axes: NDArray[float] = attrs.field(validator=validate_shape(3, 3))
    """
    A (3, 3) array where each row represents the spacing between two neighboring grid points
    along the first, second and third axis, respectively.
    """

    data: NDArray[float] = attrs.field(validator=validate_shape(None, None, None))
    """A (K, L, M) array of data on a uniform grid"""

    @property
    def shape(self):
        """Shape of the rectangular grid."""
        return self.data.shape


def set_four_index_element(
    four_index_object: NDArray[float], i0: int, i1: int, i2: int, i3: int, value: float
):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation.

    Parameters
    ----------
    four_index_object
        The four-index object. It will be written to.
        shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
    i0, i1, i2, i3
        The indices to assign to.
    value
        The value of the matrix element to store.

    """
    four_index_object[i0, i1, i2, i3] = value
    four_index_object[i1, i0, i3, i2] = value
    four_index_object[i2, i1, i0, i3] = value
    four_index_object[i0, i3, i2, i1] = value
    four_index_object[i2, i3, i0, i1] = value
    four_index_object[i3, i2, i1, i0] = value
    four_index_object[i1, i2, i3, i0] = value
    four_index_object[i3, i0, i1, i2] = value


def volume(cellvecs: NDArray[float]) -> float:
    """Calculate the (generalized) cell volume.

    Parameters
    ----------
    cellvecs
        A numpy matrix of shape (x,3) where x is in {1,2,3}.
        Each row is one cellvector.

    Returns
    -------
    In case of 3D, the cell volume.
    In case of 2D, the cell area.
    In case of 1D, the cell length.

    """
    nvecs = cellvecs.shape[0]
    if len(cellvecs.shape) == 1 or nvecs == 1:
        return np.linalg.norm(cellvecs)
    if nvecs == 2:
        return np.linalg.norm(np.cross(cellvecs[0], cellvecs[1]))
    if nvecs == 3:
        return np.linalg.det(cellvecs)
    raise ValueError("Argument cellvecs should be of shape (x, 3), where x is in {1, 2, 3}")


def derive_naturals(
    dm: NDArray[float], overlap: NDArray[float]
) -> tuple[NDArray[float], NDArray[float]]:
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
    coeffs = evecs[:, : coeffs.shape[1]]
    occs = evals
    return coeffs, occs


def check_dm(dm: NDArray[float], overlap: NDArray[float], eps: float = 1e-4, occ_max: float = 1.0):
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
        raise ValueError(
            "The density matrix has eigenvalues considerably smaller than "
            f"zero. error={occupations.min():e}"
        )
    if occupations.max() > occ_max + eps:
        raise ValueError(
            "The density matrix has eigenvalues considerably larger than "
            "max. error=%e" % (occupations.max() - 1)
        )


STRTOBOOL = {
    "y": True,
    "yes": True,
    "t": True,
    "true": True,
    "on": True,
    "1": True,
    "n": False,
    "no": False,
    "f": False,
    "false": False,
    "off": False,
    "0": False,
}


def strtobool(value: str) -> bool:
    """Interpret string as a boolean."""
    result = STRTOBOOL.get(value.lower())
    if result is None:
        raise ValueError(f"'{value}' cannot be converted to boolean")
    return result
