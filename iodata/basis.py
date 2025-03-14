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
"""Utility functions for working with basis sets.

Notes
-----
Basis set conventions and terminology are documented in :ref:`basis_conventions`.

"""

from functools import wraps
from numbers import Integral
from typing import Union

import attrs
from numpy.typing import NDArray

from .attrutils import convert_array_to, validate_shape

__all__ = ("MolecularBasis", "Shell", "angmom_its", "angmom_sti")

ANGMOM_CHARS = "spdfghiklmnoqrtuvwxyzabce"


def _alsolist(f):
    """Wrap a function to accepts also list as first argument and then return list."""

    @wraps(f)
    def wrapper(firsts, *args, **kwargs):
        if isinstance(firsts, (Integral, str)):
            return f(firsts, *args, **kwargs)
        return [f(first, *args, **kwargs) for first in firsts]

    return wrapper


@_alsolist
def angmom_sti(char: Union[str, list[str]]) -> Union[int, list[int]]:
    """Convert an angular momentum from string to integer format.

    Parameters
    ----------
    char
        Character representation of angular momentum, (s, p, d, ...)

    Returns
    -------
    An integer representation of the angular momentum.
    If a list of str char is given, a list of integers in returned.

    """
    return ANGMOM_CHARS.index(char.lower())


@_alsolist
def angmom_its(angmom: Union[int, list[int]]) -> Union[str, list[str]]:
    """Convert an angular momentum from integer to string representation.

    Parameters
    ----------
    angmom
        The integer representation of the angular momentum.

    Returns
    -------
    The string representation of the angular momentum.
    If a list of integer angmom is given, a list of str is returned.

    """
    if angmom < 0:
        raise ValueError("Angmom cannot be negative.")
    return ANGMOM_CHARS[angmom]


@attrs.define
class Shell:
    """A shell of (generalized) contracted Gaussian basis functions with common primitive exponents.

    The basis functions in a ``Shell`` instance are centered on a single atomic nucleus.
    However, an atom can be associated with multiple ``Shell`` instances.

    Notes
    -----
    Basis set conventions and terminology are documented in :ref:`basis_conventions`.
    """

    icenter: int = attrs.field()
    """An integer index specifying the row in the atcoords array of IOData object."""

    angmoms: NDArray[int] = attrs.field(
        converter=convert_array_to(int),
        validator=validate_shape(("coeffs", 1)),
    )
    """An integer array of angular momentum quantum numbers, non-negative, with shape (ncon,).

    In the case of ordinary (not generalized) contractions, this array contains one element.

    For generalized contractions, this array contains multiple elements.
    The same angular momentum may or may not appear multiple times.

    The most common form of generalized contraction is the SP shell,
    e.g. as found in the Pople basis sets (6-31G and others),
    in which case this array is ``[0, 1]``.

    Other forms of generalized contractions exist,
    but only some quantum chemistry codes have efficient implementations for them.
    For example, the ANO-RCC basis for carbon has 8 S-type basis functions,
    all different linear combinations of the same 14 Gaussian primitives.
    In this case, this array is ``[0, 0, 0, 0, 0, 0, 0, 0]``.
    """

    kinds: NDArray[str] = attrs.field(
        converter=convert_array_to(str),
        validator=validate_shape(("coeffs", 1)),
    )
    """
    Array of strings describing the kind of contractions:
    ``'c'`` for Cartesian and ``'p'`` for pure.
    Pure functions are only allowed for ``angmom > 1``.
    The length equals the number of contractions (``ncon = len(kinds)``).
    """

    exponents: NDArray[float] = attrs.field(
        converter=convert_array_to(float),
        validator=validate_shape(("coeffs", 0)),
    )
    """The array containing the exponents of the primitives, with shape (nexp,)."""

    coeffs: NDArray[float] = attrs.field(
        converter=convert_array_to(float),
        validator=validate_shape(("exponents", 0), ("kinds", 0)),
    )
    """
    The array containing the coefficients of the normalized primitives in each contraction;
    shape = (nexp, ncon).
    Each column represents one linear combination of basis functions.
    These coefficients assume that the primitives are L2 (orbitals) or L1
    (densities) normalized, but contractions are not necessarily normalized.
    (This depends on the code that generated the contractions.)
    """

    @property
    def nbasis(self) -> int:
        """Number of basis functions (e.g. 3 for a P shell and 4 for an SP shell)."""
        result = 0
        for angmom, kind in zip(self.angmoms, self.kinds):
            if kind == "c":  # Cartesian
                result += ((angmom + 1) * (angmom + 2)) // 2
            elif kind == "p" and angmom >= 2:
                result += 2 * angmom + 1
            else:
                raise TypeError(f"Unknown shell kind '{kind}'; expected 'c' or 'p'.")
        return result

    @property
    def nexp(self) -> int:
        """Number of exponents in the contracted shell, also known as the contraction length."""
        return len(self.exponents)

    @property
    def ncon(self) -> int:
        """Number of generalized contractions.

        This is the number of different linear combinations of Gaussian basis functions with
        the same set of exponents.

        This is usually 1; e.g., it would be 2 for an SP shell.
        """
        return len(self.angmoms)


@attrs.define
class MolecularBasis:
    """A complete molecular orbital or density basis set, a collection of contracted shells."""

    shells: list[Shell] = attrs.field()
    """A list of objects of type Shell which can support generalized contractions."""

    conventions: dict[tuple[int, str], list[str]] = attrs.field()
    """
    A dictionary specifying the ordered basis functions for a given angular momentum and kind.
    The key is a tuple of angular momentum integer and kind character ('c' for Cartesian
    and 'p' for pure/spherical) and the value is a list of basis function strings.
    For example,

    .. code-block:: python

        {
            ### Conventions for Cartesian functions
            # E.g., alphabetically ordered Cartesian functions.
            (0, 'c'): ['1'],
            (1, 'c'): ['x', 'y', 'z'],
            (2, 'c'): ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'],
            ### Conventions for pure functions.
            # The notation is referring to real solid spherical harmonics.
            # See https://en.wikipedia.org/wiki/Solid_harmonics#Real_form
            # 'c{m}' = solid harmonic containing cos(m phi)
            # 's{m}' = solid harmonic containing sin(m phi)
            # where m is the magnetic quantum number and phi is the
            # azimuthal angle.
            # For example, wikipedia-ordered real spherical harmonics,
            # see https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
            (2, 'p'): ['s2', 's1', 'c0', 'c1', 'c2'],
            # Different quantum-chemistry codes may use incompatible
            # orderings and sign conventions. E.g. Molden files written
            # by ORCA use the following convention for pure f functions:
            (3, 'p'): ['c0', 'c1', 's1', 'c2', 's2', '-c3', '-s3'],
            # Note that the minus sign in the last two basis functions
            # denotes that the signs of these harmonics have been changed.
        }

    The basis function strings in the conventions dictionary are documented
    in :ref:`basis_conventions`.
    """

    primitive_normalization: str = attrs.field()
    """
    The normalization convention of primitives,
    which can be 'L2' (orbitals) or 'L1' (densities) normalized.
    """

    @property
    def nbasis(self) -> int:
        """Number of basis functions."""
        return sum(shell.nbasis for shell in self.shells)
