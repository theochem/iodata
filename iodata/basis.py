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
"""Utility functions for working with basis sets."""

from functools import wraps
from numbers import Integral
from typing import List, Dict, NamedTuple, Tuple, Union

import numpy as np

__all__ = ['angmom_sti', 'angmom_its', 'Shell', 'MolecularBasis',
           'convert_convention_shell', 'convert_conventions',
           'iter_cart_alphabet', 'HORTON2_CONVENTIONS', 'PSI4_CONVENTIONS',
           'GBASIS_CONVENTIONS']

ANGMOM_CHARS = 'spdfghiklmnoqrtuvwxyzabce'


def _alsolist(f):
    """Wrap a function to accepts also list as first argument and then return list."""
    @wraps(f)
    def wrapper(firsts, *args, **kwargs):
        if isinstance(firsts, (Integral, str)):
            return f(firsts, *args, **kwargs)
        return [f(first, *args, **kwargs) for first in firsts]
    return wrapper


@_alsolist
def angmom_sti(char: Union[str, List[str]]) -> Union[int, List[int]]:
    """Convert an angular momentum from string to integer format.

    Parameters
    ----------
    char
        Character representation of angular momentum, (s, p, d, ...)

    Returns
    -------
    angmom
        An integer representation of the angular momentum. If a list of str
        char is given, a list of integers in returned.

    """
    return ANGMOM_CHARS.index(char.lower())


@_alsolist
def angmom_its(angmom: Union[int, List[int]]) -> Union[str, List[str]]:
    """Convert an angular momentum from integer to string representation.

    Parameters
    ----------
    angmom
        The integer representation of the angular momentum.

    Returns
    -------
    char
        The string representation of the angular momentum. If a list of integer
        angmom is given, a list of str is returned.

    """
    if angmom < 0:
        raise ValueError("Angmom cannot be negative.")
    return ANGMOM_CHARS[angmom]


class Shell(NamedTuple):
    """Describe a single shell in a molecular basis set.

    Attributes
    ----------
    icenter
        An integer referring to a row in the array atcoords in an IOData object.
    angmoms
        An integer array of angular momentum quantum numbers, non-negative, with
        shape (ncon,).
    kinds
        List of strings describing the kind of contraction: 'c' for Cartesian
        and 'p' for pure. Pure functions are only allowed for angmom>1.
        The length equals the number of contractions: len(angmoms)=ncon.
    exponents
        an array of exponents of primitives, with shape (nprim,).
    coeffs
        an array with contraction coefficients, with shape (nprim, ncon). These
        coefficients assume that the primitives are L2 (orbitals) or L1
        (densities) normalized, but contractions are not necessarily normalized.
        (This depends on the code which generated the contractions.)

    """

    icenter: int
    angmoms: List[int]
    kinds: List[str]
    exponents: np.ndarray
    coeffs: np.ndarray

    @property
    def nbasis(self) -> int:
        """Return the number of basis functions."""
        result = 0
        for angmom, kind in zip(self.angmoms, self.kinds):
            if kind == 'c':  # Cartesian
                result += ((angmom + 1) * (angmom + 2)) // 2
            elif kind == 'p' and angmom >= 2:
                result += 2 * angmom + 1
            else:
                raise TypeError('Unknown shell kind \'{}\'.'.format(kind))
        return result

    @property
    def nprim(self) -> int:
        """Return the number of primitives. Also known as the contraction length."""
        return len(self.exponents)

    @property
    def ncon(self) -> int:
        """Return the number of contractions."""
        return len(self.angmoms)


class MolecularBasis(NamedTuple):
    """Describe a complete molecular orbital or density basis set.

    Attributes
    ----------
    shells
        a list of objects of the type Shell
    conventions
        a dictionary with as key a typle of angular momentum integer and kind
        character, and as value a list of basis function strings, e.g.

        .. code-block:: python

            {
                (0, 'c'): ['1'],
                (1, 'c'): ['x', 'y', 'z'],
                # alphabetically ordered Cartesian functions
                (2, 'c'): ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'],
                # or Wikipedia-ordered real solid spherical harmonics
                # c = cosine-like
                # s = sine-like
                (2, 'p'): ['dc2', 'dc1', 'dc0', '-ds1', '-ds2'],
                ...
            }

    primitive_normalization
        Either 'L1' or 'L2'.

    """

    shells: tuple
    conventions: Dict[str, str]
    primitive_normalization: str

    @property
    def nbasis(self) -> int:
        """Return the number of basis functions."""
        return sum(shell.nbasis for shell in self.shells)

    def get_segmented(self):
        """Unroll generalized contractions."""
        shells = []
        for shell in self.shells:
            for angmom, kind, coeffs in zip(shell.angmoms, shell.kinds, shell.coeffs.T):
                shells.append(Shell(shell.icenter, [angmom], [kind],
                                    shell.exponents, coeffs.reshape(-1, 1)))
        # pylint: disable=no-member
        return self._replace(shells=shells)


def convert_convention_shell(conv1: List[str], conv2: List[str], reverse=False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Return a permutation vector and sign changes to convert from 1 to 2.

    The transformation from convention 1 to convention 2 can be done applying
    the results of this function as follows:

    .. code-block:: python

        vector2 = vector1[permutation]*signs

    When using the option ``reverse=True``, one can use the results to convert
    in the opposite sense:

    .. code-block:: python

        vector1 = vector2[permutation]*signs

    Parameters
    ----------
    conv1, conv2
        Two lists, with the same strings (in different order), where each string
        may be prefixed with a '-'.
    reverse:
        When, true the conversion from 2 to 1 is returned.

    Returns
    -------
    permutation
        An integer array that permutes basis function from 1 to 2.
    signs
        Sign changes when going from 1 to 2, must be applied after permutation

    """
    if len(conv1) != len(conv2):
        raise TypeError('conv1 and conv2 must contain the same number of elements.')
    # Get signs from both
    signs1 = [1 - 2 * el1.startswith('-') for el1 in conv1]
    signs2 = [1 - 2 * el2.startswith('-') for el2 in conv2]
    # Strip signs from both
    conv1 = [el1.lstrip('-') for el1 in conv1]
    conv2 = [el2.lstrip('-') for el2 in conv2]
    if len(conv1) != len(set(conv1)):
        raise TypeError('Argument conv1 contains duplicates.')
    if len(conv2) != len(set(conv2)):
        raise TypeError('Argument conv2 contains duplicates.')
    if set(conv1) != set(conv2):
        raise TypeError('Without the minus signs, conv1 and conv2 must contain '
                        'the same elements. Got {} and {}.'.format(conv1, conv2))
    # Get the permutation
    if reverse:
        permutation = [conv2.index(el1) for el1 in conv1]
        signs = [signs2[i] * sign1 for i, sign1 in zip(permutation, signs1)]
    else:
        permutation = [conv1.index(el2) for el2 in conv2]
        signs = [signs1[i] * sign2 for i, sign2 in zip(permutation, signs2)]
    return permutation, signs


def convert_conventions(molbasis: MolecularBasis, new_conventions: Dict[str, List[str]],
                        reverse=False) -> Tuple[np.ndarray, np.ndarray]:
    """Return a permutation vector and sign changes to convert from 1 to 2.

    The transformation from molbasis.convention to the new convention can be done
    applying the results of this function as follows:

    .. code-block:: python

        vector2 = vector1[permutation]*signs

    When using the option ``reverse=True``, one can use the results to convert
    in the opposite sense:

    .. code-block:: python

        vector1 = vector2[permutation]*signs


    Parameters
    ----------
    molbasis
        The description of a molecular basis set.
    new_conventions
        The new conventions for ordering and signs, to which data for the
        orbital basis needs to be converted.
    reverse:
        When, true the conversion from 2 to 1 is returned.

    Returns
    -------
    permutation
        An integer array that permutes basis function from 1 to 2.
    signs
        Sign changes when going from 1 to 2, must be applied after permutation

    """
    permutation = []
    signs = []
    for shell in molbasis.shells:
        for angmom, kind in zip(shell.angmoms, shell.kinds):
            key = (angmom, kind)
            conv1 = molbasis.conventions[key]
            conv2 = new_conventions[key]
            shell_permutation, shell_signs = convert_convention_shell(conv1, conv2, reverse)
            offset = len(permutation)
            for i in shell_permutation:
                permutation.append(i + offset)
            signs.extend(shell_signs)
    return np.array(permutation), np.array(signs)


def iter_cart_alphabet(n: int) -> np.ndarray:
    """Loop over powers of Cartesian basis functions in alphabetical order.

    See https://theochem.github.io/horton/2.1.1/tech_ref_gaussian_basis.html
    for details.

    Parameters
    ----------
    n
        The angular momentum, i.e. sum of Cartesian powers in this case.

    """
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)


def get_default_conventions():
    """Produce a conventions dictionary compatible with HORTON2.

    Do not change this!!! This is also used by several file formats from other
    QC codes who happen to follow the same conventions.
    """
    horton2 = {(0, 'c'): ['1']}
    psi4 = horton2.copy()
    gbasis = horton2.copy()
    for angmom in range(1, 25):
        conv_cart = list('x' * nx + 'y' * ny + 'z' * nz
                         for nx, ny, nz in iter_cart_alphabet(angmom))
        key = (angmom, 'c')
        horton2[key] = conv_cart
        psi4[key] = conv_cart
        gbasis[key] = conv_cart[::-1]
        if angmom > 1:
            char = angmom_its(angmom)
            conv_pure = [char + 'c0']
            for absm in range(1, angmom + 1):
                conv_pure.append('{}c{}'.format(char, absm))
                conv_pure.append('{}s{}'.format(char, absm))
            key = (angmom, 'p')
            horton2[key] = conv_pure
            psi4[key] = conv_pure[:1:-2] + conv_pure[:1] + conv_pure[1::2]
            gbasis[key] = psi4[key]
    return horton2, psi4, gbasis


HORTON2_CONVENTIONS, PSI4_CONVENTIONS, GBASIS_CONVENTIONS = get_default_conventions()
