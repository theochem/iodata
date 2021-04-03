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
from typing import List, Dict, Tuple, Union

import attr
import numpy as np

from .attrutils import validate_shape


__all__ = ['angmom_sti', 'angmom_its', 'Shell', 'MolecularBasis',
           'convert_convention_shell', 'convert_conventions',
           'iter_cart_alphabet', 'HORTON2_CONVENTIONS', 'CCA_CONVENTIONS']

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


@attr.s(auto_attribs=True, slots=True,
        on_setattr=[attr.setters.validate, attr.setters.convert])
class Shell:
    """A shell in a molecular basis representing (generalized) contractions with the same exponents.

    Attributes
    ----------
    icenter
        An integer index specifying the row in the atcoords array of IOData object.
    angmoms
        An integer array of angular momentum quantum numbers, non-negative, with
        shape (ncon,).
    kinds
        List of strings describing the kind of contractions: 'c' for Cartesian
        and 'p' for pure. Pure functions are only allowed for angmom>1.
        The length equals the number of contractions: len(angmoms)=ncon.
    exponents
        The array containing the exponents of the primitives, with shape (nprim,).
    coeffs
        The array containing the coefficients of the normalized primitives in each contraction;
        shape = (nprim, ncon).
        These coefficients assume that the primitives are L2 (orbitals) or L1
        (densities) normalized, but contractions are not necessarily normalized.
        (This depends on the code which generated the contractions.)

    """

    icenter: int
    angmoms: List[int] = attr.ib(validator=validate_shape(("coeffs", 1)))
    kinds: List[str] = attr.ib(validator=validate_shape(("coeffs", 1)))
    exponents: np.ndarray = attr.ib(validator=validate_shape(("coeffs", 0)))
    coeffs: np.ndarray = attr.ib(validator=validate_shape(("exponents", 0), ("kinds", 0)))

    @property
    def nbasis(self) -> int:  # noqa: D401
        """Number of basis functions (e.g. 3 for a P shell and 4 for an SP shell)."""
        result = 0
        for angmom, kind in zip(self.angmoms, self.kinds):
            if kind == 'c':  # Cartesian
                result += ((angmom + 1) * (angmom + 2)) // 2
            elif kind == 'p' and angmom >= 2:
                result += 2 * angmom + 1
            else:
                raise TypeError('Unknown shell kind \'{}\'; expected \'c\' or \'p\'.'.format(kind))
        return result

    @property
    def nprim(self) -> int:  # noqa: D401
        """Number of primitives, also known as the contraction length."""
        return len(self.exponents)

    @property
    def ncon(self) -> int:  # noqa: D401
        """Number of contractions. This is usually 1; e.g., it would be 2 for an SP shell."""
        return len(self.angmoms)


@attr.s(auto_attribs=True, slots=True,
        on_setattr=[attr.setters.validate, attr.setters.convert])
class MolecularBasis:
    """A complete molecular orbital or density basis set.

    Attributes
    ----------
    shells
        A list of objects of type Shell which can support generalized contractions.
    conventions
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

    primitive_normalization
        The normalization convention of primitives, which can be 'L2' (orbitals) or 'L1'
        (densities) normalized.

    """

    shells: List[Shell]
    conventions: Dict[str, str]
    primitive_normalization: str

    @property
    def nbasis(self) -> int:  # noqa: D401
        """Number of basis functions."""
        return sum(shell.nbasis for shell in self.shells)

    def get_segmented(self):
        """Unroll generalized contractions."""
        shells = []
        for shell in self.shells:
            for angmom, kind, coeffs in zip(shell.angmoms, shell.kinds, shell.coeffs.T):
                shells.append(Shell(shell.icenter, [angmom], [kind],
                                    shell.exponents, coeffs.reshape(-1, 1)))
        # pylint: disable=no-member
        return attr.evolve(self, shells=shells)


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


def get_default_conventions() -> Tuple[Dict, Dict]:
    """Produce conventions dictionaries compatible with HORTON2 and CCA.

    Do not change this! Both conventions are also used by several file formats
    from other QC codes.

    Common Component Architecture (CCA) conventions are defined in appendix B of
    the following article:

    Kenny, J. P.; Janssen, C. L.; Valeev, E. F.; Windus, T. L. Components for
    Integral Evaluation in Quantum Chemistry: Components for Integral Evaluation
    in Quantum Chemistry. J. Comput. Chem. 2008, 29 (4), 562–577.
    https://doi.org/10.1002/jcc.20815.

    The ordering of the spherical harmonics within one shell is rather vague
    in appendix B and a more precise description is given on the LibInt Wiki:

    https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API

    Returns
    -------
    horton2_conventions
        A conventions dictionary for HORTON2, of which parts are used by various
        file formats.
    cca_conventions
        A conventions dictionary compatible with the Common Component
        Architecture (CCA).

    """
    horton2 = {(0, 'c'): ['1']}
    cca = horton2.copy()
    for angmom in range(1, 25):
        conv_cart = list('x' * nx + 'y' * ny + 'z' * nz
                         for nx, ny, nz in iter_cart_alphabet(angmom))
        key = (angmom, 'c')
        horton2[key] = conv_cart
        cca[key] = conv_cart
        if angmom > 1:
            conv_pure = ['c0']
            for absm in range(1, angmom + 1):
                conv_pure.append('c{}'.format(absm))
                conv_pure.append('s{}'.format(absm))
            key = (angmom, 'p')
            horton2[key] = conv_pure
            cca[key] = conv_pure[:1:-2] + conv_pure[:1] + conv_pure[1::2]
    return horton2, cca


HORTON2_CONVENTIONS, CCA_CONVENTIONS = get_default_conventions()
