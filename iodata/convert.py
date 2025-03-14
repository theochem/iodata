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
"""Functions to convert IOData-related objects.

This module includes the lower-level implementations to the functions in the prepare module.
The functions in this module can be used without writing data to files.
Most of them are used by the prepare module.
"""

import attrs
import numpy as np
from numpy.typing import NDArray

from .basis import MolecularBasis, Shell
from .orbitals import MolecularOrbitals

__all__ = (
    "CCA_CONVENTIONS",
    "HORTON2_CONVENTIONS",
    "convert_conventions",
    "convert_to_segmented",
    "convert_to_unrestricted",
    "iter_cart_alphabet",
)


def _convert_convention_shell(
    conv1: list[str], conv2: list[str], reverse=False
) -> tuple[NDArray[int], NDArray[int]]:
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
        raise ValueError("conv1 and conv2 must contain the same number of elements.")
    # Get signs from both
    signs1 = [1 - 2 * el1.startswith("-") for el1 in conv1]
    signs2 = [1 - 2 * el2.startswith("-") for el2 in conv2]
    # Strip signs from both
    conv1 = [el1.lstrip("-") for el1 in conv1]
    conv2 = [el2.lstrip("-") for el2 in conv2]
    if len(conv1) != len(set(conv1)):
        raise ValueError("Argument conv1 contains duplicates.")
    if len(conv2) != len(set(conv2)):
        raise ValueError("Argument conv2 contains duplicates.")
    if set(conv1) != set(conv2):
        raise ValueError(
            "Without the minus signs, conv1 and conv2 must contain "
            f"the same elements. Got {conv1} and {conv2}."
        )
    # Get the permutation
    if reverse:
        permutation = [conv2.index(el1) for el1 in conv1]
        signs = [signs2[i] * sign1 for i, sign1 in zip(permutation, signs1)]
    else:
        permutation = [conv1.index(el2) for el2 in conv2]
        signs = [signs1[i] * sign2 for i, sign2 in zip(permutation, signs2)]
    return permutation, signs


def convert_conventions(
    molbasis: MolecularBasis, new_conventions: dict[str, list[str]], reverse=False
) -> tuple[NDArray[int], NDArray[int]]:
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
            shell_permutation, shell_signs = _convert_convention_shell(conv1, conv2, reverse)
            offset = len(permutation)
            permutation.extend(i + offset for i in shell_permutation)
            signs.extend(shell_signs)
    return np.array(permutation), np.array(signs)


def iter_cart_alphabet(n: int) -> NDArray[int]:
    """Loop over powers of Cartesian basis functions in alphabetical order.

    See https://theochem.github.io/horton/2.1.1/tech_ref_gaussian_basis.html
    for details.

    Parameters
    ----------
    n
        The angular momentum, i.e. sum of Cartesian powers in this case.

    """
    if n < 0:
        raise ValueError(f"The angular momentum cannot be negative. Got {n}")
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)


def _get_default_conventions() -> tuple[dict, dict]:
    """Produce conventions dictionaries compatible with HORTON2 and CCA.

    Do not change this! Both conventions are also used by several file formats
    from other QC codes.

    Common Component Architecture (CCA) conventions are defined in appendix B of
    the following article:

    Kenny, J. P.; Janssen, C. L.; Valeev, E. F.; Windus, T. L. Components for
    Integral Evaluation in Quantum Chemistry: Components for Integral Evaluation
    in Quantum Chemistry. J. Comput. Chem. 2008, 29 (4), 562-577.
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
    horton2 = {(0, "c"): ["1"]}
    cca = horton2.copy()
    for angmom in range(1, 25):
        conv_cart = ["x" * nx + "y" * ny + "z" * nz for nx, ny, nz in iter_cart_alphabet(angmom)]
        key = (angmom, "c")
        horton2[key] = conv_cart
        cca[key] = conv_cart
        if angmom > 1:
            conv_pure = ["c0"]
            for absm in range(1, angmom + 1):
                conv_pure.append(f"c{absm}")
                conv_pure.append(f"s{absm}")
            key = (angmom, "p")
            horton2[key] = conv_pure
            cca[key] = conv_pure[:1:-2] + conv_pure[:1] + conv_pure[1::2]
    return horton2, cca


HORTON2_CONVENTIONS, CCA_CONVENTIONS = _get_default_conventions()


def convert_to_unrestricted(mo: MolecularOrbitals) -> MolecularOrbitals:
    """Convert orbitals to ``kind="unrestricted"``.

    Parameters
    ----------
    mo
        Restricted molecular orbitals to be converted.

    Returns
    -------
    new_mo
        The given object if the orbitals were already unrestricted or a new unrestricted copy.

    Raises
    ------
    ValueError
        When the given orbitals are generalized.
    """
    if mo.kind == "generalized":
        raise ValueError("Generalized orbitals cannot be converted to unrestricted.")
    if mo.kind == "unrestricted":
        return mo
    return MolecularOrbitals(
        "unrestricted",
        mo.norba,
        mo.norbb,
        None if mo.occs is None else np.concatenate([mo.occsa, mo.occsb]),
        None if mo.coeffs is None else np.concatenate([mo.coeffs, mo.coeffs], axis=1),
        None if mo.energies is None else np.concatenate([mo.energies, mo.energies]),
        None if mo.irreps is None else np.concatenate([mo.irreps, mo.irreps]),
    )


def convert_to_segmented(obasis: MolecularBasis, keep_sp: bool = False) -> MolecularBasis:
    """Convert basis with generalized contractions to one with only single contractions.

    Parameters
    ----------
    obasis
        The basis set to convert.
    keep_sp
        If True, SP shells are not split up.
        This can be useful for file formats only support
        SP-generalized contractions and no other ones.

    Returns
    -------
    A new ``MolecularBasis`` instance with separate single contractions.
    """
    shells = []
    for shell in obasis.shells:
        if (shell.ncon == 1) or (keep_sp and shell.ncon == 2 and (shell.angmoms == [0, 1]).all()):
            shells.append(shell)
        else:
            for angmom, kind, coeffs in zip(shell.angmoms, shell.kinds, shell.coeffs.T):
                shells.append(
                    Shell(shell.icenter, [angmom], [kind], shell.exponents, coeffs.reshape(-1, 1))
                )
    return attrs.evolve(obasis, shells=shells)
