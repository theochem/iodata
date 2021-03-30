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
"""CP2K ATOM output file format."""


from typing import Dict, Union, List, Tuple

import numpy as np
from scipy.special import factorialk

from ..basis import angmom_sti, MolecularBasis, Shell, HORTON2_CONVENTIONS
from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..utils import LineIterator


__all__ = []


PATTERNS = ['*.cp2k.out']


CONVENTIONS = {
    (0, 'c'): HORTON2_CONVENTIONS[(0, 'c')],
    (1, 'c'): HORTON2_CONVENTIONS[(1, 'c')],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
}


def _get_cp2k_norm_corrections(l: int, alphas: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """Compute the corrections for the normalization of the basis functions.

    This correction is needed because the CP2K atom code works with a different
    type of normalization for the primitives. IOData assumes Gaussian primitives
    are always L2-normalized.

    Parameters
    ----------
    l
        The angular momentum of the (pure) basis function. (s=0, p=1, ...)
    alphas
        The exponent or exponents of the Gaussian primitives for which the correction
        is to be computed.

    Returns
    -------
    corrections
        The scale factor for the expansion coefficients of the wavefunction in
        terms of primitive Gaussians. The inverse of this correction can be
        applied to the contraction coefficients.

    """
    expzet = 0.25 * (2 * l + 3)
    prefac = np.sqrt(np.sqrt(np.pi) / 2.0 ** (l + 2) * factorialk(2 * l + 1, 2))
    zeta = 2.0 * alphas
    return zeta ** expzet / prefac


def _read_cp2k_contracted_obasis(lit: LineIterator) -> MolecularBasis:
    """Read a contracted basis set from an open CP2K ATOM output file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    obasis
        The orbital basis.

    """
    shells = []
    while True:
        line = next(lit)
        if line[3:12] != 'Functions':
            break
        angmom = angmom_sti(line[1:2])
        exponents = []
        coeffs = []
        for line in lit:
            if line[3:12] == 'Functions' or line.startswith(' *******************'):
                break
            values = [float(w) for w in line.split()]
            # one exponent per line
            exponents.append(values[0])
            # many contraction coefficients per line, all corresponding to the
            # same primitive, so rows in coeffs
            coeffs.append(
                np.array(values[1:]) / _get_cp2k_norm_corrections(angmom, values[0]))
        # Push back the last line for the next iteration
        lit.back(line)
        # Build the shell
        exponents = np.array(exponents)
        coeffs = np.array(coeffs)
        kind = 'c' if angmom < 2 else 'p'
        shells.append(Shell(0, np.array([angmom] * coeffs.shape[1]),
                            [kind] * coeffs.shape[1],
                            exponents, coeffs))

    return MolecularBasis(shells, CONVENTIONS, 'L2')


def _read_cp2k_uncontracted_obasis(lit: LineIterator) -> MolecularBasis:
    """Read an uncontracted basis set from an open CP2K ATOM output file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    obasis
        The orbital basis parameters read from the file. Can be used to
        initialize a GOBasis object.

    """
    # Load the relevant data from the file
    shells = []
    next(lit)
    while True:
        line = next(lit)
        if line[3:13] != 'Exponents:':
            break
        angmom = angmom_sti(line[1:2])
        exponents = []
        coeffs = []
        while True:
            if line.strip() == "" or "*****" in line:
                break
            words = line.split()
            # read the exponent
            exponent = float(words[-1])
            exponents.append(exponent)
            coeffs.append(1.0 / _get_cp2k_norm_corrections(angmom, exponent))
            line = next(lit)
        # Build the shell
        kind = 'c' if angmom < 2 else 'p'
        for exponent, coeff in zip(exponents, coeffs):
            shells.append(Shell(
                0, np.array([angmom]), [kind],
                np.array([exponent]), np.array([[coeff]])))

    return MolecularBasis(shells, CONVENTIONS, 'L2')


# pylint: disable=inconsistent-return-statements
def _read_cp2k_obasis(lit: LineIterator) -> dict:
    """Read atomic orbital basis set from a CP2K ATOM file object.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        The atomic orbital basis data which can be used to initialize a
        ``GOBasis`` class.

    """
    next(lit)  # Skip empty line
    line = next(lit)  # Check for contracted versus uncontracted
    if line == (' ********************** Contracted Gaussian Type Orbitals '
                '**********************\n'):
        return _read_cp2k_contracted_obasis(lit)
    if line == (' ********************* Uncontracted Gaussian Type Orbitals '
                '*********************\n'):
        return _read_cp2k_uncontracted_obasis(lit)
    lit.error('Could not find basis set in CP2K ATOM output.')


def _read_cp2k_occupations_energies(lit: LineIterator, restricted: bool) \
        -> List[Tuple[int, int, float, float]]:
    """Read orbital occupation numbers and energies from a CP2K ATOM file object.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    restricted
        If ``True`` the wave-function is considered to be restricted. If
        ``False`` the unrestricted wave-function is assumed.

    Returns
    -------
    oe_alpha, oe_beta
        A list with orbital properties. Each element is a tuple with the
        following info: (angular_momentum l, spin component: 'alpha' or
        'beta', occupation number, orbital energy).

    """
    oe_alpha = []
    oe_beta = []
    empty = 0
    while empty < 2:
        line = next(lit)
        words = line.split()
        if not words:
            empty += 1
            continue
        empty = 0
        s = int(words[0])
        l = int(words[2 - restricted])
        occ = float(words[3 - restricted])
        ener = float(words[4 - restricted])
        if restricted or words[1] == 'alpha':
            oe_alpha.append((l, s, occ, ener))
        else:
            oe_beta.append((l, s, occ, ener))
    return oe_alpha, oe_beta


def _read_cp2k_orbital_coeffs(lit: LineIterator, oe: List[Tuple[int, int, float, float]]) \
        -> Dict[Tuple[int, int], np.ndarray]:
    """Read the expansion coefficients of the orbital from an open CP2K ATOM output.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    oe
        The orbital occupation numbers and energies read with
        ``_read_cp2k_occupations_energies``.

    Returns
    -------
    result
        Key is an (l, s) pair and value is an array with orbital coefficients.

    """
    allcoeffs = {}
    next(lit)
    while len(allcoeffs) < len(oe):
        line = next(lit)
        assert line.startswith("    ORBITAL      L =")
        words = line.split()
        angmom = int(words[3])
        state = int(words[6])
        coeffs = []
        for line in lit:
            if line.strip() == "":
                break
            coeffs.append(float(line))
        allcoeffs[(angmom, state)] = np.array(coeffs)
    return allcoeffs


def _get_norb_nel(oe: List[Tuple[int, int, float, float]]) -> Tuple[int, float]:
    """Return number of orbitals and electrons.

    Parameters
    ----------
    oe
         The orbital occupation numbers and energies read with
         ``_read_cp2k_occupations_energies``.

    Returns
    -------
    Tuple
        Number of orbitals and electrons

    """
    norb = 0
    nel = 0
    for row in oe:
        norb += 2 * row[0] + 1
        nel += row[2]
    return norb, nel


def _fill_orbitals(orb_coeffs: np.ndarray,
                   orb_energies: np.ndarray,
                   orb_occupations: np.ndarray,
                   oe: List[Tuple[int, int, float, float]],
                   coeffs: Dict[Tuple[int, int], np.ndarray],
                   obasis: MolecularBasis,
                   restricted: bool):
    """Fill in orbital coefficients, energies and occupation numbers.

    The data is entered int ``orb_coeffs``, ``orb_energies``, and ``orb_occupations``.

    Parameters
    ----------
    orb_coeffs
        The orbital coefficients. Will be written to.
    orb_energies
        The orbital energies. Will be written to.
    orb_occupations
        The orbital coefficients. Will be written to.
    oe
        The orbital occupation numbers and energies read with
        ``_read_cp2k_occupations_energies``.
    coeffs
        The orbital coefficients read with ``_read_cp2k_orbital_coeffs``.
    obasis
        The molecular basis set
    restricted
        Is wavefunction restricted or unrestricted?

    """
    # Find the offsets for each angular momentum
    offset = 0
    offsets = []
    ls = np.concatenate([shell.angmoms for shell in obasis.shells])
    for l in sorted(set(ls)):
        offsets.append(offset)
        offset += (2 * l + 1) * (l == ls).sum()
    del offset

    # Fill in the coefficients
    iorb = 0
    for l, state, occ, ener in oe:
        cs = coeffs.get((l, state))
        stride = 2 * l + 1
        for im in range(2 * l + 1):
            orb_energies[iorb] = ener
            orb_occupations[iorb] = occ / float((restricted + 1) * (2 * l + 1))
            for ic, c in enumerate(cs):
                orb_coeffs[offsets[l] + stride * ic + im, iorb] = c
            iorb += 1


LOAD_ONE_NOTES = """

This function assumes that the following subsections are present in the CP2K
ATOM input file, in the section ``ATOM%PRINT``:

.. code-block:: text

  &PRINT
    &POTENTIAL
    &END POTENTIAL
    &BASIS_SET
    &END BASIS_SET
    &ORBITALS
    &END ORBITALS
  &END PRINT

"""


# pylint: disable=too-many-branches,too-many-statements
@document_load_one(
    "CP2K ATOM outupt",
    ['atcoords', 'atcorenums', 'atnums', 'energy', 'mo', 'obasis'],
    [], {}, LOAD_ONE_NOTES)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Find the element number
    atnum = None
    for line in lit:
        if line.startswith(' Atomic Energy Calculation'):
            atnum = int(line[-5:-1])
            break

    # Go to the all-electron basis set and read it.
    for line in lit:
        if line.startswith(' All Electron Basis'):
            break
    ae_obasis = _read_cp2k_obasis(lit)

    # Go to the pseudo basis set and read it.
    for line in lit:
        if line.startswith(' Pseudopotential Basis'):
            break
    pp_obasis = _read_cp2k_obasis(lit)

    # Search for (un)restricted
    restricted = None
    for line in lit:
        if line.startswith(' METHOD    |'):
            if 'U' in line:
                restricted = False
                break
            if 'R' in line:
                restricted = True
                break

    # Search for the core charge (pseudo number)
    atcorenum = None
    for line in lit:
        if line.startswith('          Core Charge'):
            atcorenum = float(line[70:])
            assert atcorenum == int(atcorenum)
            break
        if line.startswith(' Electronic structure'):
            atcorenum = float(atnum)
            break

    # Select the correct basis
    if atcorenum == atnum:
        obasis = ae_obasis
    else:
        obasis = pp_obasis

    # Search for energy
    for line in lit:
        if line.startswith(' Energy components [Hartree]           Total Energy ::'):
            energy = float(line[60:])
            break

    # Read orbital energies and occupations
    for line in lit:
        if line.startswith(' Orbital energies'):
            break
    next(lit)
    oe_alpha, oe_beta = _read_cp2k_occupations_energies(lit, restricted)

    # Read orbital expansion coefficients
    line = next(lit)
    if line not in [" Atomic orbital expansion coefficients [Alpha]\n",
                    " Atomic orbital expansion coefficients []\n"]:
        lit.error('Could not find orbital coefficients in CP2K ATOM output.')
    coeffs_alpha = _read_cp2k_orbital_coeffs(lit, oe_alpha)

    if not restricted:
        line = next(lit)
        if line != " Atomic orbital expansion coefficients [Beta]\n":
            lit.error('Could not find beta orbital coefficient in CP2K ATOM output.')
        coeffs_beta = _read_cp2k_orbital_coeffs(lit, oe_beta)

    # Turn orbital data into a MolecularOrbitals object.
    if restricted:
        norb, nel = _get_norb_nel(oe_alpha)
        assert nel % 2 == 0
        orb_alpha_coeffs = np.zeros([obasis.nbasis, norb])
        orb_alpha_energies = np.zeros(norb)
        orb_alpha_occs = np.zeros(norb)
        _fill_orbitals(orb_alpha_coeffs, orb_alpha_energies, orb_alpha_occs,
                       oe_alpha, coeffs_alpha, obasis, restricted)
        mo = MolecularOrbitals(
            'restricted', norb, norb, 2 * orb_alpha_occs, orb_alpha_coeffs,
            orb_alpha_energies)
    else:
        norb_alpha = _get_norb_nel(oe_alpha)[0]
        norb_beta = _get_norb_nel(oe_beta)[0]
        assert norb_alpha == norb_beta
        orb_alpha_coeffs = np.zeros([obasis.nbasis, norb_alpha])
        orb_alpha_energies = np.zeros(norb_alpha)
        orb_alpha_occs = np.zeros(norb_alpha)
        orb_beta_coeffs = np.zeros([obasis.nbasis, norb_beta])
        orb_beta_energies = np.zeros(norb_beta)
        orb_beta_occs = np.zeros(norb_beta)
        _fill_orbitals(orb_alpha_coeffs, orb_alpha_energies, orb_alpha_occs,
                       oe_alpha, coeffs_alpha, obasis, restricted)
        _fill_orbitals(orb_beta_coeffs, orb_beta_energies, orb_beta_occs,
                       oe_beta, coeffs_beta, obasis, restricted)

        mo = MolecularOrbitals(
            'unrestricted', norb_alpha, norb_beta,
            np.concatenate((orb_alpha_occs, orb_beta_occs), axis=0),
            np.concatenate((orb_alpha_coeffs, orb_beta_coeffs), axis=1),
            np.concatenate((orb_alpha_energies, orb_beta_energies), axis=0),
        )

    result = {
        'obasis': obasis,
        'mo': mo,
        'atcoords': np.zeros((1, 3), float),
        'atnums': np.array([atnum]),
        'energy': energy,
        'atcorenums': np.array([atcorenum]),
    }
    return result
