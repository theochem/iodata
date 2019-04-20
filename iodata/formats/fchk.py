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
"""Module for handling GAUSSIAN FCHK file format."""


from typing import Dict, List, Tuple

import numpy as np

from ..basis import MolecularBasis, Shell, HORTON2_CONVENTIONS
from ..utils import LineIterator, MolecularOrbitals


__all__ = []


patterns = ['*.fchk']


CONVENTIONS = {
    (9, 'p'): HORTON2_CONVENTIONS[(9, 'p')],
    (8, 'p'): HORTON2_CONVENTIONS[(8, 'p')],
    (7, 'p'): HORTON2_CONVENTIONS[(7, 'p')],
    (6, 'p'): HORTON2_CONVENTIONS[(6, 'p')],
    (5, 'p'): HORTON2_CONVENTIONS[(5, 'p')],
    (4, 'p'): HORTON2_CONVENTIONS[(4, 'p')],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'yz', 'xz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'c'): HORTON2_CONVENTIONS[(4, 'c')][::-1],
    (5, 'c'): HORTON2_CONVENTIONS[(5, 'c')][::-1],
    (6, 'c'): HORTON2_CONVENTIONS[(6, 'c')][::-1],
    (7, 'c'): HORTON2_CONVENTIONS[(7, 'c')][::-1],
    (8, 'c'): HORTON2_CONVENTIONS[(8, 'c')][::-1],
    (9, 'c'): HORTON2_CONVENTIONS[(9, 'c')][::-1],
}


# pylint: disable=too-many-branches,too-many-statements
def load(lit: LineIterator) -> Dict:
    """Load data from a GAUSSIAN FCHK file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``pseudo_numbers``,
        ``obasis``, ``mo``, ``energy`` & ``mulliken_charges`` keys and
        corresponding values. It may also contain ``npa_charges``, ``esp_charges``,
        ``dm_full_mp2``, ``dm_spin_mp2``, ``dm_full_mp3``, ``dm_spin_mp3``, ``dm_full_cc``,
        ``dm_spin_cc``, ``dm_full_ci``, ``dm_spin_ci``, ``dm_full_scf``, ``dm_spin_scf``,
        ``polar``, ``dipole_moment`` & ``quadrupole_moment`` keys and their values as well.

    """
    fchk = _load_fchk_low(lit, [
        "Number of electrons", "Number of basis functions",
        "Number of independant functions",
        "Number of independent functions",
        "Number of alpha electrons", "Number of beta electrons",
        "Atomic numbers", "Current cartesian coordinates",
        "Shell types", "Shell to atom map", "Shell to atom map",
        "Number of primitives per shell", "Primitive exponents",
        "Contraction coefficients", "P(S=P) Contraction coefficients",
        "Alpha Orbital Energies", "Alpha MO coefficients",
        "Beta Orbital Energies", "Beta MO coefficients",
        "Total Energy", "Nuclear charges",
        'Total SCF Density', 'Spin SCF Density',
        'Total MP2 Density', 'Spin MP2 Density',
        'Total MP3 Density', 'Spin MP3 Density',
        'Total CC Density', 'Spin CC Density',
        'Total CI Density', 'Spin CI Density',
        'Mulliken Charges', 'ESP Charges', 'NPA Charges',
        'Polarizability', 'Dipole Moment', 'Quadrupole Moment',
    ])

    # A) Load the geometry
    numbers = fchk["Atomic numbers"]
    coordinates = fchk["Current cartesian coordinates"].reshape(-1, 3)
    pseudo_numbers = fchk["Nuclear charges"]
    # Mask out ghost atoms
    mask = pseudo_numbers != 0.0
    numbers = numbers[mask]
    # Do not overwrite coordinates array, because it is needed to specify basis
    system_coordinates = coordinates[mask]
    pseudo_numbers = pseudo_numbers[mask]

    # B) Load the orbital basis set
    shell_types = fchk["Shell types"]
    shell_map = fchk["Shell to atom map"] - 1
    nprims = fchk["Number of primitives per shell"]
    exponents = fchk["Primitive exponents"]
    ccoeffs_level1 = fchk["Contraction coefficients"]
    ccoeffs_level2 = fchk.get("P(S=P) Contraction coefficients")

    shells = []
    counter = 0
    # First loop over all shells
    for i, n in enumerate(nprims):
        if shell_types[i] == -1:
            # Special treatment for SP shell type
            shells.append(Shell(
                shell_map[i],
                [0, 1],
                ['c', 'c'],
                exponents[counter:counter + n],
                np.stack([ccoeffs_level1[counter:counter + n],
                          ccoeffs_level2[counter:counter + n]], axis=1)
            ))
        else:
            shells.append(Shell(
                shell_map[i],
                [abs(shell_types[i])],
                ['p' if shell_types[i] < 0 else 'c'],
                exponents[counter:counter + n],
                ccoeffs_level1[counter:counter + n][:, np.newaxis]
            ))
        counter += n
    del shell_map
    del shell_types
    del nprims
    del exponents

    obasis = MolecularBasis(coordinates, shells, CONVENTIONS, 'L2')

    result = {
        'title': fchk['title'],
        'coordinates': system_coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'pseudo_numbers': pseudo_numbers,
    }

    nbasis = fchk["Number of basis functions"]

    # C) Load density matrices
    for lot in 'MP2', 'MP3', 'CC', 'CI', 'SCF':
        _load_dm('Total %s Density' % lot, fchk, result, 'dm_full_%s' % lot.lower())
        _load_dm('Spin %s Density' % lot, fchk, result, 'dm_spin_%s' % lot.lower())

    # D) Load the wavefunction
    # Handle small difference in fchk files from g03 and g09
    nbasis_indep = fchk.get("Number of independant functions", nbasis)

    # Load orbitals
    nalpha = fchk['Number of alpha electrons']
    nbeta = fchk['Number of beta electrons']
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        lit.error('The number of electrons is not positive.')
    if nalpha < nbeta:
        raise ValueError('n_alpha={0} < n_beta={1} is not valid!'.format(nalpha, nbeta))

    if 'Beta Orbital Energies' in fchk:
        # unrestricted
        mo_type = 'unrestricted'
        mo_coeffs_a = np.copy(fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
        mo_coeffs_b = np.copy(fchk['Beta MO coefficients'].reshape(nbasis_indep, nbasis).T)
        mo_coeffs = np.concatenate((mo_coeffs_a, mo_coeffs_b), axis=1)
        mo_energy = np.copy(fchk['Alpha Orbital Energies'])
        mo_energy = np.concatenate((mo_energy, np.copy(fchk['Beta Orbital Energies'])), axis=0)
        mo_occs = np.zeros(2 * nbasis_indep)
        mo_occs[:nalpha] = 1.0
        mo_occs[nbasis_indep: nbasis_indep + nbeta] = 1.0

    elif fchk['Number of beta electrons'] != fchk['Number of alpha electrons']:
        # restricted open-shell
        assert nalpha > nbeta, (nalpha, nbeta)
        mo_type = 'restricted'
        mo_coeffs = np.copy(fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
        mo_energy = np.copy(fchk['Alpha Orbital Energies'])
        mo_occs = np.zeros(nbasis_indep)
        mo_occs[:nalpha] = 1.0
        mo_occs[:nbeta] = 2.0

        # Delete dm_full_scf because it is known to be buggy
        result.pop('dm_full_scf')

    else:
        # restricted close-shell
        assert nalpha == nbeta
        mo_type = 'restricted'
        mo_coeffs = np.copy(fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
        mo_energy = np.copy(fchk['Alpha Orbital Energies'])
        mo_occs = np.zeros(nbasis_indep)
        mo_occs[:nalpha] = 2.0

    # create a MO namedtuple
    result['mo'] = MolecularOrbitals(mo_type, nalpha, nbeta, mo_occs, mo_coeffs, None, mo_energy)

    # E) Load properties
    result['energy'] = fchk['Total Energy']
    if 'Polarizability' in fchk:
        result['polar'] = _triangle_to_dense(fchk['Polarizability'])
    if 'Dipole Moment' in fchk:
        result['dipole_moment'] = fchk['Dipole Moment']
    if 'Quadrupole Moment' in fchk:
        # Convert to HORTON ordering: xx, xy, xz, yy, yz, zz
        result['quadrupole_moment'] = fchk['Quadrupole Moment'][[0, 3, 4, 1, 5, 2]]

    # F) Load optional properties
    # Mask out ghost atoms from charges
    if 'Mulliken Charges' in fchk:
        result['mulliken_charges'] = fchk['Mulliken Charges'][mask]
    if 'ESP Charges' in fchk:
        result['esp_charges'] = fchk['ESP Charges'][mask]
    if 'NPA Charges' in fchk:
        result['npa_charges'] = fchk['NPA Charges'][mask]

    return result


def _load_fchk_low(lit: LineIterator, labels: List[str] = None) -> Dict:
    """Read selected fields from a formatted checkpoint file."""
    result = {}
    # Read the two-line header
    result['title'] = next(lit).strip()
    words = next(lit).split()
    if len(words) == 3:
        result['command'], result['lot'], result['obasis'] = words
    elif len(words) == 2:
        result['command'], result['lot'] = words
    else:
        lit.error('The second line of the FCHK file should contain two or three words.')

    # Read all fields, go all they way until the until unless all requested
    # labels are used.
    if labels is not None:
        labels = set(labels)
    while labels is None or labels:
        try:
            label, value = _load_fchk_field(lit, labels)
        except StopIteration:
            # We read the whole file, this happens when more labels are given
            # than those present in the file, which should be allowed.
            break
        result[label] = value
    return result


# pylint: disable=too-many-branches
def _load_fchk_field(lit: LineIterator, labels: List[str]) -> Tuple[str, object]:
    """Read a single field with one of the given labels."""
    while True:
        # find a sane header line
        line = next(lit)
        label = line[:43].strip()
        words = line[43:].split()
        if not words:
            continue
        if words[0] == 'I':
            datatype = int
        elif words[0] == 'R':
            datatype = float
        else:
            continue
        if labels is not None and label not in labels:
            continue
        if len(words) == 2:
            try:
                return label, datatype(words[1])
            except ValueError:
                lit.error("Could not interpret: {}".format(words[1]))
        elif len(words) == 3:
            if words[1] != "N=":
                lit.error("Expected N= not found.")
            length = int(words[2])
            value = np.zeros(length, datatype)
            counter = 0
            words = []
            while counter < length:
                if not words:
                    words = next(lit).split()
                word = words.pop(0)
                try:
                    value[counter] = datatype(word)
                except (ValueError, OverflowError):
                    lit.error('Could not interpret: {}'.format(word))
                counter += 1
            return label, value


def _load_dm(label: str, fchk: dict, result: dict, key: str):
    """Load a density matrix from the FCHK file if present.

    Parameters
    ----------
    label
        The label in the FCHK file.
    fchk
        The dictionary with labels from the FCHK file.
    result
        The output dictionary.
    key:
        The key to be used in the output dictionary.

    """
    if label in fchk:
        result[key] = _triangle_to_dense(fchk[label])


def _triangle_to_dense(triangle: np.ndarray) -> np.ndarray:
    """Convert a symmetric matrix in triangular storage to a dense square matrix.

    Parameters
    ----------
    triangle
        A row vector containing all the unique matrix elements of symmetric
        matrix. (Either the lower-triangular part in row major-order or the
        upper-triangular part in column-major order.)

    Returns
    -------
    ndarray
        a square symmetric matrix.

    """
    nrow = int(np.round((np.sqrt(1 + 8 * len(triangle)) - 1) / 2))
    result = np.zeros((nrow, nrow))
    begin = 0
    for irow in range(nrow):
        end = begin + irow + 1
        result[irow, :irow + 1] = triangle[begin:end]
        result[:irow + 1, irow] = triangle[begin:end]
        begin = end
    return result
