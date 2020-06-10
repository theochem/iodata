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
"""Gaussian FCHK file format."""


from fnmatch import fnmatch
from typing import List, Tuple, Iterator

import numpy as np

from ..basis import MolecularBasis, Shell, HORTON2_CONVENTIONS
from ..docstrings import document_load_one, document_load_many
from ..orbitals import MolecularOrbitals
from ..utils import LineIterator, amu


__all__ = []


PATTERNS = ['*.fchk', '*.fch']


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
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'c'): HORTON2_CONVENTIONS[(4, 'c')][::-1],
    (5, 'c'): HORTON2_CONVENTIONS[(5, 'c')][::-1],
    (6, 'c'): HORTON2_CONVENTIONS[(6, 'c')][::-1],
    (7, 'c'): HORTON2_CONVENTIONS[(7, 'c')][::-1],
    (8, 'c'): HORTON2_CONVENTIONS[(8, 'c')][::-1],
    (9, 'c'): HORTON2_CONVENTIONS[(9, 'c')][::-1],
}


# pylint: disable=too-many-branches,too-many-statements
@document_load_one(
    "Gaussian Formatted Checkpoint",
    ['atcharges', 'atcoords', 'atnums', 'atcorenums', 'energy', 'lot', 'mo', 'obasis',
     'obasis_name', 'run_type', 'title'],
    ['atfrozen', 'atgradient', 'athessian', 'atmasses', 'one_rdms', 'extra', 'moments'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    fchk = _load_fchk_low(lit, [
        "Number of electrons", "Number of basis functions",
        "Number of independant functions",  # independ__a__nt (g03)
        "Number of independent functions",  # independ__e__nt (g09, g16, ...)
        "Number of alpha electrons", "Number of beta electrons",
        "Atomic numbers", "Current cartesian coordinates",
        "Real atomic weights",
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
        'Cartesian Gradient', 'Cartesian Force Constants', 'MicOpt',
    ])

    # A) Load a bunch of simple things
    result = {
        'title': fchk['title'],
        'energy': fchk['Total Energy'],
        'lot': fchk['lot'].lower(),
        'obasis_name': fchk['obasis_name'].lower(),
        'atcoords': fchk["Current cartesian coordinates"].reshape(-1, 3),
        'atnums': fchk["Atomic numbers"],
        'atcorenums': fchk["Nuclear charges"],
    }

    atmasses = fchk.get("Real atomic weights")
    if atmasses is not None:
        result['atmasses'] = atmasses * amu
    atgradient = fchk.get('Cartesian Gradient')
    if atgradient is not None:
        result['atgradient'] = atgradient.reshape(-1, 3)
    athessian = fchk.get('Cartesian Force Constants')
    if athessian is not None:
        result['athessian'] = _triangle_to_dense(athessian)
    atfrozen = fchk.get("MicOpt")
    if atfrozen is not None:
        result['atfrozen'] = (atfrozen == -2)
    run_types = {'SP': 'energy', 'FOpt': 'opt', 'Scan': 'scan', 'Freq': 'freq'}
    run_type = run_types.get(fchk['command'])
    if run_type is not None:
        result['run_type'] = run_type

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

    result['obasis'] = MolecularBasis(shells, CONVENTIONS, 'L2')
    nbasis = fchk["Number of basis functions"]

    # C) Load density matrices
    one_rdms = {}
    _load_dm('Total SCF Density', fchk, one_rdms, 'scf')
    _load_dm('Spin SCF Density', fchk, one_rdms, 'scf_spin')
    # only one of the lots should be present, hence using the same key
    for lot in 'MP2', 'MP3', 'CC', 'CI':
        _load_dm('Total {} Density'.format(lot), fchk, one_rdms, 'post_scf')
        _load_dm('Spin {} Density'.format(lot), fchk, one_rdms, 'post_scf_spin')
    if one_rdms:
        result['one_rdms'] = one_rdms

    # D) Load the wavefunction
    # Handle small difference in spelling in fchk files from g03 and g09:
    # "independ__e__nt" versus "independ__a__nt".
    nbasis_indep = fchk.get("Number of independant functions",
                            fchk.get("Number of independent functions", nbasis))

    # Load orbitals
    nalpha = fchk['Number of alpha electrons']
    nbeta = fchk['Number of beta electrons']
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        lit.error('The number of electrons is not positive.')
    if nalpha < nbeta:
        raise ValueError('n_alpha={0} < n_beta={1} is not valid!'.format(nalpha, nbeta))

    norba = fchk['Alpha Orbital Energies'].shape[0]
    mo_coeffs = np.copy(fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
    mo_energies = np.copy(fchk['Alpha Orbital Energies'])

    if 'Beta Orbital Energies' in fchk:
        # unrestricted
        norbb = fchk['Beta Orbital Energies'].shape[0]
        mo_coeffs_b = np.copy(fchk['Beta MO coefficients'].reshape(nbasis_indep, nbasis).T)
        mo_coeffs = np.concatenate((mo_coeffs, mo_coeffs_b), axis=1)
        mo_energies = np.concatenate((mo_energies, np.copy(fchk['Beta Orbital Energies'])), axis=0)
        mo_occs = np.zeros(2 * nbasis_indep)
        mo_occs[:nalpha] = 1.0
        mo_occs[nbasis_indep: nbasis_indep + nbeta] = 1.0
        mo = MolecularOrbitals('unrestricted', norba, norbb, mo_occs, mo_coeffs, mo_energies, None)
    else:
        # restricted closed-shell and open-shell
        mo_occs = np.zeros(nbasis_indep)
        mo_occs[:nalpha] = 1.0
        mo_occs[:nbeta] = 2.0
        if nalpha != nbeta:
            # delete dm_full_scf because it is known to be buggy
            result['one_rdms'].pop('scf')
        mo = MolecularOrbitals('restricted', norba, norba, mo_occs, mo_coeffs, mo_energies, None)
    result['mo'] = mo

    # E) Load properties
    if 'Polarizability' in fchk:
        result['extra'] = {'polarizability_tensor': _triangle_to_dense(fchk['Polarizability'])}
    moments = {}
    if 'Dipole Moment' in fchk:
        moments[(1, 'c')] = fchk['Dipole Moment']
    if 'Quadrupole Moment' in fchk:
        # Convert to alphabetical ordering: xx, xy, xz, yy, yz, zz
        moments[(2, 'c')] = fchk['Quadrupole Moment'][[0, 3, 4, 1, 5, 2]]
    if moments:
        result['moments'] = moments
    atcharges = {}
    if 'Mulliken Charges' in fchk:
        atcharges['mulliken'] = fchk['Mulliken Charges']
    if 'ESP Charges' in fchk:
        atcharges['esp'] = fchk['ESP Charges']
    if 'NPA Charges' in fchk:
        atcharges['npa'] = fchk['NPA Charges']
    if atcharges:
        result['atcharges'] = atcharges

    return result


LOAD_MANY_NOTES = """
Trajectories from a Gaussian optimization, relaxed scan or IRC calculation are written in
groups of frames, called "points" in the Gaussian world, e.g. to discrimininate between
different values of the constraint in a relaxed geometry. In most cases, e.g. IRC or
conventional optimization, there is only one "point". Within one "point", one can have
multiple geometries and their properties. This information is stored in the ``extra``
attribute:

- ``ipoint`` is the counter for a point
- ``npoint`` is the total number of points.
- ``istep`` is the counter within one "point"
- ``nstep`` is the total number of geometries within in a "point".
- ``reaction_coordinate`` is only present in case of an IRC calculation.
"""


@document_load_many("XYZ", ['atcoords', 'atgradient', 'atnums', 'atcorenums',
                            'energy', 'extra', 'title'], [], {}, LOAD_MANY_NOTES)
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    fchk = _load_fchk_low(lit, [
        "Atomic numbers", "Current cartesian coordinates", "Nuclear charges",
        "IRC *", "Optimization *", "Opt point *"])

    # Determine the type of calculation: IRC or Optimization
    if "IRC Number of geometries" in fchk:
        prefix = "IRC point"
        nsteps = fchk["IRC Number of geometries"]
    elif "Optimization Number of geometries" in fchk:
        prefix = "Opt point"
        nsteps = fchk["Optimization Number of geometries"]
    else:
        lit.error("Could not find IRC or Optimization trajectory in FCHK file.")

    natom = fchk["Atomic numbers"].size
    for ipoint, nstep in enumerate(nsteps):
        results_geoms = fchk["{} {:7d} Results for each geome".format(prefix, ipoint + 1)]
        trajectory = list(zip(
            results_geoms[::2], results_geoms[1::2],
            fchk["{} {:7d} Geometries".format(prefix, ipoint + 1)].reshape(-1, natom, 3),
            fchk["{} {:7d} Gradient at each geome".format(prefix, ipoint + 1)].reshape(-1, natom, 3)
        ))
        assert len(trajectory) == nstep
        for istep, (energy, recor, atcoords, gradients) in enumerate(trajectory):
            data = {
                'title': fchk['title'],
                'atnums': fchk["Atomic numbers"],
                'atcorenums': fchk["Nuclear charges"],
                'energy': energy,
                'atcoords': atcoords,
                'atgradient': gradients,
                'extra': {
                    'ipoint': ipoint,
                    'npoint': len(nsteps),
                    'istep': istep,
                    'nstep': nstep,
                },
            }
            if prefix == "IRC point":
                data['extra']['reaction_coordinate'] = recor
            yield data


def _load_fchk_low(lit: LineIterator, label_patterns: List[str] = None) -> dict:
    """Read selected fields from a formatted checkpoint file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    label_patterns
        A list of Unix shell-style wildcard patterns of labels to read.

    Returns
    -------
    fields
        The data read from the FCHK file. Keys are the field names and values
        are either scalar or array data. Arrays are always one-dimensional.

    """
    result = {}
    # Read the two-line header
    result['title'] = next(lit).strip()
    words = next(lit).split()
    if len(words) == 3:
        result['command'], result['lot'], result['obasis_name'] = words
    elif len(words) == 2:
        result['command'], result['lot'] = words
    else:
        lit.error('The second line of the FCHK file should contain two or three words.')

    while True:
        try:
            label, value = _load_fchk_field(lit, label_patterns)
        except StopIteration:
            # We always read until the end of the file.
            break
        result[label] = value
    return result


# pylint: disable=too-many-branches
def _load_fchk_field(lit: LineIterator, label_patterns: List[str]) -> Tuple[str, object]:
    """Read a single field matching one of the given label_patterns.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    label_patterns
        A list of Unix shell-style wildcard patterns. The next field matching
        one of the patterns is returned

    Returns
    -------
    label
        The name of the field
    value
        The scalar or array data of the field.

    """
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
        if not (label_patterns is None
                or any(fnmatch(label, label_pattern) for label_pattern in label_patterns)):
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
