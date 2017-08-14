# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Gaussian LOG and FCHK file formats"""
from collections import OrderedDict

import numpy as np

from .utils import set_four_index_element

__all__ = ['load_operators_g09', 'FCHKFile', 'load_fchk']


def load_operators_g09(fn):
    """Loads several two- and four-index operators from a Gaussian log file.

       **Argument:**

       fn
            The filename of the Gaussian log file.

       The following two-index operators are loaded if present: overlap,
       kinetic, nuclear attraction. The following four-index operator is loaded
       if present: electrostatic repulsion. In order to make all these matrices
       are present in the Gaussian log file, the following commands must be used
       in the Gaussian input file:

            scf(conventional) iop(3/33=5) extralinks=l316 iop(3/27=999)

       **Returns:** A dictionary that may contain the keys: ``olp``, ``kin``,
       ``na`` and/or ``er``.
    """

    with open(fn) as f:
        # First get the line with the number of orbital basis functions
        for line in f:
            if line.startswith('    NBasis ='):
                nbasis = int(line[12:18])
                break

        # Then load the two- and four-index operators. This part is written such
        # that it does not make any assumptions about the order in which these
        # operators are printed.
        result = {}
        for line in f:
            if line.startswith(' *** Overlap ***'):
                result['olp'] = _load_twoindex_g09(f, nbasis)
            elif line.startswith(' *** Kinetic Energy ***'):
                result['kin'] = _load_twoindex_g09(f, nbasis)
            elif line.startswith(' ***** Potential Energy *****'):
                result['na'] = _load_twoindex_g09(f, nbasis)
            elif line.startswith(' *** Dumping Two-Electron integrals ***'):
                result['er'] = _load_fourindex_g09(f, nbasis)
        return result


def _load_twoindex_g09(f, nbasis):
    """Load a two-index operator from a Gaussian log file

       **Arguments:**

       f
            A file object for the Gaussian log file in read mode.

       nbasis
            The number of orbital basis functions.
    """
    result = np.zeros((nbasis, nbasis))
    block_counter = 0
    while block_counter < nbasis:
        # skip the header line
        next(f)
        # determine the number of rows in this part
        nrow = nbasis - block_counter
        for i in range(nrow):
            words = next(f).split()[1:]
            for j in range(len(words)):
                value = float(words[j].replace('D', 'E'))
                result[i + block_counter, j + block_counter] = value
                result[j + block_counter, i + block_counter] = value
        block_counter += 5
    return result


def _load_fourindex_g09(f, nbasis):
    """Load a four-index operator from a Gaussian log file

       **Arguments:**

       f
            A file object for the Gaussian log file in read mode.

       nbasis
            The number of orbital basis functions.
    """
    result = np.zeros((nbasis, nbasis, nbasis, nbasis))
    # Skip first six lines
    for i in range(6):
        next(f)
    # Start reading elements until a line is encountered that does not start
    # with ' I='
    while True:
        line = next(f)
        if not line.startswith(' I='):
            break
        # print line[3:7], line[9:13], line[15:19], line[21:25], line[28:].replace('D', 'E')
        i = int(line[3:7]) - 1
        j = int(line[9:13]) - 1
        k = int(line[15:19]) - 1
        l = int(line[21:25]) - 1
        value = float(line[29:].replace('D', 'E'))
        # Gaussian uses the chemists notation for the 4-center indexes. HORTON
        # uses the physicists notation.
        set_four_index_element(result, i, k, j, l, value)
    return result


class FCHKFile(dict):
    """Reader for Formatted checkpoint files

       After initialization, the data from the file is available in the fields
       dictionary. Also the following attributes are read from the file: title,
       command, lot (level of theory) and basis.
    """

    def __init__(self, filename, field_labels=None):
        """
           **Arguments:**

           filename
                The formatted checkpoint file.

           **Optional arguments:**

           field_labels
                When provided, only these fields are read from the formatted
                checkpoint file. (This can save a lot of time.)
        """
        dict.__init__(self, [])
        self.filename = filename
        self._read(filename, set(field_labels))

    def _read(self, filename, field_labels=None):
        """Read all the requested fields"""

        # if fields is None, all fields are read
        def read_field(f):
            """Read a single field"""
            datatype = None
            while datatype is None:
                # find a sane header line
                line = f.readline()
                if line == "":
                    return False

                label = line[:43].strip()
                if field_labels is not None:
                    if len(field_labels) == 0:
                        return False
                    elif label not in field_labels:
                        return True
                    else:
                        field_labels.discard(label)
                line = line[43:]
                words = line.split()
                if len(words) == 0:
                    return True

                if words[0] == 'I':
                    datatype = int
                elif words[0] == 'R':
                    datatype = float

            if len(words) == 2:
                try:
                    value = datatype(words[1])
                except ValueError:
                    return True
            elif len(words) == 3:
                if words[1] != "N=":
                    raise IOError("Unexpected line in formatted checkpoint file %s\n%s" % (filename, line[:-1]))
                length = int(words[2])
                value = np.zeros(length, datatype)
                counter = 0
                try:
                    while counter < length:
                        line = f.readline()
                        if line == "":
                            raise IOError("Unexpected end of formatted checkpoint file %s" % filename)
                        for word in line.split():
                            try:
                                value[counter] = datatype(word)
                            except (ValueError, OverflowError) as e:
                                raise IOError('Could not interpret word while reading %s: %s' % (word, filename))
                            counter += 1
                except ValueError:
                    return True
            else:
                raise IOError("Unexpected line in formatted checkpoint file %s\n%s" % (filename, line[:-1]))

            self[label] = value
            return True

        f = open(filename, 'r')
        self.title = f.readline()[:-1].strip()
        words = f.readline().split()
        if len(words) == 3:
            self.command, self.lot, self.obasis = words
        elif len(words) == 2:
            self.command, self.lot = words
        else:
            raise IOError('The second line of the FCHK file should contain two or three words.')

        while read_field(f):
            pass

        f.close()


def triangle_to_dense(triangle):
    """Convert a symmetric matrix in triangular storage to a dense square matrix.

       **Arguments:**

       triangle
            A row vector containing all the unique matrix elements of symmetrix
            matrix. (Either the lower-triangular part in row major-order or the
            upper-triangular part in column-major order.)

       **Returns:** a square symmetrix matrix.
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


def load_fchk(filename):
    """Load from a formatted checkpoint file.

       **Arguments:**

       filename
            The filename of the Gaussian formatted checkpoint file.

       **Returns** a dictionary with: ``title``, ``coordinates``, ``numbers``,
       ``obasis``, ``orb_alpha``, ``permutation``, ``energy``,
       ``pseudo_numbers``, ``mulliken_charges``. The dictionary may also
       contain: ``npa_charges``, ``esp_charges``, ``orb_beta``, ``dm_full_mp2``,
       ``dm_spin_mp2``, ``dm_full_mp3``, ``dm_spin_mp3``, ``dm_full_cc``,
       ``dm_spin_cc``, ``dm_full_ci``, ``dm_spin_ci``, ``dm_full_scf``,
       ``dm_spin_scf``, ``polar``, ``dipole_moment``, ``quadrupole_moment``.
    """

    fchk = FCHKFile(filename, [
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
    alphas = fchk["Primitive exponents"]
    ccoeffs_level1 = fchk["Contraction coefficients"]
    ccoeffs_level2 = fchk.get("P(S=P) Contraction coefficients")

    my_shell_types = []
    my_shell_map = []
    my_nprims = []
    my_alphas = []
    con_coeffs = []
    counter = 0
    for i, n in enumerate(nprims):
        if shell_types[i] == -1:
            # Special treatment for SP shell type
            my_shell_types.append(0)
            my_shell_types.append(1)
            my_shell_map.append(shell_map[i])
            my_shell_map.append(shell_map[i])
            my_nprims.append(nprims[i])
            my_nprims.append(nprims[i])
            my_alphas.append(alphas[counter:counter + n])
            my_alphas.append(alphas[counter:counter + n])
            con_coeffs.append(ccoeffs_level1[counter:counter + n])
            con_coeffs.append(ccoeffs_level2[counter:counter + n])
        else:
            my_shell_types.append(shell_types[i])
            my_shell_map.append(shell_map[i])
            my_nprims.append(nprims[i])
            my_alphas.append(alphas[counter:counter + n])
            con_coeffs.append(ccoeffs_level1[counter:counter + n])
        counter += n
    my_shell_types = np.array(my_shell_types)
    my_shell_map = np.array(my_shell_map)
    my_nprims = np.array(my_nprims)
    my_alphas = np.concatenate(my_alphas)
    con_coeffs = np.concatenate(con_coeffs)
    del shell_map
    del shell_types
    del nprims
    del alphas

    obasis = {"centers": coordinates, "shell_map": my_shell_map, "nprims": my_nprims,
              "shell_types": my_shell_types, "alphas": my_alphas, "con_coeffs": con_coeffs}

    # permutation of the orbital basis functions
    permutation_rules = {
        -9: np.arange(19),
        -8: np.arange(17),
        -7: np.arange(15),
        -6: np.arange(13),
        -5: np.arange(11),
        -4: np.arange(9),
        -3: np.arange(7),
        -2: np.arange(5),
        0: np.array([0]),
        1: np.arange(3),
        2: np.array([0, 3, 4, 1, 5, 2]),
        3: np.array([0, 4, 5, 3, 9, 6, 1, 8, 7, 2]),
        4: np.arange(15)[::-1],
        5: np.arange(21)[::-1],
        6: np.arange(28)[::-1],
        7: np.arange(36)[::-1],
        8: np.arange(45)[::-1],
        9: np.arange(55)[::-1],
    }
    permutation = []
    for shell_type in my_shell_types:
        permutation.extend(permutation_rules[shell_type] + len(permutation))
    permutation = np.array(permutation, dtype=int)

    result = {
        'title': fchk.title,
        'coordinates': system_coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'permutation': permutation,
        'pseudo_numbers': pseudo_numbers,
    }

    nbasis = fchk.get("Number of basis functions")

    # C) Load density matrices
    def load_dm(label):
        if label in fchk:
            dm = np.zeros((nbasis, nbasis))
            start = 0
            for i in range(nbasis):
                stop = start + i + 1
                dm[i, :i + 1] = fchk[label][start:stop]
                dm[:i + 1, i] = fchk[label][start:stop]
                start = stop
            return dm

    # First try to load the post-hf density matrices.
    for key in 'MP2', 'MP3', 'CC', 'CI', 'SCF':
        dm_full = load_dm('Total %s Density' % key)
        if dm_full is not None:
            result['dm_full_%s' % key.lower()] = dm_full
        dm_spin = load_dm('Spin %s Density' % key)
        if dm_spin is not None:
            result['dm_spin_%s' % key.lower()] = dm_spin

    # D) Load the wavefunction
    # Handle small difference in fchk files from g03 and g09
    nbasis_indep = fchk.get("Number of independant functions") or \
                   fchk.get("Number of independent functions")
    if nbasis_indep is None:
        nbasis_indep = nbasis

    # Load orbitals
    nalpha = fchk['Number of alpha electrons']
    nbeta = fchk['Number of beta electrons']
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        raise ValueError('The file %s does not contain a positive number of electrons.' % filename)
    result['orb_alpha'] = (nbasis, nbasis_indep)
    result['orb_alpha_coeffs'] = np.copy(fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
    result['orb_alpha_energies'] = np.copy(fchk['Alpha Orbital Energies'])
    aoccs = np.zeros(nbasis)
    aoccs[:nalpha] = 1.0
    result['orb_alpha_occs'] = aoccs
    if 'Beta Orbital Energies' in fchk:
        # UHF case
        result['orb_beta'] = (nbasis, nbasis_indep)
        result['orb_beta_coeffs'] = np.copy(fchk['Beta MO coefficients'].reshape(nbasis_indep, nbasis).T)
        result['orb_beta_energies'] = np.copy(fchk['Beta Orbital Energies'])
        boccs = np.zeros(nbasis)
        boccs[:nbeta] = 1.0
        result['orb_beta_occs'] = boccs

    elif fchk['Number of beta electrons'] != fchk['Number of alpha electrons']:
        # ROHF case
        result['orb_beta'] = (nbasis, nbasis_indep)
        result['orb_beta_coeffs'] = fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T
        result['orb_beta_energies'] = fchk['Alpha Orbital Energies']
        boccs = np.zeros(nbasis)
        boccs[:nbeta] = 1.0
        result['orb_beta_occs'] = boccs

        # Delete dm_full_scf because it is known to be buggy
        result.pop('dm_full_scf')

    # E) Load properties
    result['energy'] = fchk['Total Energy']
    if 'Polarizability' in fchk:
        result['polar'] = triangle_to_dense(fchk['Polarizability'])
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
