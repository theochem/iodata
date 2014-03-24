# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
'''Gaussian LOG and FCHK file fromats'''


import numpy as np

from horton.meanfield.wfn import AufbauOccModel, RestrictedWFN, UnrestrictedWFN


__all__ = ['load_operators_g09', 'FCHKFile', 'load_fchk']


def load_operators_g09(fn, lf):
    """Loads several one- and two-body operators from a Gaussian log file.

       **Arugment:**

       fn
            The filename of the Gaussian log file.

       lf
            A LinalgFactory instance.

       The following one-body operators are loaded if present: overlap, kinetic,
       nuclear attraction. The following two-body operator is loaded if present:
       electrostatic repulsion. In order to make all these matrices are present
       in the Gaussian log file, the following commands must be used in the
       Gaussian input file:

            scf(conventional) iop(3/33=5) extralinks=l316 iop(3/27=999)
    """

    with open(fn) as f:
        # First get the line with the number of orbital basis functions
        for line in f:
            if line.startswith('    NBasis ='):
                nbasis = int(line[12:18])
                break

        # Then load the one- and two-body operators. This part is written such
        # that it does not make any assumptions about the order in which these
        # operators are printed.

        result = {}
        for line in f:
            if line.startswith(' *** Overlap ***'):
                result['olp'] = _load_onebody_g09(f, nbasis, lf)
            elif line.startswith(' *** Kinetic Energy ***'):
                result['kin'] = _load_onebody_g09(f, nbasis, lf)
            elif line.startswith(' ***** Potential Energy *****'):
                result['na'] = _load_onebody_g09(f, nbasis, lf)
            elif line.startswith(' *** Dumping Two-Electron integrals ***'):
                result['er'] = _load_twobody_g09(f, nbasis, lf)

        return result


def _load_onebody_g09(f, nbasis, lf):
    """Load a one-body operator from a Gaussian log file

       **Arguments:**

       f
            A file object for the Gaussian log file in read mode.

       nbasis
            The number of orbital basis functions.

       lf
            A LinalgFactory instance.
    """
    result = lf.create_one_body(nbasis)
    block_counter = 0
    while block_counter < nbasis:
        # skip the header line
        f.next()
        # determine the number of rows in this part
        nrow = nbasis - block_counter
        for i in xrange(nrow):
            words = f.next().split()[1:]
            for j in xrange(len(words)):
                value = float(words[j].replace('D', 'E'))
                result.set_element(i+block_counter, j+block_counter, value)
        block_counter += 5
    return result


def _load_twobody_g09(f, nbasis, lf):
    """Load a two-body operator from a Gaussian log file

       **Arguments:**

       f
            A file object for the Gaussian log file in read mode.

       nbasis
            The number of orbital basis functions.

       lf
            A LinalgFactory instance.
    """
    result = lf.create_two_body(nbasis)
    # Skip first six lines
    for i in xrange(6):
        f.next()
    # Start reading elements until a line is encountered that does not start
    # with ' I='
    while True:
        line = f.next()
        if not line.startswith(' I='):
            break
        #print line[3:7], line[9:13], line[15:19], line[21:25], line[28:].replace('D', 'E')
        i = int(line[3:7])-1
        j = int(line[9:13])-1
        k = int(line[15:19])-1
        l = int(line[21:25])-1
        value = float(line[29:].replace('D', 'E'))
        # Gaussian uses the chemists notation for the 4-center indexes. Horton
        # uses the physicists notation.
        result.set_element(i, k, j, l, value)
    return result


class FCHKFile(object):
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
                            except (ValueError, OverflowError), e:
                                raise IOError('Could not interpret word while reading %s: %s' % (word, filename))
                            counter += 1
                except ValueError:
                    return True
            else:
                raise IOError("Unexpected line in formatted checkpoint file %s\n%s" % (filename, line[:-1]))

            self.fields[label] = value
            return True

        self.fields = {}
        f = file(filename, 'r')
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


def load_fchk(filename, lf):
    '''Load from a formatted checkpoint file.

       **Arguments:**

       filename
            The filename of the Gaussian formatted checkpoint file.

       lf
            A LinalgFactory instance.

       **Returns** a dictionary with: ``coordinates``, ``numbers``, ``obasis``,
       ``wfn``, ``permutation``, ``energy``, ``pseudo_numbers``,
       ``mulliken_charges``. Optionally, the dictionary may also contain:
       ``npa_charges`` and/or ``esp_charges``.
    '''
    from horton.gbasis import GOBasis

    fchk = FCHKFile(filename, [
        "Number of electrons", "Number of independant functions",
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
        'Mulliken Charges', 'ESP Charges','NPA Charges',
    ])

    # A) Load the geometry
    numbers = fchk.fields["Atomic numbers"]
    coordinates = fchk.fields["Current cartesian coordinates"].reshape(-1,3)
    pseudo_numbers = fchk.fields["Nuclear charges"]
    # Mask out ghost atoms
    mask = pseudo_numbers != 0.0
    numbers = numbers[mask]
    # Do not overwrite coordinates array, because it is needed to specify basis
    system_coordinates = coordinates[mask]
    pseudo_numbers = pseudo_numbers[mask]

    # B) Load the orbital basis set
    shell_types = fchk.fields["Shell types"]
    shell_map = fchk.fields["Shell to atom map"] - 1
    nprims = fchk.fields["Number of primitives per shell"]
    alphas = fchk.fields["Primitive exponents"]
    ccoeffs_level1 = fchk.fields["Contraction coefficients"]
    ccoeffs_level2 = fchk.fields.get("P(S=P) Contraction coefficients")

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
            my_alphas.append(alphas[counter:counter+n])
            my_alphas.append(alphas[counter:counter+n])
            con_coeffs.append(ccoeffs_level1[counter:counter+n])
            con_coeffs.append(ccoeffs_level2[counter:counter+n])
        else:
            my_shell_types.append(shell_types[i])
            my_shell_map.append(shell_map[i])
            my_nprims.append(nprims[i])
            my_alphas.append(alphas[counter:counter+n])
            con_coeffs.append(ccoeffs_level1[counter:counter+n])
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

    obasis = GOBasis(coordinates, my_shell_map, my_nprims, my_shell_types, my_alphas, con_coeffs)

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
        permutation.extend(permutation_rules[shell_type]+len(permutation))
    permutation = np.array(permutation, dtype=int)

    # C) Load density matrices
    def load_dm(label):
        if label in fchk.fields:
            dm = lf.create_one_body(obasis.nbasis)
            start = 0
            for i in xrange(obasis.nbasis):
                stop = start+i+1
                dm._array[i,:i+1] = fchk.fields[label][start:stop]
                dm._array[:i+1,i] = fchk.fields[label][start:stop]
                start = stop
            return dm

    # First try to load the post-hf density matrices.
    load_orbitals = True
    for key in 'MP2', 'MP3', 'CC', 'CI', 'SCF':
        dm_full = load_dm('Total %s Density' % key)
        dm_spin = load_dm('Spin %s Density' % key)
        if dm_full is not None and key != 'SCF':
            load_orbitals = False
            break

    # D) Load the wavefunction
    # Handle small difference in fchk files from g03 and g09
    nbasis_indep = fchk.fields.get("Number of independant functions") or \
                   fchk.fields.get("Number of independent functions")
    if nbasis_indep is None:
        nbasis_indep = obasis.nbasis

    # Create wfn object
    if 'Beta Orbital Energies' in fchk.fields:
        nalpha = fchk.fields['Number of alpha electrons']
        nbeta = fchk.fields['Number of beta electrons']
        if nalpha < 0 or nbeta < 0 or nalpha+nbeta <= 0:
            raise ValueError('The file %s does not contain a positive number of electrons.' % filename)
        occ_model = AufbauOccModel(nalpha, nbeta) if load_orbitals else None
        wfn = UnrestrictedWFN(lf, obasis.nbasis, occ_model, norb=nbasis_indep)
        if load_orbitals:
            exp_alpha = wfn.init_exp('alpha')
            exp_alpha.coeffs[:] = fchk.fields['Alpha MO coefficients'].reshape(nbasis_indep, obasis.nbasis).T
            exp_alpha.energies[:] = fchk.fields['Alpha Orbital Energies']
            exp_beta = wfn.init_exp('beta')
            exp_beta.coeffs[:] = fchk.fields['Beta MO coefficients'].reshape(nbasis_indep, obasis.nbasis).T
            exp_beta.energies[:] = fchk.fields['Beta Orbital Energies']
            occ_model.assign(exp_alpha, exp_beta)
    else:
        nelec = fchk.fields["Number of electrons"]
        if nelec <= 0:
            raise ValueError('The file %s does not contain a positive number of electrons.' % filename)
        assert nelec % 2 == 0
        occ_model = AufbauOccModel(nelec/2) if load_orbitals else None
        wfn = RestrictedWFN(lf, obasis.nbasis, occ_model, norb=nbasis_indep)
        if load_orbitals:
            exp_alpha = wfn.init_exp('alpha')
            exp_alpha.coeffs[:] = fchk.fields['Alpha MO coefficients'].reshape(nbasis_indep, obasis.nbasis).T
            exp_alpha.energies[:] = fchk.fields['Alpha Orbital Energies']
            occ_model.assign(exp_alpha)

    # Store the density matrices
    if dm_full is not None:
        wfn.update_dm('full', dm_full)
    if dm_spin is not None:
        wfn.update_dm('spin', dm_spin)

    # E) Load properties
    energy = fchk.fields['Total Energy']

    result = {
        'coordinates': system_coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'wfn': wfn,
        'permutation': permutation,
        'energy': energy,
        'pseudo_numbers': pseudo_numbers,
    }

    # F) Load optional properties
    # Mask out ghost atoms from charges
    if 'Mulliken Charges' in fchk.fields:
        result['mulliken_charges'] = fchk.fields['Mulliken Charges'][mask]
    if 'ESP Charges' in fchk.fields:
        result['esp_charges'] = fchk.fields['ESP Charges'][mask]
    if 'NPA Charges' in fchk.fields:
        result['npa_charges'] = fchk.fields['NPA Charges'][mask]

    return result
