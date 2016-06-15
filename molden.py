# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
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
'''Molden wavefunction input file format'''


import numpy as np

from horton.units import angstrom
from horton.log import log
from horton.periodic import periodic
from horton.gbasis.iobas import str_to_shell_types, shell_type_to_str
from horton.gbasis.gobasis import GOBasis
from horton.gbasis.cext import gob_cart_normalization, get_shell_nbasis


__all__ = ['load_molden', 'dump_molden']


def _get_molden_permutation(obasis, reverse=False):
    # Reorder the Cartesian basis functions to obtain the HORTON standard ordering.
    permutation_rules = {
       2: np.array([0, 3, 4, 1, 5, 2]),
       3: np.array([0, 4, 5, 3, 9, 6, 1, 8, 7, 2]),
       4: np.array([0, 3, 4, 9, 12, 10, 13, 5, 1, 6, 11, 14, 7, 8, 2]),
    }
    permutation = []
    for shell_type in obasis.shell_types:
        rule = permutation_rules.get(shell_type)
        if reverse and rule is not None:
            reverse_rule = np.zeros(len(rule), int)
            for i, j in enumerate(rule):
                reverse_rule[j] = i
            rule = reverse_rule
        if rule is None:
            rule = np.arange(get_shell_nbasis(shell_type))
        permutation.extend(rule+len(permutation))
    return np.array(permutation, dtype=int)


def load_molden(filename, lf):
    '''Load data from a molden input file.

       **Arguments:**

       filename
            The filename of the molden input file.

       lf
            A LinalgFactory instance.

       **Returns:** a dictionary with: ``coordinates``, ``numbers``, ``obasis``,
       ``exp_alpha``, ``signs``. It may also contain: ``title``, ``exp_beta``.
    '''

    def helper_coordinates(f, cunit):
        '''Load element numbers and coordinates'''
        numbers = []
        coordinates = []
        while True:
            last_pos = f.tell()
            line = f.readline()
            if len(line) == 0:
                break
            words = line.split()
            if len(words) != 6:
                # Go back to previous line and stop
                f.seek(last_pos)
                break
            else:
                numbers.append(int(words[2]))
                coordinates.append([float(words[3]), float(words[4]), float(words[5])])
        numbers = np.array(numbers, int)
        coordinates = np.array(coordinates)*cunit
        return numbers, coordinates


    def helper_obasis(f, coordinates, pure):
        '''Load the orbital basis'''
        shell_types = []
        shell_map = []
        nprims = []
        alphas = []
        con_coeffs = []

        icenter = 0
        in_atom = False
        in_shell = False
        while True:
            last_pos = f.tell()
            line = f.readline()
            if len(line) == 0:
                break
            words = line.split()
            if len(words) == 0:
                in_atom = False
                in_shell = False
            elif len(words) == 2 and not in_atom:
                icenter = int(words[0])-1
                in_atom = True
                in_shell = False
            elif len(words) == 3:
                in_shell = True
                shell_map.append(icenter)
                shell_label = words[0].lower()
                shell_type = str_to_shell_types(shell_label, pure.get(shell_label, False))[0]
                shell_types.append(shell_type)
                nprims.append(int(words[1]))
            elif len(words) == 2 and in_atom:
                assert in_shell
                alpha = float(words[0].replace('D', 'E'))
                alphas.append(alpha)
                con_coeff = float(words[1].replace('D', 'E'))
                con_coeffs.append(con_coeff)
            else:
                # done, go back one line
                f.seek(last_pos)
                break

        shell_map = np.array(shell_map)
        nprims = np.array(nprims)
        shell_types = np.array(shell_types)
        alphas = np.array(alphas)
        con_coeffs = np.array(con_coeffs)
        return GOBasis(coordinates, shell_map, nprims, shell_types, alphas, con_coeffs)


    def helper_coeffs(f, nbasis):
        '''Load the orbital coefficients'''
        coeff_alpha = []
        ener_alpha = []
        occ_alpha = []
        coeff_beta = []
        ener_beta = []
        occ_beta = []

        new_orb = True
        icoeff = nbasis
        while True:
            line = f.readline()
            if len(line) == 0 or '[' in line:
                break
            # prepare array with orbital coefficients
            if '=' in line:
                if line.startswith(' Ene='):
                    energy = float(line[5:])
                elif line.startswith(' Spin='):
                    spin = line[6:].strip()
                elif line.startswith(' Occup='):
                    occ = float(line[7:])
                new_orb = True
            else:
                if new_orb:
                    # store col, energy and occ
                    col = np.zeros((nbasis, 1))
                    if spin.lower() == 'alpha':
                        coeff_alpha.append(col)
                        ener_alpha.append(energy)
                        occ_alpha.append(occ)
                    else:
                        coeff_beta.append(col)
                        ener_beta.append(energy)
                        occ_beta.append(occ)
                    new_orb = False
                    if icoeff < nbasis:
                        raise IOError('Too little expansions coefficients in one orbital in molden file.')
                    icoeff = 0
                words = line.split()
                if icoeff >= nbasis:
                    raise IOError('Too many expansions coefficients in one orbital in molden file.')
                col[icoeff] = float(words[1])
                icoeff += 1
                assert int(words[0]) == icoeff

        coeff_alpha = np.hstack(coeff_alpha)
        ener_alpha = np.array(ener_alpha)
        occ_alpha = np.array(occ_alpha)
        if len(coeff_beta) == 0:
            coeff_beta = None
            ener_beta = None
            occ_beta = None
        else:
            coeff_beta = np.hstack(coeff_beta)
            ener_beta = np.array(ener_beta)
            occ_beta = np.array(occ_beta)
        return (coeff_alpha, ener_alpha, occ_alpha), (coeff_beta, ener_beta, occ_beta)

    # First pass: scan the file for pure/cartesian modifiers.
    # Unfortunately, some program put this information _AFTER_ the basis
    # set specification.
    pure = {'d': False, 'f': False, 'g': False}
    with open(filename) as f:
        for line in f:
            line = line.lower()
            if line.startswith('[5d]') or line.startswith('[5d7f]'):
                pure['d'] = True
                pure['f'] = True
            elif line.lower().startswith('[7f]'):
                pure['f'] = True
            elif line.lower().startswith('[5d10f]'):
                pure['d'] = True
                pure['f'] = False
            elif line.lower().startswith('[9g]'):
                pure['g'] = True

    # Second pass: read all the other info.
    numbers = None
    coordinates = None
    obasis = None
    coeff_alpha = None
    ener_alpha = None
    occ_alpha = None
    coeff_beta = None
    ener_beta = None
    occ_beta = None
    title = None
    with open(filename) as f:
        line = f.readline()
        if line != '[Molden Format]\n':
            raise IOError('Molden header not found')
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if line == '[Title]':
                title = f.readline().strip()
            elif line.startswith('[Atoms]'):
                if 'au' in line.lower():
                    cunit = 1
                elif 'angs' in line.lower():
                    cunit = angstrom
                numbers, coordinates = helper_coordinates(f, cunit)
            elif line == '[GTO]':
                obasis = helper_obasis(f, coordinates, pure)
            elif line == '[STO]':
                raise NotImplementedError('Slater-type orbitals are not supported in HORTON.')
            elif line == '[MO]':
                data_alpha, data_beta = helper_coeffs(f, obasis.nbasis)
                coeff_alpha, ener_alpha, occ_alpha = data_alpha
                coeff_beta, ener_beta, occ_beta = data_beta

    if coordinates is None:
        raise IOError('Coordinates not found in molden input file.')
    if obasis is None:
        raise IOError('Orbital basis not found in molden input file.')
    if coeff_alpha is None:
        raise IOError('Alpha orbitals not found in molden input file.')

    if lf.default_nbasis is not None and lf.default_nbasis != obasis.nbasis:
        raise TypeError('The value of lf.default_nbasis does not match nbasis reported in the molden.input file.')
    lf.default_nbasis = obasis.nbasis
    if coeff_beta is None:
        nalpha = int(np.round(occ_alpha.sum()))/2
        exp_alpha = lf.create_expansion(obasis.nbasis, coeff_alpha.shape[1])
        exp_alpha.coeffs[:] = coeff_alpha
        exp_alpha.energies[:] = ener_alpha
        exp_alpha.occupations[:] = occ_alpha/2
        exp_beta = None
    else:
        nalpha = int(np.round(occ_alpha.sum()))
        nbeta = int(np.round(occ_beta.sum()))
        assert coeff_alpha.shape == coeff_beta.shape
        assert ener_alpha.shape == ener_beta.shape
        assert occ_alpha.shape == occ_beta.shape
        exp_alpha = lf.create_expansion(obasis.nbasis, coeff_alpha.shape[1])
        exp_alpha.coeffs[:] = coeff_alpha
        exp_alpha.energies[:] = ener_alpha
        exp_alpha.occupations[:] = occ_alpha
        exp_beta = lf.create_expansion(obasis.nbasis, coeff_beta.shape[1])
        exp_beta.coeffs[:] = coeff_beta
        exp_beta.energies[:] = ener_beta
        exp_beta.occupations[:] = occ_beta

    permutation = _get_molden_permutation(obasis)

    result = {
        'coordinates': coordinates,
        'exp_alpha': exp_alpha,
        'lf': lf,
        'numbers': numbers,
        'obasis': obasis,
        'permutation': permutation,
    }
    if title is not None:
        result['title'] = title
    if exp_beta is not None:
        result['exp_beta'] = exp_beta

    _fix_molden_from_buggy_codes(result, filename)

    return result


def _is_normalized_properly(lf, obasis, permutation, exp_alpha, exp_beta, signs=None, threshold=1e-4):
    '''Test the normalization of the occupied and virtual orbitals

       **Arguments:**

       lf
            The linalg factory (needed for computing the overlap matrix).

       obasis
            An instance of the GOBasis class.

       permutation
            The permutation of the basis functions to bring them in HORTON's
            standard ordering.

       exp_alpha
            The alpha orbitals.

       exp_beta
            The beta orbitals (may be None).

       **Optional arguments:**

       signs
            Changes in sign conventions.

       threshold
            When the maximal error on the norm is large than the threshold,
            the function returns False. True is returned otherwise.
    '''
    # Set default value for signs
    if signs is None:
        signs = np.ones(obasis.nbasis, int)
    # Compute the overlap matrix.
    olp = obasis.compute_overlap(lf)
    # Compute the norm of each occupied and virtual orbital. Keep track of
    # the largest deviation from unity
    error_max = 0.0
    for iorb in xrange(exp_alpha.nfn):
        vec = exp_alpha._coeffs[:,iorb]*signs
        if permutation is not None:
            vec = vec[permutation]
        norm = olp.inner(vec, vec)
        error_max = max(error_max, abs(norm-1))
    if exp_beta is not None:
        for iorb in xrange(exp_beta.nfn):
            vec = (exp_beta._coeffs[:,iorb]*signs)
            if permutation is not None:
                vec = vec[permutation]
            norm = olp.inner(vec, vec)
            error_max = max(error_max, abs(norm-1))
    # final judgement
    return error_max <= threshold


def _get_orca_signs(obasis):
    '''Return an array with sign corrections for orbitals read from ORCA.

       **Arguments:**

       obasis
            An instance of GOBasis.
    '''
    sign_rules = {
      -4: [1,1,1,1,1,-1,-1,-1,-1],
      -3: [1,1,1,1,1,-1,-1],
      -2: [1,1,1,1,1],
       0: [1],
       1: [1,1,1],
    }
    signs = []
    for shell_type in obasis.shell_types:
        if shell_type in sign_rules:
            signs.extend(sign_rules[shell_type])
        else:
            signs.extend([1]*get_shell_nbasis(shell_type))
    return np.array(signs, dtype=int)


def _get_fixed_con_coeffs(obasis, mode):
    '''Return corrected contraction coefficients, assuming they came from an
       ORCA/PSI4 molden/mkl file.

       **Arguments:**

       obasis
            An instance of GOBasis.

       mode
            A string, either 'orca' or 'psi4'
    '''
    assert mode in ['orca', 'psi4']
    fixed_con_coeffs = obasis.con_coeffs.copy()
    iprim = 0
    for ishell in xrange(obasis.nshell):
        shell_type = obasis.shell_types[ishell]
        for ialpha in xrange(obasis.nprims[ishell]):
            alpha = obasis.alphas[iprim]
            if shell_type == 0:
                scale = gob_cart_normalization(alpha, np.array([0,0,0]))
            elif shell_type == 1:
                scale = gob_cart_normalization(alpha, np.array([1,0,0]))
            elif shell_type == -2:
                scale = gob_cart_normalization(alpha, np.array([1,1,0]))
                if mode == 'psi4':
                    scale /= np.sqrt(3)
            elif shell_type == -3:
                scale = gob_cart_normalization(alpha, np.array([1,1,1]))
                if mode == 'psi4':
                    scale /= np.sqrt(15)
            elif shell_type == -4:
                scale = gob_cart_normalization(alpha, np.array([2,1,1]))
                if mode == 'psi4':
                    scale /= np.sqrt(105)
            else:
                # Not sure yet what to do, just do something, error will be
                # caught later.
                scale = 1
            fixed_con_coeffs[iprim] /= scale
            iprim += 1
    return fixed_con_coeffs


def _fix_molden_from_buggy_codes(result, filename):
    '''Detect errors in the data loaded from a molden/mkl/... file and correct.

       **Argument:**

       result
            A dictionary with the data loaded in the ``load_molden`` function.

       This function can recognize erroneous files created by PSI4 and ORCA. The
       data in the obasis and signs fields will be updated accordingly.
    '''
    obasis = result['obasis']
    permutation = result.get('permutation', None)
    if _is_normalized_properly(result['lf'], obasis, permutation, result['exp_alpha'], result.get('exp_beta')):
        # The file is good. No need to change data.
        return
    if log.do_medium:
        log('Detected incorrect normalization of orbitals loaded from a file.')
    # Try to fix it as if it was a file generated by ORCA.
    orca_signs = _get_orca_signs(obasis)
    orca_con_coeffs = _get_fixed_con_coeffs(obasis, 'orca')
    orca_obasis = GOBasis(obasis.centers, obasis.shell_map, obasis.nprims, obasis.shell_types, obasis.alphas, orca_con_coeffs)
    if _is_normalized_properly(result['lf'], orca_obasis, permutation, result['exp_alpha'], result.get('exp_beta'), orca_signs):
        if log.do_medium:
            log('Detected typical ORCA errors in file. Fixing them...')
        result['obasis'] = orca_obasis
        result['signs'] = orca_signs
        return
    # Try to fix it as if it was a file generated by PSI4.
    psi4_con_coeffs = _get_fixed_con_coeffs(obasis, 'psi4')
    psi4_obasis = GOBasis(obasis.centers, obasis.shell_map, obasis.nprims, obasis.shell_types, obasis.alphas, psi4_con_coeffs)
    if _is_normalized_properly(result['lf'], psi4_obasis, permutation, result['exp_alpha'], result.get('exp_beta')):
        if log.do_medium:
            log('Detected typical PSI4 errors in file. Fixing them...')
        result['obasis'] = psi4_obasis
        return

    raise IOError(('Could not correct the data read from %s. The molden or '
                   'mkl file you are trying to load contains errors. Please '
                   'report this problem to Toon.Verstraelen@UGent.be, so he '
                   'can fix it.') % filename)


def dump_molden(filename, data):
    '''Write data to a file in the molden input format.

       **Arguments:**

       filename
            The filename of the molden input file, which is an output file for
            this routine.

       data
            An IOData instance. Must contain ``coordinates``, ``numbers``,
            ``obasis``, ``exp_alpha``. May contain ``title``, ``exp_beta``.
    '''
    with open(filename, 'w') as f:
        # Print the header
        print >> f, '[Molden Format]'
        print >> f, '[Title]'
        print >> f, ' %s' % getattr(data, 'title', 'Created with HORTON')
        print >> f

        # Print the elements numbers and the coordinates
        print >> f, '[Atoms] AU'
        for i in xrange(data.natom):
            number = data.numbers[i]
            x, y, z = data.coordinates[i]
            print >> f, '%2s %3i %3i  %25.18f %25.18f %25.18f' % (
                periodic[number].symbol.ljust(2), i+1, number, x, y, z
            )

        # Print the basis set
        if isinstance(data.obasis, GOBasis):
            # Figure out the pure/Cartesian situation. Note that the Molden
            # format doesnot support mixed Cartesian and pure functions in the
            # way HORTON does.In practice, such combinations are too unlikely
            # to be relevant.
            pure = {'d': None, 'f': None, 'g': None}
            try:
                for shell_type in data.obasis.shell_types:
                    if shell_type == 2:
                        assert pure['d'] is None or not pure['d']
                        pure['d'] = False
                    elif shell_type == -2:
                        assert pure['d'] is None or pure['d']
                        pure['d'] = True
                    elif shell_type == 3:
                        assert pure['f'] is None or not pure['f']
                        pure['f'] = False
                    elif shell_type == -3:
                        assert pure['f'] is None or pure['f']
                        pure['f'] = True
                    elif shell_type == 4:
                        assert pure['g'] is None or not pure['g']
                        pure['g'] = False
                    elif shell_type == -4:
                        assert pure['g'] is None or pure['g']
                        pure['g'] = True
                    else:
                        assert abs(shell_type) < 2
            except AssertionError:
                raise IOError('The basis set is not supported by the Molden format.')

            # Write out the Cartesian/Pure conventions. What a messy format...
            if pure['d']:
                if pure['f']:
                    print >> f, '[5D]'
                else:
                    print >> f, '[5D10F]'
            else:
                if pure['f']:
                    print >> f, '[7F]'
            if pure['g']:
                print >> f, '[9G]'

            # first convert it to a format that is amenable for printing.
            centers = [list() for i in xrange(data.obasis.ncenter)]
            begin_prim = 0
            for ishell in xrange(data.obasis.nshell):
                icenter = data.obasis.shell_map[ishell]
                shell_type = data.obasis.shell_types[ishell]
                sts = shell_type_to_str(shell_type)
                end_prim = begin_prim + data.obasis.nprims[ishell]
                prims = []
                for iprim in xrange(begin_prim, end_prim):
                    alpha = data.obasis.alphas[iprim]
                    con_coeff = data.obasis.con_coeffs[iprim]
                    prims.append((alpha, con_coeff))
                centers[icenter].append((sts, prims))
                begin_prim = end_prim

            print >> f, '[GTO]'
            for icenter in xrange(data.obasis.ncenter):
                print >> f, '%3i 0' % (icenter+1)
                for sts, prims in centers[icenter]:
                    print >> f, '%1s %3i 1.0' % (sts, len(prims))
                    for alpha, con_coeff in prims:
                        print >> f, '%20.10f %20.10f' % (alpha, con_coeff)
                print >> f
        else:
            raise NotImplementedError('A Gaussian orbital basis is required to write a molden input file.')

        def helper_exp(spin, occ_scale=1.0):
            exp = getattr(data, 'exp_%s' % spin)
            for ifn in xrange(exp.nfn):
                print >> f, ' Sym=     1a'
                print >> f, ' Ene= %20.14E' % exp.energies[ifn]
                print >> f, ' Spin= %s' % spin.capitalize()
                print >> f, ' Occup= %8.6f' % (exp.occupations[ifn]*occ_scale)
                for ibasis in xrange(exp.nbasis):
                    print >> f, '%3i %20.12f' % (ibasis+1, exp.coeffs[permutation[ibasis],ifn])

        # Construct the permutation of the basis functions
        permutation = _get_molden_permutation(data.obasis, reverse=True)

        # Print the mean-field orbitals
        if hasattr(data, 'exp_beta'):
            print >> f, '[MO]'
            helper_exp('alpha')
            helper_exp('beta')
        else:
            print >> f, '[MO]'
            helper_exp('alpha', 2.0)
