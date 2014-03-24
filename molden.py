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
'''Molden Wavefunction Input File Format'''


import numpy as np

from horton.periodic import periodic
from horton.gbasis.io import str_to_shell_types, shell_type_to_str
from horton.gbasis.gobasis import GOBasis
from horton.io.common import renorm_helper, get_orca_signs, typecheck_dump
from horton.meanfield.wfn import RestrictedWFN, UnrestrictedWFN


__all__ = ['load_molden', 'dump_molden']


def load_molden(filename, lf):
    '''Load data from a molden input file (with ORCA sign conventions).

       **Arguments:**

       filename
            The filename of the molden input file.

       lf
            A LinalgFactory instance.

       **Returns:** a dictionary with: ``coordinates``, ``numbers``, ``obasis``,
       ``wfn``, ``signs``.
    '''

    def helper_coordinates(f):
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
        coordinates = np.array(coordinates)
        return numbers, coordinates


    def helper_obasis(f, coordinates):
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
                # always assume pure basis functions
                shell_type = str_to_shell_types(words[0], True)[0]
                shell_types.append(shell_type)
                nprims.append(int(words[1]))
            elif len(words) == 2 and in_atom:
                assert in_shell
                alpha = float(words[0])
                alphas.append(alpha)
                con_coeff = renorm_helper(float(words[1]), alpha, shell_type)
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

        icoeff = -1
        while True:
            line = f.readline()
            if icoeff == nbasis:
                icoeff = -1
            if len(line) == 0:
                break
            elif icoeff == -1:
                # read 1a line
                if line != ' Sym=     1a\n':
                    raise IOError('Symmetry in wavefunctions is not supported.')
                # prepare array with orbital coefficients
                col = np.zeros((nbasis,1), float)
                icoeff = -2
            elif icoeff == -2:
                # read energy
                assert line.startswith(' Ene=')
                energy = float(line[5:])
                icoeff = -3
            elif icoeff == -3:
                # read expansion coefficients
                assert line.startswith(' Spin=')
                spin = line[6:].strip()
                if spin == 'Alpha':
                    coeff_alpha.append(col)
                    ener_alpha.append(energy)
                else:
                    coeff_beta.append(col)
                    ener_beta.append(energy)
                icoeff = -4
            elif icoeff == -4:
                assert line.startswith(' Occup=')
                occ = float(line[7:])
                if spin == 'Alpha':
                    occ_alpha.append(occ)
                else:
                    occ_beta.append(occ)
                icoeff = 0
            elif icoeff >= 0:
                words = line.split()
                col[icoeff] = float(words[1])
                icoeff+=1
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

    numbers = None
    coordinates = None
    obasis = None
    coeff_alpha = None
    ener_alpha = None
    occ_alpha = None
    coeff_beta = None
    ener_beta = None
    occ_beta = None
    with open(filename) as f:
        line = f.readline()
        if line != '[Molden Format]\n':
            raise IOError('Molden header not found')
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if line == '[Atoms] AU':
                numbers, coordinates = helper_coordinates(f)
            elif line == '[GTO]':
                obasis = helper_obasis(f, coordinates)
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

    lf.set_default_nbasis(obasis.nbasis)
    if coeff_beta is None:
        nalpha = int(np.round(occ_alpha.sum()))/2
        wfn = RestrictedWFN(lf, obasis.nbasis, norb=coeff_alpha.shape[1])
        exp_alpha = wfn.init_exp('alpha')
        exp_alpha.coeffs[:] = coeff_alpha
        exp_alpha.energies[:] = ener_alpha
        exp_alpha.occupations[:] = occ_alpha/2
    else:
        nalpha = int(np.round(occ_alpha.sum()))
        nbeta = int(np.round(occ_beta.sum()))
        assert coeff_alpha.shape == coeff_beta.shape
        assert ener_alpha.shape == ener_beta.shape
        assert occ_alpha.shape == occ_beta.shape
        wfn = UnrestrictedWFN(lf, obasis.nbasis, norb=coeff_alpha.shape[1])
        exp_alpha = wfn.init_exp('alpha')
        exp_alpha.coeffs[:] = coeff_alpha
        exp_alpha.energies[:] = ener_alpha
        exp_alpha.occupations[:] = occ_alpha
        exp_beta = wfn.init_exp('beta')
        exp_beta.coeffs[:] = coeff_beta
        exp_beta.energies[:] = ener_beta
        exp_beta.occupations[:] = occ_beta

    signs = get_orca_signs(obasis)

    return {
        'coordinates': coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'wfn': wfn,
        'signs': signs,
    }


def dump_molden(filename, data):
    '''Write molecule data to a file in the molden input format.

       **Arguments:**

       filename
            The filename of the molden input file, which is an output file for
            this routine.

       data
            A dictionary with molecule that needs to be written. Must contain
            ``coordinates``, ``numbers``, ``obasis``, ``wfn``.
    '''
    coordinates, numbers, obasis, wfn = typecheck_dump(data, ['coordinates', 'numbers', 'obasis', 'wfn'])
    natom = len(numbers)
    with open(filename, 'w') as f:
        # Print the header
        print >> f, '[Molden Format]'
        print >> f, '[Title]'
        print >> f, ' File created with Horton'
        print >> f

        # Print the elements numbers and the coordinates
        print >> f, '[Atoms] AU'
        for i in xrange(natom):
            number = numbers[i]
            x, y, z = coordinates[i]
            print >> f, '%2s %3i %3i  %20.10f %20.10f %20.10f' % (
                periodic[number].symbol.ljust(2), i+1, number, x, y, z
            )

        # Print the basis set
        if isinstance(obasis, GOBasis):
            if obasis.shell_types.max() > 1:
                raise ValueError('Only pure Gaussian basis functions are supported in dump_molden.')
            # first convert it to a format that is amenable for printing.
            centers = [list() for i in xrange(obasis.ncenter)]
            begin_prim = 0
            for ishell in xrange(obasis.nshell):
                icenter = obasis.shell_map[ishell]
                shell_type = obasis.shell_types[ishell]
                sts = shell_type_to_str(shell_type)
                end_prim = begin_prim + obasis.nprims[ishell]
                prims = []
                for iprim in xrange(begin_prim, end_prim):
                    alpha = obasis.alphas[iprim]
                    con_coeff = renorm_helper(obasis.con_coeffs[iprim], alpha, shell_type, reverse=True)
                    prims.append((alpha, con_coeff))
                centers[icenter].append((sts, prims))
                begin_prim = end_prim

            print >> f, '[GTO]'
            for icenter in xrange(obasis.ncenter):
                print >> f, '%3i 0' % (icenter+1)
                for sts, prims in centers[icenter]:
                    print >> f, '%1s %3i 1.0' % (sts, len(prims))
                    for alpha, con_coeff in prims:
                        print >> f, '%20.10f %20.10f' % (alpha, con_coeff)
                print >> f

            # For now, only pure basis functions are supported.
            print >> f, '[5D]'
            print >> f, '[9G]'

            # The sign conventions...
            signs = get_orca_signs(obasis)
        else:
            raise NotImplementedError('A Gaussian orbital basis is required to write a molden input file.')

        def helper_exp(spin, occ_scale=1.0):
            if not 'exp_%s' % spin in wfn.cache:
                raise TypeError('The restricted WFN does not have an expansion of the %s orbitals.' % spin)
            exp = wfn.get_exp(spin)
            for ifn in xrange(exp.nfn):
                print >> f, ' Sym=     1a'
                print >> f, ' Ene= %20.14E' % exp.energies[ifn]
                print >> f, ' Spin= %s' % spin.capitalize()
                print >> f, ' Occup= %8.6f' % (exp.occupations[ifn]*occ_scale)
                for ibasis in xrange(exp.nbasis):
                    print >> f, '%3i %20.12f' % (ibasis+1, exp.coeffs[ibasis,ifn]*signs[ibasis])

        # Print the mean-field orbitals
        if isinstance(wfn, RestrictedWFN):
            print >> f, '[MO]'
            helper_exp('alpha', 2.0)
        elif isinstance(wfn, UnrestrictedWFN):
            print >> f, '[MO]'
            helper_exp('alpha')
            helper_exp('beta')
        else:
            raise NotImplementedError
