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
'''Molekel wavefunction input file format'''


import numpy as np

from horton.units import angstrom
from horton.io.molden import _fix_molden_from_buggy_codes
from horton.gbasis.iobas import str_to_shell_types
from horton.gbasis.gobasis import GOBasis


__all__ = ['load_mkl']


def load_mkl(filename, lf):
    '''Load data from a Molekel file.

       **Arguments:**

       filename
            The filename of the mkl file.

       lf
            A LinalgFactory instance.

       **Returns** a dictionary with: ``coordinates``, ``numbers``, ``obasis``,
       ``exp_alpha``. It may also contain: ``exp_beta``, ``signs``.
    '''

    def helper_char_mult(f):
        return [int(word) for word in f.readline().split()]


    def helper_coordinates(f):
        numbers = []
        coordinates = []
        while True:
            line = f.readline()
            if len(line) == 0 or line.strip() == '$END':
                break
            words = line.split()
            numbers.append(int(words[0]))
            coordinates.append([float(words[1]), float(words[2]), float(words[3])])
        numbers = np.array(numbers, int)
        coordinates = np.array(coordinates)*angstrom
        return numbers, coordinates


    def helper_obasis(f, coordinates):
        shell_types = []
        shell_map = []
        nprims = []
        alphas = []
        con_coeffs = []

        center_counter = 0
        in_shell = False
        nprim = None
        while True:
            line = f.readline()
            lstrip = line.strip()
            if len(line) == 0 or lstrip == '$END':
                break
            if len(lstrip) == 0:
                continue
            if lstrip == '$$':
                center_counter += 1
                in_shell = False
            else:
                words = line.split()
                if len(words) == 2:
                    assert in_shell
                    alpha = float(words[0])
                    alphas.append(alpha)
                    con_coeffs.append(float(words[1]))
                    nprim += 1
                else:
                    if nprim is not None:
                        nprims.append(nprim)
                    shell_map.append(center_counter)
                    # always assume pure basis functions
                    shell_type = str_to_shell_types(words[1], pure=True)[0]
                    shell_types.append(shell_type)
                    in_shell = True
                    nprim = 0
        if nprim is not None:
            nprims.append(nprim)

        shell_map = np.array(shell_map)
        nprims = np.array(nprims)
        shell_types = np.array(shell_types)
        alphas = np.array(alphas)
        con_coeffs = np.array(con_coeffs)
        return GOBasis(coordinates, shell_map, nprims, shell_types, alphas, con_coeffs)


    def helper_coeffs(f, nbasis):
        coeffs = []
        energies = []

        in_orb = 0
        while True:
            line = f.readline()
            lstrip = line.strip()
            if len(line) == 0 or lstrip == '$END':
                break
            if in_orb == 0:
                # read a1g line
                words = lstrip.split()
                ncol = len(words)
                assert ncol > 0
                for word in words:
                    assert word == 'a1g'
                cols = [np.zeros((nbasis,1), float) for icol in xrange(ncol)]
                in_orb = 1
            elif in_orb == 1:
                # read energies
                words = lstrip.split()
                assert len(words) == ncol
                for word in words:
                    energies.append(float(word))
                in_orb = 2
                ibasis = 0
            elif in_orb == 2:
                # read expansion coefficients
                words = lstrip.split()
                assert len(words) == ncol
                for icol in xrange(ncol):
                    cols[icol][ibasis] = float(words[icol])
                ibasis += 1
                if ibasis == nbasis:
                    in_orb = 0
                    coeffs.extend(cols)

        return np.hstack(coeffs), np.array(energies)


    def helper_occ(f):
        occs = []
        while True:
            line = f.readline()
            lstrip = line.strip()
            if len(line) == 0 or lstrip == '$END':
                break
            for word in lstrip.split():
                occs.append(float(word))
        return np.array(occs)


    charge = None
    spinmult = None
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
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if line == '$CHAR_MULT':
                charge, spinmult = helper_char_mult(f)
            elif line == '$COORD':
                numbers, coordinates = helper_coordinates(f)
            elif line == '$BASIS':
                obasis = helper_obasis(f, coordinates)
            elif line == '$COEFF_ALPHA':
                coeff_alpha, ener_alpha = helper_coeffs(f, obasis.nbasis)
            elif line == '$OCC_ALPHA':
                occ_alpha = helper_occ(f)
            elif line == '$COEFF_BETA':
                coeff_beta, ener_beta = helper_coeffs(f, obasis.nbasis)
            elif line == '$OCC_BETA':
                occ_beta = helper_occ(f)

    if charge is None:
        raise IOError('Charge and multiplicity not found in mkl file.')
    if coordinates is None:
        raise IOError('Coordinates not found in mkl file.')
    if obasis is None:
        raise IOError('Orbital basis not found in mkl file.')
    if coeff_alpha is None:
        raise IOError('Alpha orbitals not found in mkl file.')
    if occ_alpha is None:
        raise IOError('Alpha occupation numbers not found in mkl file.')

    if lf.default_nbasis is not None and lf.default_nbasis != obasis.nbasis:
        raise TypeError('The value of lf.default_nbasis does not match nbasis reported in the mkl file.')
    lf.default_nbasis = obasis.nbasis
    nelec = numbers.sum() - charge
    if coeff_beta is None:
        assert nelec % 2 == 0
        assert abs(occ_alpha.sum() - nelec) < 1e-7
        exp_alpha = lf.create_expansion(obasis.nbasis, coeff_alpha.shape[1])
        exp_alpha.coeffs[:] = coeff_alpha
        exp_alpha.energies[:] = ener_alpha
        exp_alpha.occupations[:] = occ_alpha/2
        exp_beta = None
    else:
        if occ_beta is None:
            raise IOError('Beta occupation numbers not found in mkl file while beta orbitals were present.')
        nalpha = int(np.round(occ_alpha.sum()))
        nbeta = int(np.round(occ_beta.sum()))
        assert nelec == nalpha+nbeta
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

    result = {
        'coordinates': coordinates,
        'exp_alpha': exp_alpha,
        'lf': lf,
        'numbers': numbers,
        'obasis': obasis,
    }
    if exp_beta is not None:
        result['exp_beta'] = exp_beta
    _fix_molden_from_buggy_codes(result, filename)
    return result
