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


import numpy as np

from horton.io.common import renorm_helper, get_orca_signs
from horton.meanfield.wfn import AufbauOccModel, ClosedShellWFN, OpenShellWFN


__all__ = ['load_molden']


def load_molden(filename, lf):
    '''Load data from a molden input file (with ORCA sign conventions).

       **Arguments:**

       filename
            The filename of the molden input file.

       lf
            A LinalgFactory instance.
    '''

    def helper_coordinates(f):
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
        from horton.gbasis.io import str_to_shell_types
        from horton.gbasis.gobasis import GOBasis
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
        raise IOError('Coordinates not found in mkl file.')
    if obasis is None:
        raise IOError('Orbital basis not found in mkl file.')
    if coeff_alpha is None:
        raise IOError('Alpha orbitals not found in mkl file.')

    if coeff_beta is None:
        nalpha = int(np.round(occ_alpha.sum()))/2
        occ_model = AufbauOccModel(nalpha)
        wfn = ClosedShellWFN(occ_model, lf, obasis.nbasis, norb=coeff_alpha.shape[1])
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
        occ_model = AufbauOccModel(nalpha, nbeta)
        wfn = OpenShellWFN(occ_model, lf, obasis.nbasis, norb=coeff_alpha.shape[1])
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
