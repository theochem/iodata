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
"""WFN File format (Gaussian and GAMESS)"""
import numpy as np

from .overlap import init_scales
from .periodic import sym2num

__all__ = ['load_wfn_low', 'get_permutation_orbital',
           'get_permutation_basis', 'get_mask', 'load_wfn']


def load_wfn_low(filename):
    """Load data from a WFN file into arrays.

       **Arguments:**

       filename
            The filename of the wfn file.
    """

    def helper_num(f):
        """Read number of orbitals, primitives and atoms."""
        line = f.readline()
        assert line.startswith('GAUSSIAN')
        return [int(i) for i in line.split() if i.isdigit()]

    def helper_coordinates(f):
        """Read the coordiantes of the atoms."""
        numbers = np.empty(num_atoms, int)
        coordinates = np.empty((num_atoms, 3), float)
        for atom in range(num_atoms):
            line = f.readline()
            line = line.split()
            numbers[atom] = sym2num[line[0].title()]
            coordinates[atom, :] = [line[4], line[5], line[6]]
        return numbers, coordinates

    def helper_section(f, start, skip):
        """Read CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS sections."""
        section = []
        while len(section) < num_primitives:
            line = f.readline()
            assert line.startswith(start)
            line = line.split()
            section.extend(line[skip:])
        assert len(section) == num_primitives
        return section

    def helper_mo(f):
        """Read one section of MO information."""
        line = f.readline()
        assert line.startswith('MO')
        line = line.split()
        count = line[1]
        occ, energy = line[-5], line[-1]
        coeffs = helper_section(f, ' ', 0)
        coeffs = [i.replace('D', 'E') for i in coeffs]
        return count, occ, energy, coeffs

    def helper_energy(f):
        """Read energy."""
        line = f.readline().lower()
        while 'energy' not in line and line is not None:
            line = f.readline().lower()
        energy = float(line.split('energy =')[1].split()[0])
        return energy

    # read sections of wfn file
    with open(filename) as f:
        title = f.readline().strip()
        num_mo, num_primitives, num_atoms = helper_num(f)
        numbers, coordinates = helper_coordinates(f)
        # centers are indexed from zero in HORTON
        centers = np.array([int(i) - 1 for i in helper_section(f, 'CENTRE ASSIGNMENTS', 2)])
        type_assignment = np.array([int(i) for i in helper_section(f, 'TYPE ASSIGNMENTS', 2)])
        exponent = np.array([float(i.replace('D', 'E')) for i in helper_section(f, 'EXPONENTS', 1)])
        mo_count = np.empty(num_mo, int)
        mo_occ = np.empty(num_mo, float)
        mo_energy = np.empty(num_mo, float)
        coefficients = np.empty([num_primitives, num_mo], float)
        for mo in range(num_mo):
            mo_count[mo], mo_occ[mo], mo_energy[mo], coefficients[:, mo] = helper_mo(f)
        energy = helper_energy(f)
    return title, numbers, coordinates, centers, type_assignment, exponent, \
           mo_count, mo_occ, mo_energy, coefficients, energy


def get_permutation_orbital(type_assignment):
    """Permute each type of orbital to get the proper order for HORTON."""
    num_primitive = len(type_assignment)
    permutation = np.arange(num_primitive)
    # degeneracy of {s:1, p:3, d:6, f:10, g:15, h:21}
    degeneracy = {1: 1, 2: 3, 5: 6, 11: 10, 23: 15, 36: 21}
    index = 0
    while index < num_primitive:
        value = type_assignment[index]
        length = degeneracy[value]
        if value != 1 and value == type_assignment[index + 1]:
            sub_count = 1
            while index + sub_count < num_primitive and type_assignment[index + sub_count] == value:
                sub_count += 1
            sub_type = np.empty(sub_count, int)
            sub_type[:] = permutation[index: index + sub_count]
            for i in range(sub_count):
                permutation[index: index + length] = sub_type[i] + np.arange(length) * sub_count
                index += length
        else:
            index += length
    assert (np.sort(permutation) == np.arange(num_primitive)).all()
    return permutation


def get_permutation_basis(type_assignment):
    """
    Permute the basis functions to get the proper order for HORTON.

    Permutation conventions are as follows:

     d orbitals:
       wfn:     [5, 6, 7, 8, 9, 10]
       HORTON:  [5, 8, 9, 6, 10, 7]
       permute: [0, 3, 4, 1, 5, 2]

     f orbitals:
       wfn:     [11, 12, 13, 17, 14, 15, 18, 19, 16, 20]
       HORTON:  [11, 14, 15, 17, 20, 18, 12, 16, 19, 13]
       permute: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]

     g orbital:
       wfn:     [23, 29, 32, 27, 22, 28, 35, 34, 26, 31, 33, 30, 25, 24, 21]
       HORTON:  [21, 24, 25, 30, 33, 31, 26, 34, 35, 28, 22, 27, 32, 29, 23]
       permute: [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

     h orbital:
       wfn:     [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
       HORTON:  [56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36]
       permute: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    """
    permutation = get_permutation_orbital(type_assignment)
    type_assignment = type_assignment[permutation]
    for count, value in enumerate(type_assignment):
        if value == 5:
            # d-orbitals
            permute = [0, 3, 4, 1, 5, 2]
            permutation[count: count + 6] = permutation[count: count + 6][permute]
        elif value == 11:
            # f-orbitals
            permute = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
            permutation[count: count + 10] = permutation[count: count + 10][permute]
        elif value == 23:
            # g-orbitals
            permute = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            permutation[count: count + 15] = permutation[count:count + 15][permute]
        elif value == 36:
            # h-orbitals
            permutation[count: count + 21] = permutation[count: count + 21][::-1]
    return permutation


def get_mask(type_assignment):
    """Return array to mask orbital types other than s, 1st_p, 1st_d, 1st_f, 1st_g, 1st_h."""
    # index of [s, 1st_p, 1st_d, 1st_f, 1st_g, 1st_h]
    temp = [1, 2, 5, 11, 21, 36]
    mask = np.array([i in temp for i in type_assignment])
    return mask


def load_wfn(filename):
    """Load data from a WFN file.

    Parameters
    ----------
    filename : str
        The filename of the wfn file.

    Returns
    -------
    results : dict
        Data loaded from file, with keys ``title``, ``coordinates``, ``numbers``,
        ``energy``, ``obasis`` and ``orb_alpha``. May contain ``orb_beta``.
    """
    title, numbers, coordinates, centers, type_assignment, exponents, \
    mo_count, mo_occ, mo_energy, coefficients, energy = load_wfn_low(filename)
    permutation = get_permutation_basis(type_assignment)
    # permute arrays containing wfn data
    type_assignment = type_assignment[permutation]
    mask = get_mask(type_assignment)
    reduced_size = np.array(mask, int).sum()
    num_mo = coefficients.shape[1]
    alphas = np.empty(reduced_size)
    alphas[:] = exponents[permutation][mask]
    assert (centers == centers[permutation]).all()
    shell_map = centers[mask]
    # cartesian basis: {S:0, P:1, D:2, F:3, G:4, H:5}
    shell = {1: 0, 2: 1, 5: 2, 11: 3, 21: 4, 36: 5}
    shell_types = type_assignment[mask]
    shell_types = np.array([shell[i] for i in shell_types])
    assert shell_map.size == shell_types.size == reduced_size
    nprims = np.ones(reduced_size, int)
    con_coeffs = np.ones(reduced_size)
    # build basis set
    obasis = {"centers": coordinates, "shell_map": shell_map, "nprims": nprims,
              "shell_types": shell_types, "alphas": alphas, "con_coeffs": con_coeffs}
    nbasis = coefficients.shape[0]
    coefficients = coefficients[permutation]
    scales, dummy = init_scales(obasis["alphas"], obasis["nprims"], obasis["shell_types"])
    coefficients /= scales.reshape(-1, 1)
    # make the wavefunction
    if mo_occ.max() > 1.0:
        # close shell system
        orb_alpha = (nbasis, coefficients.shape[1])
        orb_alpha_coeffs = coefficients
        orb_alpha_energies = mo_energy
        orb_alpha_occs = mo_occ / 2
        orb_beta = None
    else:
        # open shell system
        # counting the number of alpha and beta orbitals
        index = 1
        while index < num_mo and mo_energy[index] >= mo_energy[index - 1] and mo_count[index] == \
                        mo_count[index - 1] + 1:
            index += 1
        orb_alpha = (nbasis, index)
        orb_alpha_coeffs = np.copy(coefficients[:, :index])
        orb_alpha_energies = np.copy(mo_energy[:index])
        orb_alpha_occs = np.copy(mo_occ[:index])
        orb_beta = (nbasis, num_mo - index)
        orb_beta_coeffs = np.copy(coefficients[:, index:])
        orb_beta_energies = np.copy(mo_energy[index:])
        orb_beta_occs = np.copy(mo_occ[index:])

    result = {
        'title': title,
        'coordinates': coordinates,
        'orb_alpha': orb_alpha,
        'orb_alpha_coeffs': orb_alpha_coeffs,
        'orb_alpha_energies': orb_alpha_energies,
        'orb_alpha_occs': orb_alpha_occs,
        'numbers': numbers,
        'obasis': obasis,
        'energy': energy,
    }
    if orb_beta is not None:
        result['orb_beta'] = orb_beta
        result['orb_beta_coeffs'] = orb_beta_coeffs
        result['orb_beta_energies'] = orb_beta_energies
        result['orb_beta_occs'] = orb_beta_occs
    return result
