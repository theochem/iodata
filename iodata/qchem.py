# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --
# pragma pylint: disable=invalid-name, too-many-branches, too-many-statements
"""Module for handling Q-Chem computation output."""

import re

from typing import Tuple

import numpy as np

from .periodic import sym2num
from .utils import angstrom, amu, calorie, avogadro

__all__ = ['load_qchem_low']

patterns = ['*.freq.out']


def load_qchem_low(filename: str, hessfile: str = None) -> Tuple:
    """Load a molecule from a Q-Chem frequency output."""
    with open(filename) as f:
        # f = open(filename)
        for line in f:
            if line.strip().startswith('$comment'):
                break
        title = next(f).strip()

        for line in f:
            if line.strip().startswith('EXCHANGE'):
                exchange = line.strip().split()[1]
            if line.strip().startswith('BASIS'):
                basis = line.strip().split()[1]
            # get coordinates
            if line.strip().startswith(
                    'Standard Nuclear Orientation (Angstroms)'):
                break
        next(f)
        next(f)
        coordinates = []
        atom_names = []
        for line in f:
            if line.strip().startswith('----'):
                break
            words = line.split()
            atom_names.append(words[1])
            coor = [float(words[2]), float(words[3]), float(words[4])]
            coordinates.append(coor)
        atom_names = np.array(atom_names, dtype=np.unicode_)
        coordinates = np.array(coordinates, dtype=np.float) * angstrom
        num_atoms = coordinates.shape[0]

        atomic_num = np.array([sym2num[i] for i in atom_names], dtype=np.int)

        # nuclear repulsion energy in hartrees
        for line in f:
            if line.strip().startswith('Nuclear Repulsion Energy'):
                break
        nucl_repul_energy = np.array(line.split()[4], dtype=np.float)
        # electrons
        for line in f:
            if line.strip().startswith('There are') and \
                    line.strip().endswith('electrons'):
                break
        num_alpha_electron = np.array(line.split()[2], dtype=np.int)
        num_beta_electron = np.array(line.split()[5], dtype=np.int)

        # grep the SCF energy
        energy = None
        for line in f:
            if line.strip().endswith('met'):
                # in hartree
                energy = np.array(line.split()[1], dtype=np.float)
                break
        # MO orbital energy
        mo_energy = []
        for line in f:
            if line.strip().startswith('Alpha MOs'):
                break
        for line in f:
            energy_tmp = re.findall(r'[-+]?\d+.\d+', line)
            mo_energy.extend(energy_tmp)
            if line.strip().startswith(
                    'Ground-State Mulliken Net Atomic Charges'):
                break
        mo_energy = [float(i) for i in mo_energy]
        alpha_mo_energy = np.array(mo_energy[:(len(mo_energy) // 2)],
                                   dtype=np.float)
        beta_mo_energy = np.array(mo_energy[(len(mo_energy) // 2):],
                                  dtype=np.float)

        # Ground-state Mulliken net atomic charges
        mulliken_charges = []

        for line in f:
            if line.strip().startswith(
                    'Atom                 Charge (a.u.)'):
                break
        next(f)
        for line in f:
            if line.strip().startswith('---'):
                break
            else:
                mulliken_charges.append(line.split()[2])

        mulliken_charges = np.array(mulliken_charges, dtype=np.float)
        # atomic mol_charges
        for line in f:
            if line.strip().startswith('Sum of atomic charges'):
                break
        mol_charges = np.array(line.strip().split()[-1], dtype=np.float)

        # Polarizability matrix in a.u.
        polar_matrix = []
        for line in f:
            if line.strip().startswith('Polarizability Matrix (a.u.)'):
                break
        next(f)
        for line in f:
            if not line.strip().startswith('i: 0 AtomOff'):
                polar_matrix.append(line.strip().split()[1:])
            else:
                break
        polar_matrix = np.array(polar_matrix, dtype=np.float)

        # Get Hessian of the SCF energy
        hessian = np.zeros((3 * num_atoms, 3 * num_atoms), float)
        if hessfile is None:
            for line in f:
                if line.strip().startswith(
                        'Hessian of the SCF Energy') or \
                        line.strip().startswith('Final Hessian'):
                    break
            nb = int(np.ceil(num_atoms * 3 / 6))
            for i in range(nb):
                next(f)
                row = 0
                for line in f:
                    words = line.split()
                    # / angstrom**2
                    hessian[row, 6 * i:6 * (i + 1)] = np.array(
                        sum([[float(word)] for word in words[1:]], []))
                    row += 1
                    if row >= 3 * num_atoms:
                        break

        # get masses in atomic units
        masses = np.zeros(num_atoms, np.float)
        for line in f:
            # Vibrational energy
            if line.strip().startswith('Zero point vibrational energy'):
                vib_energy = np.array(line.strip().split()[-2],
                                      dtype=np.float)
                break
        next(f)
        count = 0
        for line in f:
            masses[count] = float(line.split()[-1]) * amu
            count += 1
            if count >= num_atoms:
                break

        # get symmetry number
        for line in f:
            if line.strip().startswith('Rotational Symmetry Number is'):
                num_sym = np.array(line.split()[-1], dtype=np.int)
            # Enthalply in kcal/mol
            if line.strip().startswith('Total Enthalpy'):
                enthalply = np.array(line.strip().split()[-2], dtype=np.float)
            # Entropy in cal/mol.K
            if line.strip().startswith('Total Entropy'):
                entropy = np.array(line.strip().split()[-2], dtype=np.float)
                break

        # or get Hessian from other file
        if hessfile is not None:
            f = open(hessfile, 'r')
            row = 0
            col = 0
            for line in f:
                hessian[row, col] = float(
                    line.split()[0]) * 1000 * calorie / avogadro / angstrom ** 2
                col += 1
                if col >= 3 * num_atoms:
                    row += 1
                    col = row
            f.close()
            for i in range(len(hessian)):
                for j in range(0, i):
                    hessian[i, j] = hessian[j, i]

    return title, basis, exchange, atom_names, atomic_num, num_atoms, num_sym, \
        masses, coordinates, polar_matrix, hessian, nucl_repul_energy, \
        num_alpha_electron, num_beta_electron, mol_charges, \
        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
        vib_energy, enthalply, entropy
