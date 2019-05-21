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
"""Module for handling Qchem file format."""
# pragma pylint: disable=invalid-name, too-many-branches, too-many-statements

import re

# from typing import Tuple, List
from typing import Tuple

import numpy as np

# from ..basis import MolecularBasis, Shell
# from ..overlap import gob_cart_normalization
from ..periodic import sym2num
from ..orbitals import MolecularOrbitals
from ..utils import LineIterator, angstrom, amu, calorie, avogadro

__all__ = []

patterns = ['*.qchem']


def load_qchem_low(lit: LineIterator, lit_hess: LineIterator = None) -> Tuple:
    """Load data from a qchem file into arrays.

    Parameters
    ----------
    lit : LineIterator
        The line iterator to read the data from qchem output.
    lit_hess : LineIterator, optional
        The line iterator to read the athessian data.

    """
    # read sections of qchem file
    # title
    for line in lit:
        if line.strip().startswith('$comment'):
            break
    title = next(lit).strip()

    # exchange and basis
    for line in lit:
        if line.strip().startswith('jobtype'):
            runtype = line.strip().split()[1].lower()
        if line.strip().startswith('EXCHANGE'):
            exchange = line.strip().split()[1]
        if line.strip().startswith('BASIS'):
            basis = line.strip().split()[1]
        # get atom coordinates
        if line.strip().startswith(
                'Standard Nuclear Orientation (Angstroms)'):
            break
    next(lit)
    next(lit)
    atcoords = []
    atnames = []
    for line in lit:
        if line.strip().startswith('----'):
            break
        words = line.split()
        atnames.append(words[1])
        coor = [float(words[2]), float(words[3]), float(words[4])]
        atcoords.append(coor)
    atnames = np.array(atnames, dtype=np.unicode_)
    atcoords = np.array(atcoords, dtype=np.float) * angstrom
    num_atoms = atcoords.shape[0]

    atomic_num = np.array([sym2num[i] for i in atnames], dtype=np.int)

    # nuclear repulsion energy in hartrees
    for line in lit:
        if line.strip().startswith('Nuclear Repulsion Energy'):
            break
    nucl_repul_energy = np.array(line.split()[4], dtype=np.float)
    # electrons
    for line in lit:
        if line.strip().startswith('There are') and line.strip().endswith('electrons'):
            break
    num_alpha_electron = np.array(line.split()[2], dtype=np.int)
    num_beta_electron = np.array(line.split()[5], dtype=np.int)

    # grep the SCF energy
    energy = None
    for line in lit:
        if line.strip().endswith('met'):
            # in hartree
            energy = np.array(line.split()[1], dtype=np.float)
            break
    # MO orbital energy
    mo_energy = []
    for line in lit:
        if line.strip().startswith('Alpha MOs'):
            break
    for line in lit:
        energy_tmp = re.findall(r'[-+]?\d+.\d+', line)
        mo_energy.extend(energy_tmp)
        if line.strip().startswith('Ground-State Mulliken Net Atomic Charges'):
            break
    mo_energy = [float(i) for i in mo_energy]
    alpha_mo_energy = np.array(mo_energy[:(len(mo_energy) // 2)], dtype=np.float)
    beta_mo_energy = np.array(mo_energy[(len(mo_energy) // 2):], dtype=np.float)

    # Ground-state Mulliken net atomic charges
    mulliken_charges = []

    for line in lit:
        if line.strip().startswith('Atom                 Charge (a.u.)'):
            break
    next(lit)
    for line in lit:
        if line.strip().startswith('---'):
            break
        else:
            mulliken_charges.append(line.split()[2])

    mulliken_charges = np.array(mulliken_charges, dtype=np.float)
    # atomic mol_charges
    for line in lit:
        if line.strip().startswith('Sum of atomic charges'):
            break
    mol_charges = np.array(line.strip().split()[-1], dtype=np.float)

    # Polarizability matrix in a.u.
    polar_matrix = []
    for line in lit:
        if line.strip().startswith('Polarizability Matrix (a.u.)'):
            break
    next(lit)
    for line in lit:
        if not line.strip().startswith('i: 0 AtomOff'):
            polar_matrix.append(line.strip().split()[1:])
        else:
            break
    polar_matrix = np.array(polar_matrix, dtype=np.float)

    # Get Hessian of the SCF energy
    athessian = np.zeros((3 * num_atoms, 3 * num_atoms), float)
    if lit_hess is None:
        for line in lit:
            if line.strip().startswith(
                    'Hessian of the SCF Energy') or \
                    line.strip().startswith('Final Hessian'):
                break
        nb = int(np.ceil(num_atoms * 3 / 6))
        for i in range(nb):
            next(lit)
            row = 0
            for line in lit:
                words = line.split()
                # / angstrom**2
                athessian[row, 6 * i:6 * (i + 1)] = np.array(
                    sum([[float(word)] for word in words[1:]], []))
                row += 1
                if row >= 3 * num_atoms:
                    break
    # or get Hessian from other file
    else:
        row = 0
        col = 0
        for line in lit_hess:
            athessian[row, col] = float(line.split()[0]) * 1000 * calorie / avogadro / angstrom ** 2
            col += 1
            if col >= 3 * num_atoms:
                row += 1
                col = row
        for i in range(len(lit_hess)):
            for j in range(0, i):
                athessian[i, j] = athessian[j, i]

    # get masses in atomic units
    masses = np.zeros(num_atoms, np.float)
    for line in lit:
        # Vibrational energy
        if line.strip().startswith('Zero point vibrational energy'):
            vib_energy = np.array(line.strip().split()[-2],
                                  dtype=np.float)
            break
    next(lit)
    count = 0
    for line in lit:
        masses[count] = float(line.split()[-1]) * amu
        count += 1
        if count >= num_atoms:
            break

    # get symmetry number
    for line in lit:
        if line.strip().startswith('Rotational Symmetry Number is'):
            num_sym = np.array(line.split()[-1], dtype=np.int)
        # Enthalpy in kcal/mol
        if line.strip().startswith('Total Enthalpy'):
            enthalpy = np.array(line.strip().split()[-2], dtype=np.float)
        # Entropy in cal/mol.K
        if line.strip().startswith('Total Entropy'):
            entropy = np.array(line.strip().split()[-2], dtype=np.float)
            break

    return title, runtype, basis, exchange, atnames, atomic_num, num_atoms, num_sym, \
        masses, atcoords, polar_matrix, athessian, nucl_repul_energy, \
        num_alpha_electron, num_beta_electron, mol_charges, \
        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
        vib_energy, enthalpy, entropy
