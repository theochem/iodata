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
"""Q-Chem Log file format.

"""
import re

import numpy as np

from ..docstrings import document_load_one
from ..periodic import sym2num
from ..utils import LineIterator

__all__ = []

PATTERNS = ['*.log']


# todo: fix @document_load_one argument with the results from load_one
@document_load_one("Q-Chem Log", [])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    pass


def load_qchemlog_low(lit: LineIterator) -> dict:
    """Load the information from Q-Chem log file."""
    data = {}
    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            # Read until the end of the file.
            break

        # get the atomic information
        if line.startswith('$molecule'):
            data['charge'], spin_multi, data['natom'], \
            atnums, data['atcoords'] = _helper_atoms(lit)
            natoms = len(atnums)
            data['atnum'] = atnums
            # print(lit.lineno)
            # 145
        # job specifications
        elif line.startswith('$rem'):
            data['run_type'], method, basis_set, unrestricted, data['g_rot'] = _helper_job(lit)
            # print(lit.lineno)
            # 166
        # standard nuclear orientation
        elif line.startswith('Standard Nuclear Orientation (Angstroms)'):
            atnums, alpha_elec, beta_elec, nuclear_replusion_energy, energy, atcoords = \
                _helper_electron(lit)
            # print(lit.lineno)
            # 236
        # orbital energies
        elif line.startswith('Orbital Energies (a.u.)'):
            alpha_mo_occupied, beta_mo_occupied, alpha_mo_unoccupied, beta_mo_unoccupied, \
            norba, norbb, energies = _helper_orbital_energies(lit)
            # print(lit.lineno)
            #  266
        # mulliken charges
        elif line.startswith('Ground-State Mulliken Net Atomic Charges'):
            mulliken_charges = _helper_mulliken(lit)
            # print(lit.lineno)
            # 275
        #  cartesian multipole moments
        elif line.startswith('Cartesian Multipole Moments'):
            dipole = _helper_dipole_moments(lit)
            # print(lit.lineno)
            # 286
        # polarizability matrix
        elif line.startswith('Polarizability Matrix (a.u.)'):
            polarizability_tensor = _helper_polar(lit)
            # print(lit.lineno)
            # 325
        # hessian matrix
        elif line.startswith('Hessian of the SCF Energy'):
            hessian = _helper_hessian(lit)
            hessian = hessian.reshape(natoms * 3, natoms * 3)
            # print(lit.lineno)
            # 355
        # vibrational analysis
        elif line.startswith('**                       VIBRATIONAL ANALYSIS'):
            imaginary_freq, vib_energy = _helper_vibrational(lit)
            # print(lit.lineno)
            # 383
        elif line.startswith('Rotational Symmetry Number'):
            enthalpy_dict, entropy_dict = _helper_thermo(lit)
            # print(lit.lineno)
            # 407
    return data


def _helper_atoms(lit: LineIterator) -> tuple:
    """Load list of coordinates from an Q-Chem log output file format."""
    # net charge and spin multiplicity
    charge, spin_multi = [int(i) for i in next(lit).strip().split()]
    natom = 0
    atom_symbols = []
    atcoords = []
    for line in lit:
        if line.strip() == '$end':
            break
        else:
            atom_symbols.append(line.strip().split()[0])
            atcoords.append([float(i) for i in line.strip().split()[1:]])
            natom += 1
    atnums = np.array([sym2num[i] for i in atom_symbols])
    # coordinates in angstroms
    # todo: check should use the 'Standard Nuclear Orientation (Angstrom)' or just here
    atcoords = np.array(atcoords)
    return charge, spin_multi, natom, atnums, atcoords


def _helper_job(lit):
    """Load job specifications from Q-Chem log out file format."""
    for line in lit:
        if line.strip() == '$end':
            break
        else:
            line_str = line.strip()
            # https://manual.q-chem.com/5.2/A3.S4.html
            # ideriv: the order of derivatives that are evaluated analytically
            # incdft: iteration number after which incremental Fock matrix algorithm is initiated

            # job type
            if line_str.startswith('jobtype'):
                run_type = line_str.split()[1]
            elif line_str.startswith('method'):
                method = line_str.split()[1]
            elif line_str.startswith('unrestricted'):
                unrestricted = int(line_str.split()[1])
            elif line_str.startswith('basis'):
                basis_set = line_str.split()[1]
            # the symmetry
            elif line_str.startswith('symmetry'):
                g_rot = int(line_str.split()[1])
    return run_type, method, basis_set, unrestricted, g_rot


# this version works
def _helper_electron(lit):
    """Load electron information from Q-Chem log out file format."""
    next(lit)
    next(lit)
    atom_symbols = []
    atcoords = []
    for line in lit:
        if line.strip().startswith('-------------'):
            break
        else:
            # print(line.strip())
            atom_symbols.append(line.strip().split()[1])
            atcoords.append([float(i) for i in line.strip().split()[2:]])
    atnums = np.array([sym2num[i] for i in atom_symbols])
    atcoords = np.array(atcoords)
    # nuclear_replusion_energy = float(re.findall('\d+\.\d+', next(line).strip())[-2])
    nuclear_replusion_energy = float(next(lit).strip().split()[-2])
    # number of num alpha electron and beta elections
    alpha_elec, beta_elec = [int(i) for i in re.findall('\d', next(lit).strip())]
    # total energy
    for line in lit:
        if line.strip().startswith('Total energy in the final basis set'):
            break
    energy = float(line.strip().split()[-1])
    return atnums, alpha_elec, beta_elec, nuclear_replusion_energy, energy, atcoords


def _helper_orbital_energies(lit):
    """Load orbital energies."""
    # alpha occupied MOs
    alpha_mo_occupied = _helper_section("-- Occupied --", "-- Virtual --", lit, backward=True)
    # alpha unoccupied MOs
    alpha_mo_unoccupied = _helper_section("-- Virtual --", "", lit, backward=False)
    # beta occupied MOs
    beta_mo_occupied = _helper_section("-- Occupied --", "-- Virtual --", lit, backward=True)
    # beta unoccupied MOs
    beta_mo_unoccupied = _helper_section("-- Virtual --", "-" * 62, lit, backward=False)

    # number of alpha molecular orbitals
    norba = len(alpha_mo_occupied + alpha_mo_unoccupied)
    # number of beta molecular orbitals
    norbb = len(beta_mo_occupied + beta_mo_unoccupied)
    # energies
    energies = None
    # todo: not sure how to arrange the four type of molecular orbitals here
    return np.array(alpha_mo_occupied, dtype=np.float), \
           np.array(beta_mo_occupied, dtype=np.float), \
           np.array(alpha_mo_unoccupied, dtype=np.float), \
           np.array(beta_mo_unoccupied, dtype=np.float), \
           norba, norbb, energies


def _helper_section(start, end, lit, backward=False):
    """Load data between starting and ending strings."""
    data = []
    for line in lit:
        line_str = line.strip()
        if line_str == start:
            break
    for line in lit:
        if line.strip() == end:
            break
        else:
            data.extend(line.strip().split())
    if backward:
        lit.back(line)
    return data


# this one works
def _helper_mulliken(lit):
    """Load mulliken net atomic charges."""
    while True:
        line = next(lit).strip()
        if line.startswith('------'):
            break
    mulliken_charges = []
    for line in lit:
        if line.strip().startswith('--------'):
            break
        else:
            mulliken_charges.append(line.strip().split()[-2])
    return np.array(mulliken_charges, dtype=np.float)


# this one works
def _helper_dipole_moments(lit):
    """Load cartesian multiple moments."""
    for line in lit:
        if line.strip().startswith("Dipole Moment (Debye)"):
            break
    dipole_moments = next(lit).strip().split()
    # only load the float numbers
    dipole_moments = [dipole for idx, dipole in enumerate(dipole_moments) if idx % 2 != 0]
    # total dipole_moments
    dipole_tol = float(next(lit).strip().split()[-1])
    # todo:  check if need other moments as well
    return np.array(dipole_moments, dtype=np.float), dipole_tol


# this one works
def _helper_polar(lit):
    """Load polarizability matrix."""
    next(lit)
    polarizability_tensor = []
    for line in lit:
        if line.strip().startswith('Calculating analytic Hessian'):
            break
        else:
            polarizability_tensor.append(line.strip().split()[1:])
    return np.array(polarizability_tensor, dtype=np.float)


# this one works
def _helper_hessian(lit):
    """Load hessian matrix."""
    # hessian in Cartesian coordinates, shape(3 * natom, 3 * natom)
    next(lit)
    hessian = []
    for line in lit:
        if line.strip().startswith('****************'):
            break
        else:
            hessian.extend(line.strip().split()[1:])
    hessian = [i for i in hessian if not i.isdigit()]
    return np.array(hessian, dtype=np.float)


# this one works
def _helper_vibrational(lit):
    """Load vibrational analysis."""
    # for line in lit:
    #     if line.strip().startswith('Mode:'):
    #         break
    # todo:  check what information we need to include here
    for line in lit:
        if line.strip().startswith('This Molecule has'):
            break
    imaginary_freq = int(line.strip().split()[3])
    # todo: unit conversions
    vib_energy = float(next(lit).strip().split()[-2])
    return imaginary_freq, vib_energy


# this one works
def _helper_thermo(lit):
    """Load thermodynamics properties."""
    # unit conversion
    enthalpy_dict = {}
    entropy_dict = {}
    for line in lit:
        line_str = line.strip()
        if line_str.startswith("Archival summary:"):
            break
        else:
            if line_str.startswith("Translational Enthalpy"):
                enthalpy_dict["trans_enthalpy"] = float(line_str.split()[-2])
            elif line_str.startswith("Rotational Enthalpy"):
                enthalpy_dict["rot_enthalpy"] = float(line_str.split()[-2])
            elif line_str.startswith("Vibrational Enthalpy"):
                enthalpy_dict["vib_enthalpy"] = float(line_str.split()[-2])
            elif line_str.startswith("Total Enthalpy"):
                enthalpy_dict["enthalpy_total"] = float(line_str.split()[-2])
            elif line_str.startswith("Translational Entropy"):
                entropy_dict["trans_entropy"] = float(line_str.split()[-2])
            elif line_str.startswith("Rotational Entropy"):
                entropy_dict["rot_entropy"] = float(line_str.split()[-2])
            elif line_str.startswith("Vibrational Entropy"):
                entropy_dict["vib_entropy"] = float(line_str.split()[-2])
            elif line_str.startswith("Total Entropy"):
                entropy_dict["entropy_total"] = float(line_str.split()[-2])
            else:
                continue
    return enthalpy_dict, entropy_dict
