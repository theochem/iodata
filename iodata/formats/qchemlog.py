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
from typing import List, Tuple

import numpy as np

from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..periodic import sym2num
from ..utils import LineIterator, kcalmol

__all__ = []

PATTERNS = ['*.qchemlog']


@document_load_one("qchemlog", ['atcoords', 'athessian', 'atmasses', 'atnums', 'charge', 'energy',
                                'g_rot', 'mo', 'lot', 'nelec', 'obasis_name', 'run_type', 'extra'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_qchemlog_low(lit)
    result_labels = ['atcoords', 'atmasses', 'atnums','charge',
                     'charge', 'energy', 'g_rot', 'run_type']
    result = {label: data.get(label, None) for label in result_labels}
    # build molecular orbitals
    # todo: double check if this is right
    # mo_energies
    # # restricted case
    # if not data['unrestricted']:
    #     mo_energies = np.concatenate(
    #         (data['alpha_mo_occupied'], data['alpha_mo_unoccupied']), axis=0)
    #     mo_coeffs = np.empty((data['nbasis'], data['norba'])) * np.nan
    #     mo_occs = np.zeros(mo_coeffs.shape[1]) * np.nan
    #     mo = MolecularOrbitals("restricted", data['norba'], data['norba'],
    #                            mo_occs, mo_coeffs, mo_energies, None)
    # # unrestricted case
    # else:
    #     mo_energies = np.concatenate(
    #         (data['alpha_mo_occupied'], data['alpha_mo_unoccupied'],
    #          data['beta_mo_occupied'], data['beta_mo_unoccupied']), axis=0)
    #     mo_coeffs = np.empty((data['nbasis'], data['norba']+data['norbb'])) * np.nan
    #     mo_occs = np.zeros(mo_coeffs.shape[1]) * np.nan
    #     mo = MolecularOrbitals("unrestricted", data['norba'], data['norbb'],
    #                            mo_occs, mo_coeffs, mo_energies, None)
    # result['mo'] = mo

    result['lot'] = data['method']
    result['nelec'] = data['alpha_elec'] + data['beta_elec']
    result['obasis_name'] = data['basis_set'].lower()
    # moments
    moments_labels = ['dipole_moment', 'quadrupole_moments', 'dipole_tol']
    moments = {label: data.get(label, None) for label in moments_labels}
    # extra information
    extra_labels = ['spin_multi', 'nuclear_repulsion_energy', 'nbasis', 'charge',
                    'mulliken_charges', 'polarizability_tensor', 'hessian',
                    'imaginary_freq', 'vib_energy', 'entropy_dict']
    extra = {label: data.get(label, None) for label in extra_labels}
    # unit conversions for vibrational energy
    extra['vib_energy'] = extra.get('vib_energy') * kcalmol
    # convert kcal/mol to atomic units
    enthalpy_dict = {k: v * kcalmol for k, v in data['enthalpy_dict'].items()}
    # todo: unit conversions for entropy
    extra['enthalpy_dict'] = enthalpy_dict
    extra['moments'] = moments
    result['extra'] = extra
    return result


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
            data['charge'], data['spin_multi'], data['natom'], \
              data['atnums'], data['atcoords'] = _helper_atoms(lit)
            natoms = len(data['atnums'])
        # job specifications
        elif line.startswith('$rem'):
            data['run_type'], data['method'], data['basis_set'], \
              data['unrestricted'], data['symm'] = _helper_job(lit)
        # standard nuclear orientation
        elif line.startswith('Standard Nuclear Orientation (Angstroms)'):
            # atnums, alpha_elec, beta_elec, nbasis, nuclear_replusion_energy, energy, atcoords
            _, data['alpha_elec'], data['beta_elec'], data['nbasis'], \
              data['nuclear_repulsion_energy'], data['energy'], _ = _helper_electron(lit)
        # orbital energies
        elif line.startswith('Orbital Energies (a.u.)'):
            data['alpha_mo_occupied'], data['beta_mo_occupied'], data['alpha_mo_unoccupied'], \
              data['beta_mo_unoccupied'], data['norba'], data['norbb'] = _helper_orbital_energies(lit)
        # mulliken charges
        elif line.startswith('Ground-State Mulliken Net Atomic Charges'):
            data['mulliken_charges'] = _helper_mulliken(lit)
        #  cartesian multipole moments
        elif line.startswith('Cartesian Multipole Moments'):
            data['dipole_moment'], data['quadrupole_moments'], data[
                'dipole_tol'] = _helper_dipole_moments(lit)
        # polarizability matrix
        elif line.startswith('Polarizability Matrix (a.u.)'):
            data['polarizability_tensor'] = _helper_polar(lit)
        # hessian matrix
        elif line.startswith('Hessian of the SCF Energy'):
            data['hessian'] = _helper_hessian(lit, natoms)
        # vibrational analysis
        elif line.startswith('**                       VIBRATIONAL ANALYSIS'):
            data['imaginary_freq'], data['vib_energy'], data['atmasses'] = _helper_vibrational(lit)
        elif line.startswith('Rotational Symmetry Number'):
            # rotational symmetry number
            data['g_rot'] = int(line.split()[-1])
            data['enthalpy_dict'], data['entropy_dict'] = _helper_thermo(lit)

    return data


def _helper_atoms(lit: LineIterator) -> Tuple:
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
    atcoords = np.array(atcoords)
    return charge, spin_multi, natom, atnums, atcoords


def _helper_job(lit: LineIterator) -> Tuple:
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
                symm = int(line_str.split()[1])
    return run_type, method, basis_set, unrestricted, symm


def _helper_electron(lit: LineIterator) -> Tuple:
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
    alpha_elec, beta_elec = [int(i) for i in re.findall(r'\d', next(lit).strip())]
    # number of basis
    next(lit)
    nbasis = int(next(lit).strip().split()[-3])
    # total energy
    for line in lit:
        if line.strip().startswith('Total energy in the final basis set'):
            break
    energy = float(line.strip().split()[-1])
    return atnums, alpha_elec, beta_elec, nbasis, nuclear_replusion_energy, energy, atcoords


def _helper_orbital_energies(lit: LineIterator) -> Tuple:
    """Load orbital energies."""
    # alpha occupied MOs
    alpha_mo_occupied = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # alpha unoccupied MOs
    alpha_mo_unoccupied = _helper_section('-- Virtual --', '', lit, backward=False)
    # beta occupied MOs
    beta_mo_occupied = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # beta unoccupied MOs
    beta_mo_unoccupied = _helper_section('-- Virtual --', '-' * 62, lit, backward=False)

    # number of alpha molecular orbitals
    norba = len(alpha_mo_occupied + alpha_mo_unoccupied)
    # number of beta molecular orbitals
    norbb = len(beta_mo_occupied + beta_mo_unoccupied)
    # todo: not sure how to arrange the four type of molecular orbital energies here
    return np.array(alpha_mo_occupied, dtype=np.float), \
           np.array(beta_mo_occupied, dtype=np.float), \
           np.array(alpha_mo_unoccupied, dtype=np.float), \
           np.array(beta_mo_unoccupied, dtype=np.float), \
           norba, norbb


def _helper_section(start: str, end: str, lit: LineIterator, backward: bool = False) -> List:
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


def _helper_mulliken(lit: LineIterator) -> np.ndarray:
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


def _helper_dipole_moments(lit: LineIterator) -> Tuple:
    """Load cartesian multiple moments."""
    for line in lit:
        if line.strip().startswith('Dipole Moment (Debye)'):
            break
    dipole_moment = next(lit).strip().split()
    # only load the float numbers
    dipole_moment = [dipole for idx, dipole in enumerate(dipole_moment) if idx % 2 != 0]
    # total dipole_moments
    dipole_tol = float(next(lit).strip().split()[-1])
    # quadrupole_moments
    next(lit)
    quadrupole_moments = []
    quadrupole_moments.extend(next(lit).strip().split())
    quadrupole_moments.extend(next(lit).strip().split())
    quadrupole_moments = [dipole for idx, dipole in enumerate(quadrupole_moments) if idx % 2 != 0]
    return np.array(dipole_moment, dtype=np.float), \
           np.array(quadrupole_moments, dtype=np.float), dipole_tol


def _helper_polar(lit: LineIterator) -> np.ndarray:
    """Load polarizability matrix."""
    next(lit)
    polarizability_tensor = []
    for line in lit:
        if line.strip().startswith('Calculating analytic Hessian'):
            break
        else:
            polarizability_tensor.append(line.strip().split()[1:])
    return np.array(polarizability_tensor, dtype=np.float)


def _helper_hessian(lit: LineIterator, natoms: int) -> np.ndarray:
    """Load hessian matrix."""
    # hessian in Cartesian coordinates, shape(3 * natom, 3 * natom)
    col_idx = [int(i) for i in next(lit).strip().split()]
    hessian = np.empty((natoms*3, natoms*3), dtype=object)
    for line in lit:
        if line.strip().startswith('****************'):
            break
        else:
            if not line.startswith('            '):
                line_list = line.strip().split()
                row_idx = int(line_list[0]) - 1
                hessian[row_idx, col_idx[0]-1:col_idx[-1]] = line_list[1:]
            else:
                col_idx = [int(i) for i in line.strip().split()]
    return hessian.astype(np.float)


def _helper_vibrational(lit: LineIterator) -> Tuple:
    """Load vibrational analysis."""
    for line in lit:
        if line.strip().startswith('This Molecule has'):
            break
    imaginary_freq = int(line.strip().split()[3])
    vib_energy = float(next(lit).strip().split()[-2])
    next(lit)
    atmasses = []
    for line in lit:
        if line.strip().startswith('Molecular Mass:'):
            break
        else:
            atmasses.append(line.strip().split()[-1])
    atmasses = np.array(atmasses, dtype=np.float)
    return imaginary_freq, vib_energy, atmasses


def _helper_thermo(lit: LineIterator) -> Tuple:
    """Load thermodynamics properties."""
    enthalpy_dict = {}
    entropy_dict = {}
    for line in lit:
        line_str = line.strip()
        if line_str.startswith('Archival summary:'):
            break
        else:
            if line_str.startswith('Translational Enthalpy'):
                enthalpy_dict['trans_enthalpy'] = float(line_str.split()[-2])
            elif line_str.startswith('Rotational Enthalpy'):
                enthalpy_dict['rot_enthalpy'] = float(line_str.split()[-2])
            elif line_str.startswith('Vibrational Enthalpy'):
                enthalpy_dict['vib_enthalpy'] = float(line_str.split()[-2])
            elif line_str.startswith('Total Enthalpy'):
                enthalpy_dict['enthalpy_total'] = float(line_str.split()[-2])
            elif line_str.startswith('Translational Entropy'):
                entropy_dict['trans_entropy'] = float(line_str.split()[-2])
            elif line_str.startswith('Rotational Entropy'):
                entropy_dict['rot_entropy'] = float(line_str.split()[-2])
            elif line_str.startswith('Vibrational Entropy'):
                entropy_dict['vib_entropy'] = float(line_str.split()[-2])
            elif line_str.startswith('Total Entropy'):
                entropy_dict['entropy_total'] = float(line_str.split()[-2])
            else:
                continue
    return enthalpy_dict, entropy_dict
