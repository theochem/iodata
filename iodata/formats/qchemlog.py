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

This module will load Q-Chem log file into IODATA.
"""

from typing import Tuple
from collections import namedtuple
from distutils.util import strtobool

import numpy as np

from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..periodic import sym2num
from ..utils import LineIterator, angstrom, kcalmol, calmol, kjmol

__all__ = []

PATTERNS = ['*.qchemlog']


@document_load_one("qchemlog", ['atcoords', 'atmasses', 'atnums', 'charge', 'energy', 'g_rot',
                                'mo', 'lot', 'nelec', 'obasis_name', 'run_type', 'extra'],
                   ['athessian'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_qchemlog_low(lit)

    # add these labels if they are loaded
    result_labels = ['atcoords', 'atmasses', 'atnums', 'charge', 'charge', 'energy', 'g_rot',
                     'run_type', 'athessian', 'lot', 'obasis_name']
    result = {label: data[label] for label in result_labels if data.get(label) is not None}
    # add number of electrons
    result['nelec'] = data['alpha_elec'] + data['beta_elec']

    # mulliken charges
    if data.get("mulliken_charges") is not None:
        result["atcharges"] = {"mulliken": data["mulliken_charges"]}

    # build molecular orbitals
    # ------------------------
    # restricted case
    if data['unrestricted'] is False:
        mo_energies = np.concatenate((data['mo_a_occ'], data['mo_a_vir']), axis=0)
        mo_coeffs = np.full((data['nbasis'], data['norba']), np.nan)
        mo_occs = np.zeros(mo_coeffs.shape[1])
        mo_occs[:data['alpha_elec']] = 1.0
        mo_occs[:data['beta_elec']] += 1.0
        mo = MolecularOrbitals("restricted", data['norba'], data['norba'],
                               mo_occs, mo_coeffs, mo_energies, None)
    # unrestricted case
    else:
        mo_energies = np.concatenate((data['mo_a_occ'], data['mo_a_vir'],
                                      data['mo_b_occ'], data['mo_b_vir']), axis=0)
        mo_coeffs = np.full((data['nbasis'], data['norba'] + data['norbb']), np.nan)
        mo_occs = np.zeros(mo_coeffs.shape[1])
        # number of alpha & beta electrons and number of alpha molecular orbitals
        na, nb = data['alpha_elec'], data['beta_elec']
        na_mo = len(data['mo_a_occ']) + len(data['mo_a_vir'])
        mo_occs[:na] = 1.0
        mo_occs[na_mo: na_mo + nb] = 1.0
        mo = MolecularOrbitals("unrestricted", data['norba'], data['norbb'],
                               mo_occs, mo_coeffs, mo_energies, None)
    result['mo'] = mo

    # moments
    moments = {}
    if 'dipole' in data:
        moments[(1, 'c')] = data['dipole']
    if 'quadrupole' in data:
        # Convert to alphabetical ordering: xx, xy, xz, yy, yz, zz
        moments[(2, 'c')] = data['quadrupole'][[0, 1, 3, 2, 4, 5]]
    if moments:
        result['moments'] = moments

    # extra dictionary
    # ----------------
    # add labels to extra dictionary if they are loaded
    extra_labels = ['spin_multi', 'nuclear_repulsion_energy',
                    'polarizability_tensor', 'imaginary_freq', 'vib_energy',
                    'frag_energies', 'orth_frag_subs_decom', 'clss_frozen_decomp', 'eda_summary']
    extra = {label: data[label] for label in extra_labels if data.get(label) is not None}
    # if present, convert vibrational energy from kcal/mol to "atomic units + K"
    if 'vib_energy' in extra:
        extra['vib_energy'] *= kcalmol

    # if present, convert enthalpy terms from kcal/mol to "atomic units + K"
    if 'enthalpy_dict' in data:
        extra['enthalpy_dict'] = {k: v * kcalmol for k, v in data['enthalpy_dict'].items()}

    # if present, convert entropy terms from cal/mol.K to "atomic units + Kalvin"
    if 'entropy_dict' in data:
        extra['entropy_dict'] = {k: v * calmol for k, v in data['entropy_dict'].items()}

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

        # save Q-chem version
        if line.startswith('You'):
            version = line.strip().split()[-1]
        # get the atomic information
        if line.startswith('$molecule'):
            if 'atnums' not in data:
                result = _helper_atoms(lit)
                data['charge'], data['spin_multi'], data['atnums'], data['atcoords'] = result
                data['natom'] = len(data['atnums'])
        # job specifications
        elif line.startswith('$rem') and 'run_type' not in data:
            data.update(_helper_job(lit, version=version))
        # standard nuclear orientation
        # elif line.startswith('Standard Nuclear Orientation (Angstroms)'):
        # todo: report bug for fragment job only need first nuclear orientation
        elif line.startswith('Standard Nuclear Orientation') and 'alpha_elec' not in data:
            # atnums, alpha_elec, beta_elec, nbasis, nuclear_replusion_energy, energy, atcoords
            _, data['alpha_elec'], data['beta_elec'], data['nbasis'], \
                data['nuclear_repulsion_energy'], data['energy'], _ = _helper_electron(lit)
        # todo: report bug about energy for qchemlog for more than one job
        # energy
        elif line.startswith('the SCF tolerance is set'):
            data['energy'] = _helper_energy(lit)
        # orbital energies
        elif line.startswith('Orbital Energies (a.u.)') and data['unrestricted'] is None:
            result = _helper_orbital_energies_restricted(lit)
            data['mo_a_occ'], data['mo_a_vir'] = result
            # compute number of alpha
            data['norba'] = len(data['mo_a_occ']) + len(data['mo_a_vir'])
        # orbital energies
        elif line.startswith('Orbital Energies (a.u.)') and data['unrestricted'] is not None:
            result = _helper_orbital_energies_unrestricted(lit)
            data['mo_a_occ'], data['mo_b_occ'], data['mo_a_vir'], data['mo_b_vir'] = result
            # compute number of alpha and beta molecular orbitals
            data['norba'] = len(data['mo_a_occ']) + len(data['mo_a_vir'])
            data['norbb'] = len(data['mo_b_occ']) + len(data['mo_b_vir'])
        # mulliken charges
        elif line.startswith('Ground-State Mulliken Net Atomic Charges'):
            data['mulliken_charges'] = _helper_mulliken(lit, version)
        #  cartesian multipole moments
        elif line.startswith('Cartesian Multipole Moments'):
            data['dipole'], data['quadrupole'], data['dipole_tol'] = _helper_dipole_moments(lit)
        # polarizability matrix
        elif line.startswith('Polarizability Matrix (a.u.)'):
            data['polarizability_tensor'] = _helper_polar(lit)
        # hessian matrix
        elif line.startswith('Hessian of the SCF Energy'):
            data['athessian'] = _helper_hessian(lit, data['natom'])
        # vibrational analysis
        elif line.startswith('**                       VIBRATIONAL ANALYSIS'):
            data['imaginary_freq'], data['vib_energy'], data['atmasses'] = _helper_vibrational(lit)
        # rotational symmetry number
        elif line.startswith('Rotational Symmetry Number'):
            data['g_rot'] = int(line.split()[-1])
            data['enthalpy_dict'], data['entropy_dict'] = _helper_thermo(lit)
        # Energy Decomposition analysis (EDA)
        elif line.startswith('Basic EDA Quantities'):
            data['frag_energies'], data['orth_frag_subs_decom'],\
                data['clss_frozen_decomp'], data['eda_summary'] = _helper_eda(lit)

    return data


def _helper_atoms(lit: LineIterator) -> Tuple:
    """Load list of coordinates from an Q-Chem log output file format."""
    # net charge and spin multiplicity
    charge, spin_multi = [int(i) for i in next(lit).strip().split()]
    # atomic numbers and atomic coordinates (in Angstrom)
    atom_symbols = []
    atcoords = []
    for line in lit:
        if line.strip() == '--':
            next(lit)
        elif line.strip() == '$end':
            break
        else:
            atom_symbols.append(line.strip().split()[0])
            atcoords.append([float(i) for i in line.strip().split()[1:]])
    atnums = np.array([sym2num[i] for i in atom_symbols])
    atcoords = np.array(atcoords)
    return charge, spin_multi, atnums, atcoords


def _helper_job(lit: LineIterator, version) -> Tuple:
    """Load job specifications from Q-Chem log out file format."""
    data_rem = {}
    for line in lit:
        if line.strip() == '$end':
            break
        line = line.strip()
        # parse job type section; some sections might not be available
        if line.lower().startswith('jobtype'):
            data_rem['run_type'] = line.split()[1].lower()
        elif line.lower().startswith('method'):
            data_rem['lot'] = line.split()[1].lower()
        elif line.lower().startswith('unrestricted'):
            data_rem['unrestricted'] = bool(line.split()[1])
            data_rem['unrestricted'] = bool(strtobool(line.split()[1]))
        elif line.split()[0].lower() == 'basis':
            data_rem['obasis_name'] = line.split()[1].lower()
        elif line.lower().startswith('symmetry'):
            data_rem['symm'] = bool(strtobool(line.split()[1]))
    return data_rem


def _helper_electron(lit: LineIterator):
    """Load electron information from Q-Chem log out file format."""
    next(lit)
    next(lit)
    # atomic numbers and atomic coordinates (converted to A.U)
    atom_symbols = []
    atcoords = []
    for line in lit:
        if line.strip().startswith('-------------'):
            break
        atom_symbols.append(line.strip().split()[1])
        atcoords.append([float(i) for i in line.strip().split()[2:]])
    atnums = np.array([sym2num[i] for i in atom_symbols])
    atcoords = np.array(atcoords)
    # nuclear_replusion_energy = float(re.findall('\d+\.\d+', next(line).strip())[-2])
    nuclear_replusion_energy = float(next(lit).strip().split()[-2])
    # number of num alpha electron and beta elections
    # todo: report bug for alpha/beta electrons higher than 9
    line = next(lit).strip().split()
    alpha_elec = int(line[2])
    beta_elec = int(line[5])
    # alpha_elec, beta_elec = [int(i) for i in re.findall(r'\d', next(lit).strip())]
    # number of basis
    next(lit)
    nbasis = int(next(lit).strip().split()[-3])
    # total energy
    energy = 0
    for line in lit:
        if line.strip().startswith('Total energy in the final basis set'):
            energy = float(line.strip().split()[-1])
            break
        elif line.strip().endswith('Convergence criterion met'):
            energy = float(line.strip().split()[1])
            break
    return atnums, alpha_elec, beta_elec, nbasis, nuclear_replusion_energy, energy, atcoords


# todo: report bug in energy. Don't know if it is something related to EDA or difference between version 5.2 and 5.3
def _helper_energy(lit: LineIterator):
    for line in lit:
        if line.strip().endswith('Convergence criterion met'):
            energy = float(line.strip().split()[1])
            break
    return energy


def _helper_orbital_energies_restricted(lit: LineIterator) -> Tuple:
    """Load occupied and virtual orbital energies for restricted calculation."""
    # alpha occupied MOs
    mo_a_occupied = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # alpha unoccupied MOs
    mo_a_unoccupied = _helper_section('-- Virtual --', '-' * 62, lit, backward=False)
    return mo_a_occupied, mo_a_unoccupied


def _helper_orbital_energies_unrestricted(lit: LineIterator) -> Tuple:
    """Load occupied and virtual orbital energies for unrestricted calculation."""
    # alpha occupied MOs
    mo_a_occupied = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # alpha unoccupied MOs
    mo_a_unoccupied = _helper_section('-- Virtual --', '', lit, backward=False)
    # beta occupied MOs
    mo_b_occupied = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # beta unoccupied MOs
    mo_b_unoccupied = _helper_section('-- Virtual --', '-' * 62, lit, backward=False)
    return mo_a_occupied, mo_b_occupied, mo_a_unoccupied, mo_b_unoccupied


def _helper_section(start: str, end: str, lit: LineIterator, backward: bool = False) -> np.ndarray:
    """Load data between starting and ending strings."""
    data = []
    for line in lit:
        line_str = line.strip()
        if line_str == start:
            break
    for line in lit:
        if line.strip() != end:
            data.extend(line.strip().split())
        else:
            break
    if backward:
        lit.back(line)
    return np.array(data, dtype=np.float)


def _helper_mulliken(lit: LineIterator, version) -> np.ndarray:
    """Load mulliken net atomic charges."""
    while True:
        line = next(lit).strip()
        if line.startswith('------'):
            break
    mulliken_charges = []
    for line in lit:
        if line.strip().startswith('--------'):
            break
        elif version == '5.2.0':
            mulliken_charges.append(line.strip().split()[-2])
        # todo: report bug for mulliken charges. Seems to be related to the version
        elif version == '5.3.0':
            mulliken_charges.append(line.strip().split()[-1])
    return np.array(mulliken_charges, dtype=np.float)


def _helper_dipole_moments(lit: LineIterator) -> Tuple:
    """Load cartesian multiple moments."""
    for line in lit:
        if line.strip().startswith('Dipole Moment (Debye)'):
            break
    # parse dipole moment (only load the float numbers)
    dipole = next(lit).strip().split()
    dipole = np.array([float(dipole) for idx, dipole in enumerate(dipole) if idx % 2 != 0])
    # parse total dipole moment
    dipole_tol = float(next(lit).strip().split()[-1])
    # parse quadrupole moment
    next(lit)
    quadrupole = next(lit).strip().split()
    quadrupole.extend(next(lit).strip().split())
    quadrupole = np.array([float(dipole) for idx, dipole in enumerate(quadrupole) if idx % 2 != 0])
    return dipole, quadrupole, dipole_tol


def _helper_polar(lit: LineIterator) -> np.ndarray:
    """Load polarizability matrix."""
    next(lit)
    polarizability_tensor = []
    for line in lit:
        if line.strip().startswith('Calculating analytic Hessian'):
            break
        polarizability_tensor.append(line.strip().split()[1:])
    return np.array(polarizability_tensor, dtype=np.float)


def _helper_hessian(lit: LineIterator, natoms: int) -> np.ndarray:
    """Load hessian matrix."""
    # hessian in Cartesian coordinates, shape(3 * natom, 3 * natom)
    col_idx = [int(i) for i in next(lit).strip().split()]
    hessian = np.empty((natoms * 3, natoms * 3), dtype=object)
    for line in lit:
        if line.strip().startswith('****************'):
            break
        if not line.startswith('            '):
            line_list = line.strip().split()
            row_idx = int(line_list[0]) - 1
            hessian[row_idx, col_idx[0] - 1:col_idx[-1]] = line_list[1:]
        else:
            col_idx = [int(i) for i in line.strip().split()]
    return hessian.astype(np.float)


def _helper_vibrational(lit: LineIterator) -> Tuple:
    """Load vibrational analysis."""
    for line in lit:
        if line.strip().startswith('This Molecule has'):
            break
    # pylint: disable= W0631
    imaginary_freq = int(line.strip().split()[3])
    vib_energy = float(next(lit).strip().split()[-2])
    next(lit)
    atmasses = []
    for line in lit:
        if line.strip().startswith('Molecular Mass:'):
            break
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
    return enthalpy_dict, entropy_dict


def _helper_eda(lit: LineIterator) -> Tuple:
    """Load Energy decomposition information"""
    next(lit)
    frag_energies = {}
    orth_frag_subs_decom = {}
    clss_frozen_decomp = {}
    eda_summary = {}
    for line in lit:
        if line.startswith('Fragment Energies'):
            for line_2 in lit:
                if line_2.startswith('--------------------'):
                    break
                else:
                    info = line_2.strip().split()
                    frag_energies[info[0]] = float(info[1])
        elif line.startswith('  Orthogonal Fragment Subspace Decomposition'):
            next(lit)
            for line_2 in lit:
                if line_2.startswith('     --------------------'):
                    break
                else:
                    info = line_2.strip().split()
                    if info[0] == 'E_elec':
                        orth_frag_subs_decom[info[0]] = float(info[4])
                    elif info[0] == 'E_pauli':
                        orth_frag_subs_decom[info[0]] = float(info[4])
                    elif info[0] == 'E_disp':
                        orth_frag_subs_decom[info[0]] = float(info[4])
        elif line.startswith('     Terms summing to E_pauli'):
            next(lit)
            pauli = []
            for line_2 in lit:
                if line_2.startswith('  --------------------'):
                    break
                else:
                    info = line_2.strip().split()
                    pauli.append(float(info[3]))
            orth_frag_subs_decom['E_pauli'] = np.array(pauli)
        elif line.startswith('  Classical Frozen Decomposition'):
            next(lit)
            for line_2 in lit:
                if line_2.startswith('  --------------------'):
                    break
                else:
                    info = line_2.strip().split()
                    if info[0] == 'E_cls_elec':
                        clss_frozen_decomp[info[0]] = float(info[5])
                    elif info[0] == 'E_cls_pauli':
                        clss_frozen_decomp[info[0]] = float(info[5])
                    elif info[0].split("[")[1] == 'E_mod_pauli':
                        clss_frozen_decomp[info[0].split("[")[1]] = float(info[5])
        elif line.startswith('Simplified EDA Summary'):
            next(lit)
            for line_2 in lit:
                if line_2.startswith('--------------------'):
                    break
                else:
                    info = line_2.strip().split()
                    if info[0] == 'PREPARATION':
                        eda_summary[info[0]] = float(info[1])
                    elif info[0] == 'FROZEN':
                        eda_summary[info[0]] = float(info[1])
                    elif info[0].split("[")[-1] == 'PAULI':
                        eda_summary[info[0].split("[")[-1]] = float(info[1].split("]")[0])
                    if info[0] == 'DISPERSION':
                        eda_summary[info[0]] = float(info[1])
                    elif info[0] == 'POLARIZATION':
                        eda_summary[info[0]] = float(info[1])
                    elif info[0] == 'CHARGE':
                        eda_summary[info[0] + ' ' + info[1]] = float(info[2])
                    elif info[0] == 'TOTAL':
                        eda_summary[info[0]] = float(info[1])
        elif line.startswith(' --------------------------------------------------------------'):
            break

    return frag_energies, orth_frag_subs_decom, clss_frozen_decomp, eda_summary
