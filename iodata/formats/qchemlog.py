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
from distutils.util import strtobool

import numpy as np

from ..docstrings import document_load_one
from ..orbitals import MolecularOrbitals
from ..periodic import sym2num
from ..utils import LineIterator, angstrom, kcalmol, calmol, kjmol

__all__ = []

PATTERNS = ['*.qchemlog']


@document_load_one("qchemlog",
                   ['atcoords', 'atmasses', 'atnums', 'energy', 'g_rot', 'mo',
                    'lot', 'obasis_name', 'run_type', 'extra'],
                   ['athessian'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_qchemlog_low(lit)

    # add these labels if they are loaded
    result_labels = ['atcoords', 'atmasses', 'atnums', 'energy', 'g_rot',
                     'run_type', 'athessian', 'lot', 'obasis_name']
    result = {label: data[label] for label in result_labels if data.get(label) is not None}

    # mulliken charges
    if data.get("mulliken_charges") is not None:
        result["atcharges"] = {"mulliken": data["mulliken_charges"]}

    # build molecular orbitals
    # ------------------------
    if data['unrestricted']:
        # unrestricted case
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
    else:
        # restricted case
        mo_energies = np.concatenate((data['mo_a_occ'], data['mo_a_vir']), axis=0)
        mo_coeffs = np.full((data['nbasis'], data['norba']), np.nan)
        mo_occs = np.zeros(mo_coeffs.shape[1])
        mo_occs[:data['alpha_elec']] = 1.0
        mo_occs[:data['beta_elec']] += 1.0
        mo = MolecularOrbitals("restricted", data['norba'], data['norba'],
                               mo_occs, mo_coeffs, mo_energies, None)
    result['mo'] = mo

    # moments
    moments = {}
    if 'dipole' in data:
        moments[(1, 'c')] = data['dipole']
    if 'quadrupole' in data:
        # Convert to alphabetical ordering: xx, xy, xz, yy, yz, zz
        moments[(2, 'c')] = data['quadrupole'][[0, 1, 3, 2, 4, 5]]
    # check total dipole parsed
    if 'dipole_tol' in data and 'dipole' in data:
        assert abs(np.linalg.norm(data['dipole']) - data['dipole_tol']) < 1.0e-4
    if moments:
        result['moments'] = moments

    # extra dictionary
    # ----------------
    # add labels to extra dictionary if they are loaded
    extra_labels = ['nuclear_repulsion_energy', 'polarizability_tensor', 'imaginary_freq',
                    'vib_energy', 'eda2', 'frags']

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

    # if present, convert eda terms from kj/mol to atomic units
    if 'eda2' in data:
        extra['eda2'] = {k: v * kjmol for k, v in data['eda2'].items()}

    result['extra'] = extra
    return result


def load_qchemlog_low(lit: LineIterator) -> dict:  # pylint: disable=too-many-branches
    """Load the information from Q-Chem log file."""
    data = {}
    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            # Read until the end of the file.
            break

        # job specifications
        if line.startswith('$rem') and 'run_type' not in data:
            data.update(_helper_rem_job(lit))

        # standard nuclear orientation (make sure multi-step jobs does not over-write this)
        elif line.startswith('Standard Nuclear Orientation (Angstroms)') and 'atcoords' not in data:
            data.update(_helper_structure(lit))

        # standard nuclear orientation for fragments in EDA jobs
        elif line.startswith('Standard Nuclear Orientation (Angstroms)'):
            if 'frags' not in data:
                data['frags'] = []
            data['frags'].append(_helper_structure(lit))

        # energy (the last energy in a multi-step job)
        elif line.startswith('Total energy in the final basis set'):
            data['energy'] = float(line.split()[-1])
        elif line.startswith('the SCF tolerance is set'):
            data['energy'] = _helper_energy(lit)

        # orbital energies (the last orbital energies in a multi-step job)
        elif line.startswith('Orbital Energies (a.u.)') and not data['unrestricted']:
            result = _helper_orbital_energies_restricted(lit)
            data['mo_a_occ'], data['mo_a_vir'] = result
            # compute number of alpha
            data['norba'] = len(data['mo_a_occ']) + len(data['mo_a_vir'])

        # orbital energies (the last orbital energies in a multi-step job)
        elif line.startswith('Orbital Energies (a.u.)') and data['unrestricted']:
            data.update(_helper_orbital_energies_unrestricted(lit))
            # compute number of alpha and beta molecular orbitals
            data['norba'] = len(data['mo_a_occ']) + len(data['mo_a_vir'])
            data['norbb'] = len(data['mo_b_occ']) + len(data['mo_b_vir'])

        # mulliken charges (the last charges in a multi-step job)
        elif line.startswith('Ground-State Mulliken Net Atomic Charges'):
            data['mulliken_charges'] = _helper_mulliken(lit)

        # cartesian multipole moments (the last mutipole moments in a multi-step job)
        elif line.startswith('Cartesian Multipole Moments'):
            data['dipole'], data['quadrupole'], data['dipole_tol'] = _helper_dipole_moments(lit)

        # polarizability matrix
        elif line.startswith('Polarizability Matrix (a.u.)'):
            data['polarizability_tensor'] = _helper_polar(lit)

        # hessian matrix
        elif line.startswith('Hessian of the SCF Energy'):
            data['athessian'] = _helper_hessian(lit, len(data['atnums']))

        # vibrational analysis
        elif line.startswith('**                       VIBRATIONAL ANALYSIS'):
            data['imaginary_freq'], data['vib_energy'], data['atmasses'] = _helper_vibrational(lit)

        # rotational symmetry number
        elif line.startswith('Rotational Symmetry Number'):
            data['g_rot'] = int(line.split()[-1])
            data['enthalpy_dict'], data['entropy_dict'] = _helper_thermo(lit)

        # energy decomposition analysis 2 (EDA2)
        elif line.startswith('Results of EDA2'):
            eda2 = _helper_eda2(lit)
            # add fragment energies to frags
            energies = eda2.pop('energies')
            for index, energy in enumerate(energies):
                data['frags'][index]['energy'] = energy
            data['eda2'] = eda2

    return data


def _helper_rem_job(lit: LineIterator) -> Tuple:
    """Load job specifications from Q-Chem output file format."""
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
            data_rem['unrestricted'] = bool(strtobool(line.split()[1]))
        elif line.split()[0].lower() == 'basis':
            data_rem['obasis_name'] = line.split()[1].lower()
        elif line.lower().startswith('symmetry'):
            data_rem['symm'] = bool(strtobool(line.split()[1]))
    return data_rem


def _helper_structure(lit: LineIterator):
    """Load electron information from Q-Chem output file format."""
    next(lit)
    next(lit)
    # atomic numbers and atomic coordinates (converted to A.U)
    atsymbols = []
    atcoords = []
    for line in lit:
        if line.strip().startswith('-------------'):
            break
        atsymbols.append(line.split()[1])
        atcoords.append([float(i) for i in line.split()[2:]])
    subdata = {"atnums": np.array([sym2num[i] for i in atsymbols]),
               "atcoords": np.array(atcoords) * angstrom,
               "nuclear_repulsion_energy": float(next(lit).split()[-2])}
    # number of alpha and beta elections
    line = next(lit).split()
    subdata["alpha_elec"] = int(line[2])
    subdata["beta_elec"] = int(line[5])
    # number of basis functions
    next(lit)
    subdata["nbasis"] = int(next(lit).split()[-3])

    return subdata


def _helper_energy(lit: LineIterator):
    for line in lit:
        if line.strip().endswith('Convergence criterion met'):
            energy = float(line.split()[1])
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
    subdata = dict()
    # alpha occupied MOs
    subdata['mo_a_occ'] = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # alpha unoccupied MOs
    subdata['mo_a_vir'] = _helper_section('-- Virtual --', '', lit, backward=False)
    # beta occupied MOs
    subdata['mo_b_occ'] = _helper_section('-- Occupied --', '-- Virtual --', lit, backward=True)
    # beta unoccupied MOs
    subdata['mo_b_vir'] = _helper_section('-- Virtual --', '-' * 62, lit, backward=False)
    return subdata


def _helper_section(start: str, end: str, lit: LineIterator, backward: bool = False) -> np.ndarray:
    """Load data between starting and ending strings."""
    data = []
    for line in lit:
        line_str = line.strip()
        if line_str == start:
            break
    for line in lit:
        if line.strip() != end:
            data.extend(line.split())
        else:
            break
    if backward:
        lit.back(line)
    return np.array(data, dtype=float)


def _helper_mulliken(lit: LineIterator) -> np.ndarray:
    """Load mulliken net atomic charges."""
    # skip line between 'Ground-State Mulliken Net Atomic Charges' line & atomic charge entries
    while True:
        line = next(lit).strip()
        if line.startswith('------'):
            break
    # store atomic charges until enf of table is reached
    mulliken_charges = []
    for line in lit:
        if line.strip().startswith('--------'):
            break
        mulliken_charges.append(line.split()[2])
    return np.array(mulliken_charges, dtype=float)


def _helper_dipole_moments(lit: LineIterator) -> Tuple:
    """Load cartesian multiple moments."""
    for line in lit:
        if line.strip().startswith('Dipole Moment (Debye)'):
            break
    # parse dipole moment (only load the float numbers)
    dipole = next(lit).split()
    dipole = np.array([float(dipole) for idx, dipole in enumerate(dipole) if idx % 2 != 0])
    # parse total dipole moment
    dipole_tol = float(next(lit).split()[-1])
    # parse quadrupole moment (xx, xy, yy, xz, yz, zz)
    next(lit)
    quadrupole = next(lit).split()
    quadrupole.extend(next(lit).split())
    quadrupole = np.array([float(dipole) for idx, dipole in enumerate(quadrupole) if idx % 2 != 0])
    return dipole, quadrupole, dipole_tol


def _helper_polar(lit: LineIterator) -> np.ndarray:
    """Load polarizability matrix."""
    next(lit)
    polarizability_tensor = []
    for line in lit:
        if line.strip().startswith('Calculating analytic Hessian'):
            break
        polarizability_tensor.append(line.split()[1:])
    return np.array(polarizability_tensor, dtype=float)


def _helper_hessian(lit: LineIterator, natom: int) -> np.ndarray:
    """Load hessian matrix."""
    # hessian in Cartesian coordinates, shape(3 * natom, 3 * natom)
    col_idx = [int(i) for i in next(lit).split()]
    hessian = np.empty((natom * 3, natom * 3), dtype=object)
    for line in lit:
        if line.strip().startswith('****************'):
            break
        if line.startswith('            '):
            col_idx = [int(i) for i in line.split()]
        else:
            line_list = line.split()
            row_idx = int(line_list[0]) - 1
            hessian[row_idx, col_idx[0] - 1:col_idx[-1]] = line_list[1:]
    return hessian.astype(float)


def _helper_vibrational(lit: LineIterator) -> Tuple:
    """Load vibrational analysis."""
    for line in lit:
        if line.strip().startswith('This Molecule has'):
            break
    # pylint: disable= W0631
    imaginary_freq = int(line.split()[3])
    vib_energy = float(next(lit).split()[-2])
    next(lit)
    atmasses = []
    for line in lit:
        if line.strip().startswith('Molecular Mass:'):
            break
        atmasses.append(line.split()[-1])
    atmasses = np.array(atmasses, dtype=float)
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


def _helper_eda2(lit: LineIterator) -> dict:  # pylint: disable=too-many-branches
    """Load Energy decomposition information."""
    next(lit)
    eda2 = {}
    for line in lit:

        if line.strip().startswith('Fragment Energies'):
            for line_2 in lit:
                if line_2.strip().startswith('-----'):
                    break
                eda2.setdefault('energies', []).append(float(line_2.split()[-1]))

        if line.strip().startswith('Orthogonal Fragment Subspace Decomposition'):
            next(lit)
            for line_2 in lit:
                if line_2.strip().startswith('-----'):
                    break
                info = line_2.split()
                if info[0] in ['E_elec', 'E_pauli', 'E_disp']:
                    eda2[info[0].lower()] = float(info[-1])

        elif line.strip().startswith('Terms summing to E_pauli'):
            next(lit)
            for line_2 in lit:
                if line_2.strip().startswith('-----'):
                    break
                info = line_2.split()
                if info[0] in ['E_kep_pauli', 'E_disp_free_pauli']:
                    eda2[info[0].lower()] = float(info[-1])

        elif line.strip().startswith('Classical Frozen Decomposition'):
            next(lit)
            for line_2 in lit:
                if line_2.strip().startswith('-----'):
                    break
                info = line_2.split()
                if info[0] in ['E_cls_elec', 'E_cls_pauli']:
                    eda2[info[0].lower()] = float(info[5])
                elif info[0].split("[")[1] == 'E_mod_pauli':
                    eda2[info[0].split("[")[1].lower()] = float(info[5])

        elif line.strip().startswith('Simplified EDA Summary'):
            next(lit)
            for line_2 in lit:
                if line_2.strip().startswith('-----'):
                    break
                info = line_2.split()
                if info[0] in ['PREPARATION', 'FROZEN', 'DISPERSION', 'POLARIZATION', 'TOTAL']:
                    eda2[info[0].lower()] = float(info[1])
                elif info[0].split("[")[-1] == 'PAULI':
                    eda2[info[0].split("[")[-1].lower()] = float(info[1].split("]")[0])
                elif info[0] == 'CHARGE':
                    eda2[info[0].lower() + ' ' + info[1].lower()] = float(info[2])

        elif line.strip().startswith('-------------------------------------------------------'):
            break

    return eda2
