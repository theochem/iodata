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
"""AIM/AIMAll WFX file format."""

from typing import Tuple, List, TextIO
import re

import numpy as np

from ..basis import MolecularBasis, Shell
# from ..docstrings import document_load_one
from ..overlap import gob_cart_normalization
from ..orbitals import MolecularOrbitals
from ..formats.wfn import CONVENTIONS, PRIMITIVE_NAMES

__all__ = []

PATTERNS = ['*.wfx']


def load_wfx_low(filename: str) -> tuple:
    """Load data from a WFX file into arrays."""
    with open(filename) as f:
        fc = f.read()
        # Check tag
        _check_tag(f_content=fc)
        # string type properties
        title, keywords, model_name = _helper_str(f_content=fc).values()
        # Check keywords
        assert (keywords in ['GTO', 'GIAO', 'CGST']), \
            "The keywords should be one out of GTO, GIAO and CGST."

        # int type properties
        num_atoms, num_primitives, num_occ_mo, num_perturbations, \
            num_electrons, num_alpha_electron, num_beta_electron, \
            num_spin_multi = _helper_int(f_content=fc).values()
        # Check number of perturbations, num_perturbations
        perturbation_check = {'GTO': 0, 'GIAO': 3, 'CGST': 6}
        assert (num_perturbations == perturbation_check[keywords]), \
            "Numbmer of perturbations is not equal to 0, 3 or 6."
        # float type properties
        charge, energy, virial_ratio, nuclear_virial, full_virial_ratio = \
            _helper_float(f_content=fc).values()
        # list type properties
        atom_names = np.array([i.split() for i in
                               _helper_section(f_content=fc,
                                               start='<Nuclear Names>',
                                               end='</Nuclear Names>')],
                              dtype=np.unicode_)
        atom_numbers = np.array(_helper_section(f_content=fc,
                                                start='<Atomic Numbers>',
                                                end='</Atomic Numbers>',
                                                line_break=True), dtype=np.int)
        mo_spin_list = _helper_section(
            f_content=fc,
            start='<Molecular Orbital Spin Types>',
            end='</Molecular Orbital Spin Types>',
            line_break=True)[0]
        mo_spin_list = [i for i in mo_spin_list if i != 'and']
        mo_spin_type = np.array(mo_spin_list, dtype=np.unicode_).reshape(-1, 1)
        atcoords = np.array(
            _helper_section(
                f_content=fc, start='<Nuclear Cartesian Coordinates>',
                end='</Nuclear Cartesian Coordinates>', line_break=True),
            dtype=np.float).reshape(-1, 3)
        icenters = np.array(_helper_section(
            f_content=fc, start='<Primitive Centers>',
            end='</Primitive Centers>', line_break=True), dtype=np.int)
        # primitive types
        primitives_types = np.array(_helper_section(
            f_content=fc, start='<Primitive Types>',
            end='</Primitive Types>', line_break=True), dtype=np.int)
        # primitives exponents
        exponent = np.array(_helper_section(
            f_content=fc, start='<Primitive Exponents>',
            end='</Primitive Exponents>', line_break=True), dtype=np.float)
        # molecular orbital
        mo_occ = np.array(_helper_section(
            f_content=fc, line_break=True,
            start='<Molecular Orbital Occupation Numbers>',
            end='</Molecular Orbital Occupation Numbers>'), dtype=np.float)
        mo_energy = np.array(_helper_section(
            f_content=fc, line_break=True,
            start='<Molecular Orbital Energies>',
            end='</Molecular Orbital Energies>'), dtype=np.float)
        # energy gradient
        gradient_atoms, gradient = _energy_gradient(f_content=fc)
        # molecular orbital
        mo_count, mo_coefficients = _helper_mo(f_content=fc,
                                               num_primitives=num_primitives)

    return \
        title, keywords, model_name, atom_names, num_atoms, num_primitives, \
        num_occ_mo, num_perturbations, num_electrons, num_alpha_electron, \
        num_beta_electron, num_spin_multi, charge, energy, \
        virial_ratio, nuclear_virial, full_virial_ratio, mo_count, \
        atom_numbers, mo_spin_type, atcoords, icenters, \
        primitives_types, exponent, mo_occ, mo_energy, gradient_atoms, \
        gradient, mo_coefficients


def _helper_section(f_content: TextIO, start: str, end: str,
                    line_break: bool = False) -> list:
    """Extract the information based on the given name."""
    section = re.findall(start + '\n(.*?)\n' + end, f_content,
                         re.DOTALL)
    section = [i.strip() for i in section]
    if line_break:
        section = [i.split() for i in section]

    return section


def _helper_str(f_content: TextIO) -> dict:
    """Compute the string type values."""
    str_label = {
        'title': ['<Title>', '</Title>'],
        'keywords': ['<Keywords>', '</Keywords>'],
        'model_name': ['<Model>', '</Model>']
    }

    dict_str = {}
    for key, val in str_label.items():
        str_info = _helper_section(f_content=f_content, start=val[0],
                                   end=val[1])
        if str_info:
            dict_str[key] = str_info[0]
        else:
            dict_str[key] = None

    return dict_str


def _helper_int(f_content: TextIO) -> dict:
    """Compute the init type values."""
    int_label = {
        'num_atoms': ['<Number of Nuclei>', '</Number of Nuclei>'],
        'num_primitives': ['<Number of Primitives>',
                           '</Number of Primitives>'],
        'num_occ_mo': ['<Number of Occupied Molecular Orbitals>',
                       '</Number of Occupied Molecular Orbitals>'],
        'num_perturbations': ['<Number of Perturbations>',
                              '</Number of Perturbations>'],
        'num_electrons': ['<Number of Electrons>',
                          '</Number of Electrons>'],
        'num_alpha_electron': ['<Number of Alpha Electrons>',
                               '</Number of Alpha Electrons>'],
        'num_beta_electron': ['<Number of Beta Electrons>',
                              '</Number of Beta Electrons>'],
        'spin_multi': ['<Electronic Spin Multiplicity>',
                       '</Electronic Spin Multiplicity>']
    }

    dict_int = {}
    for key, val in int_label.items():
        int_info = _helper_section(f_content=f_content,
                                   start=val[0],
                                   end=val[1])
        if int_info:
            dict_int[key] = np.array(int_info[0], dtype=np.int)
        else:
            dict_int[key] = np.array(None)

    return dict_int


def _helper_float(f_content: TextIO) -> dict:
    """Compute the float type values."""
    float_label = {
        'charge': ['<Net Charge>', '</Net Charge>'],
        'energy': ['<Energy = T + Vne + Vee + Vnn>',
                   '</Energy = T + Vne + Vee + Vnn>'],
        'virial_ratio': ['<Virial Ratio (-V/T)>', '</Virial Ratio (-V/T)>'],
        'nuclear_viral': ['<Nuclear Virial of Energy-Gradient-Based '
                          'Forces on Nuclei, W>',
                          '</Nuclear Virial of Energy-Gradient-Based '
                          'Forces on Nuclei, W>'],
        'full_virial_ratio': ['<Full Virial Ratio, -(V - W)/T>',
                              '</Full Virial Ratio, -(V - W)/T>']
    }
    dict_float = {}
    for key, val in float_label.items():
        if f_content.find(val[0]) > 0:
            float_info = f_content[f_content.find(
                val[0]) + len(val[0]) + 1: f_content.find(val[1])]
            dict_float[key] = np.array(float_info, dtype=float)
        # case for when string not find in the file
        elif f_content.find(val[0]) == -1:
            dict_float[key] = np.array(None)

    return dict_float


def _energy_gradient(f_content: TextIO) -> Tuple[np.ndarray, np.ndarray]:
    gradient_list = _helper_section(
        f_content=f_content,
        start='<Nuclear Cartesian Energy Gradients>',
        end='</Nuclear Cartesian Energy Gradients>',
        line_break=True)
    # build structured array
    gradient_mix = np.array(gradient_list[0]).reshape(-1, 4)
    gradient_atoms = gradient_mix[:, 0].astype(np.unicode_)
    gradient = gradient_mix[:, 1:].astype(float)
    return gradient_atoms, gradient


def _helper_mo(f_content: TextIO, num_primitives: int) \
        -> Tuple[List, np.ndarray]:
    str_idx1 = f_content.find('<Molecular Orbital Primitive Coefficients>')
    str_idx2 = f_content.find('</Molecular Orbital Primitive Coefficients>')
    mo = f_content[str_idx1 + 1 + len('<Molecular Orbital Primitive '
                                      'Coefficients>'): str_idx2]
    mo_count = [int(i) for i in
                re.findall(r'<MO Number>\n(.*?)\n</MO Number>', mo, re.S)]
    # raw-primitive expansion coefficients for MO
    coefficient_all = re.findall(
        r'[-+]?\d+.\d+[E,e][+-]\d+', mo, flags=re.MULTILINE)
    mo_coefficients = np.array(
        coefficient_all, dtype=np.float).reshape(-1, num_primitives)
    mo_coefficients = np.transpose(mo_coefficients)
    return np.array(mo_count, np.int), mo_coefficients


def _check_tag(f_content: str):
    tags_header = re.findall(r'<(?!/)(.*?)>', f_content)
    tags_tail = re.findall(r'</(.*?)>', f_content)
    # Check if header or tail tags match
    # head and tail tags of Molecular Orbital Primitive Coefficients are
    # not matched paired because there are MO between
    assert ('Molecular Orbital Primitive Coefficients' in tags_header) and \
           ('Molecular Orbital Primitive Coefficients' in tags_tail), \
        "Molecular Orbital Primitive Coefficients tags are not shown in " \
        "WFX inputfile pairwise or both are missing."
    # Check if all required tags/fields are present
    tags_required = ['Title',
                     'Keywords',
                     'Number of Nuclei',
                     'Number of Primitives',
                     'Number of Occupied Molecular Orbitals',
                     'Number of Perturbations',
                     'Nuclear Names',
                     'Nuclear Charges',
                     'Nuclear Cartesian Coordinates',
                     'Net Charge',
                     'Number of Electrons',
                     'Number of Alpha Electrons',
                     'Number of Beta Electrons',
                     'Primitive Centers',
                     'Primitive Types',
                     'Primitive Exponents',
                     'Molecular Orbital Occupation Numbers',
                     'Molecular Orbital Energies',
                     'Molecular Orbital Spin Types',
                     'Molecular Orbital Primitive Coefficients',
                     'MO Number',
                     'Energy = T + Vne + Vee + Vnn',
                     'Virial Ratio (-V/T)']
    if set(tags_header).intersection(set(tags_required)) != \
            set(tags_required):
        diff = set(tags_required) - set(tags_header).intersection(
            set(tags_required))
        error_str = ', '.join(diff)
        error_str += 'are/is required but not present in the WFX file.'
        raise IOError(error_str)
        # warnings.warn(error_str)
    # check others
    tags_header_check = [i for i in tags_header
                         if i != 'Molecular Orbital Primitive Coefficients']
    tags_tail_check = [i for i in tags_tail
                       if i != 'Molecular Orbital Primitive Coefficients']
    for tag_header, tag_tail in zip(tags_header_check, tags_tail_check):
        assert (tag_header == tag_tail), \
            "Tag header %s and tail %s do not match." \
            % (tag_header, tag_tail)


# pylint: disable=too-many-branches
def build_obasis(icenters: np.ndarray, type_assignments: np.ndarray,
                 exponents: np.ndarray) -> Tuple[MolecularBasis, np.ndarray]:
    """Construct a basis set using the arrays read from a WFX file.

    Parameters
    ----------
    icenters
        The center indices for all basis functions. shape=(nbasis,). Lowest
        index is zero.
    type_assignments
        Integer codes for basis function names. shape=(nbasis,). Lowest index
        is zero.
    exponents
        The Gaussian exponents of all basis functions. shape=(nbasis,)

    """
    # Build the basis set, keeping track of permutations in case there are
    # deviations from the default ordering of primitives in a WFN file.
    shells = []
    ibasis = 0
    nbasis = len(icenters)
    permutation = np.zeros(nbasis, dtype=int)
    # Loop over all (batches of primitive) basis functions and extract shells.
    while ibasis < nbasis:
        # Determine the angular moment of the shell
        type_assignment = type_assignments[ibasis]
        if type_assignment == 0:
            angmom = 0
        else:
            # multiple different type assignments (codes for individual basis
            # functions) can match one angular momentum.
            angmom = len(PRIMITIVE_NAMES[type_assignments[ibasis]])
        # The number of cartesian functions for the current angular momentum
        ncart = len(CONVENTIONS[(angmom, 'c')])
        # Determine how many shells are to be read in one batch. E.g. for a
        # contracted p shell, the WFN format contains first all px basis
        # functions, the all py, finally all pz. These need to be regrouped into
        # shells.
        # This pattern can almost be used to reverse-engineer contractions.
        # One should also check (i) if the corresponding mo-coefficients are the
        # same (after fixing them for normalization) and (ii) if the functions
        # are centered on the same atom.
        # For now, this implementation makes no attempt to reverse-engineer
        # contractions, but it can be done.
        ncon = 1  # the contraction length
        if angmom > 0:
            # batches for s-type functions are not necessary and may result in
            # multiple centers being pulled into one batch.
            while (ibasis + ncon < len(type_assignments)
                   and type_assignments[ibasis + ncon] == type_assignment):
                ncon += 1
        # Check if the type assignment is consistent for remaining basis
        # functions in this batch.
        for ifn in range(ncart):
            if not (type_assignments[ibasis + ncon * ifn: ibasis + ncon * (ifn + 1)]
                    == type_assignments[ibasis + ncon * ifn]).all():
                IOError("Inconcsistent type assignments in current batch of shells.")
        # Check if all basis functions in the current batch sit on
        # the same center. If not, IOData cannot read this file.
        icenter = icenters[ibasis]
        if not (icenters[ibasis: ibasis + ncon * ncart] == icenter).all():
            IOError("Incomplete shells in WFN file not supported by IOData.")
        # Check if the same exponent is used for corresponding basis functions.
        batch_exponents = exponents[ibasis: ibasis + ncon]
        for ifn in range(ncart):
            if not (exponents[ibasis + ncon * ifn: ibasis + ncon * (ifn + 1)]
                    == batch_exponents).all():
                IOError("Exponents must be the same for corresponding basis functions.")
        # A permutation is needed because we need to regroup basis functions
        # into shells.
        batch_primitive_names = [
            PRIMITIVE_NAMES[type_assignments[ibasis + ifn * ncon]]
            for ifn in range(ncart)]
        for irep in range(ncon):
            for i, primitive_name in enumerate(batch_primitive_names):
                ifn = CONVENTIONS[(angmom, 'c')].index(primitive_name)
                permutation[ibasis + irep * ncart + ifn] = ibasis + irep + i * ncon
        # WFN uses non-normalized primitives, which will be corrected for
        # when processing the MO coefficients. Normalized primitives will
        # be used here. No attempt is made here to reconstruct the contraction.
        for exponent in batch_exponents:
            shells.append(Shell(icenter, [angmom], ['c'], np.array([exponent]),
                                np.array([[1.0]])))
        # Move on to the next contraction
        ibasis += ncart * ncon
    obasis = MolecularBasis(shells, CONVENTIONS, 'L2')
    assert obasis.nbasis == nbasis
    return obasis, permutation


def load_one(filename: str) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title, _, _, _, atnums, _, _, _, _, _, _, num_spin_multi, _, energy, _, _, _, mo_count,\
        _, mo_spin_type, atcoords, icenters, type_assignments, exponents, mo_occ, mo_energy,\
        _, _, mo_coefficients = load_wfx_low(filename)

    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(icenters, type_assignments, exponents)
    # Re-order the mo coefficients.
    mo_coefficients = mo_coefficients[permutation]
    # Get the normalization of the un-normalized Cartesian basis functions.
    # Use these to rescale the mo_coefficients.
    scales = []
    for shell in obasis.shells:
        angmom = shell.angmoms[0]
        for name in obasis.conventions[(angmom, 'c')]:
            if name == '1':
                nx, ny, nz = 0, 0, 0
            else:
                nx = name.count('x')
                ny = name.count('y')
                nz = name.count('z')
            scales.append(gob_cart_normalization(shell.exponents[0], np.array([nx, ny, nz])))
    scales = np.array(scales)
    mo_coefficients /= scales.reshape(-1, 1)
    norb = mo_coefficients.shape[1]
    # make the wavefunction
    if mo_occ.max() > 1.0:
        # closed-shell system
        mo = MolecularOrbitals(
            'restricted', norb, norb,
            mo_occ, mo_coefficients, mo_energy, None)
    else:
        # open-shell system
        # counting the number of alpha orbitals
        norba = 1
        while (norba < mo_coefficients.shape[1]
               and mo_energy[norba] >= mo_energy[norba - 1]
               and mo_count[norba] == mo_count[norba - 1] + 1):
            norba += 1
        mo = MolecularOrbitals(
            'unrestricted', norba, norb - norba,
            mo_occ, mo_coefficients, mo_energy, None)

    result = {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'obasis': obasis,
        'mo': mo,
        'energy': energy,
    }
    return result
