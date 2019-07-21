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

from typing import Tuple

import warnings

import numpy as np

from ..basis import MolecularBasis, Shell
from ..utils import LineIterator
# from ..docstrings import document_load_one
from ..overlap import gob_cart_normalization
from ..orbitals import MolecularOrbitals
from ..formats.wfn import CONVENTIONS, PRIMITIVE_NAMES

__all__ = []

PATTERNS = ['*.wfx']


def load_data_wfx(lit: LineIterator) -> dict:
    """Process loaded WFX data."""

    labels_str = {
        '<Title>': 'title',
        '<Keywords>': 'keywords',
        '<Model>': 'model_name',
    }
    # integer numbers
    labels_int = {
        '<Number of Nuclei>': 'num_atoms',
        '<Number of Occupied Molecular Orbitals>': 'num_occ_mo',
        '<Number of Perturbations>': 'num_perturbations',
        '<Number of Electrons>': 'num_electrons',
        '<Number of Core Electrons>': 'num_core_electrons',
        '<Number of Alpha Electrons>': 'num_alpha_electron',
        '<Number of Beta Electrons>': 'num_beta_electron',
        '<Number of Primitives>': 'num_primitives',
        '<Electronic Spin Multiplicity>': 'spin_multi'
    }
    # float numbers
    labels_float = {
        '<Net Charge>': 'charge',
        '<Energy = T + Vne + Vee + Vnn>': 'energy',
        '<Virial Ratio (-V/T)>': 'virial_ratio',
        '<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>': 'nuc_viral',
        '<Full Virial Ratio, -(V - W)/T>': 'full_virial_ratio',
    }
    labels_array_int = {
        '<Atomic Numbers>': 'atnums',
        '<Primitive Centers>': 'centers',
        '<Primitive Types>': 'types',
    }
    labels_array_float = {
        '<Nuclear Cartesian Coordinates>': 'atcoords',
        '<Nuclear Charges>': 'nuclear_charge',
        '<Primitive Exponents>': 'exponents',
        '<Molecular Orbital Energies>': 'mo_energy',
        '<Molecular Orbital Occupation Numbers>': 'mo_occ',
        '<Molecular Orbital Primitive Coefficients>': 'mo_coeff'
    }
    labels_other = {
        '<Nuclear Names>': 'nuclear_names',
        '<Molecular Orbital Spin Types>': 'mo_spin',
        '<Nuclear Cartesian Energy Gradients>': 'nuclear_gradient',
    }

    # list of required section tags
    required_tags = list(labels_str.keys()) + list(labels_int.keys()) + list(labels_float)
    required_tags += list(labels_array_int) + list(labels_array_float) + list(labels_other)
    required_tags.remove('<Model>')
    required_tags.remove('<Number of Core Electrons>')
    required_tags.remove('<Electronic Spin Multiplicity>')
    required_tags.remove('<Atomic Numbers>')
    required_tags.remove('<Full Virial Ratio, -(V - W)/T>')
    required_tags.remove('<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>')
    required_tags.remove('<Nuclear Cartesian Energy Gradients>')

    # load raw data & check required tags
    data = parse_wfx(lit, required_tags)

    # process raw data
    result = {}
    for key, value in data.items():
        if key in labels_str:
            result[labels_str[key]] = value[0]
        elif key in labels_int:
            result[labels_int[key]] = int(value[0])
        elif key in labels_float:
            result[labels_float[key]] = float(value[0])
        elif key in labels_array_float:
            result[labels_array_float[key]] = np.fromstring(" ".join(value),
                                                            dtype=np.float,
                                                            sep=" ")
        elif key in labels_array_int:
            result[labels_array_int[key]] = np.fromstring(" ".join(value),
                                                          dtype=np.int,
                                                          sep=" ")
        elif key in labels_other:
            result[labels_other[key]] = value
        else:
            warnings.warn("Not recognized label, skip {0}".format(key))

    # reshape some arrays
    result['atcoords'] = result['atcoords'].reshape(-1, 3)
    result['mo_coeff'] = result['mo_coeff'].reshape(result['num_primitives'], -1, order='F')
    # process mo spin type
    mo_spin_list = [i.split() for i in result['mo_spin']]
    mo_spin_type = np.array(mo_spin_list, dtype=np.unicode_).reshape(-1, 1)
    result['mo_spin'] = mo_spin_type[mo_spin_type[:, 0] != 'and']
    # process nuclear gradient
    gradient_mix = np.array([i.split() for i in result.pop('nuclear_gradient')]).reshape(-1, 4)
    gradient_atoms = gradient_mix[:, 0].astype(np.unicode_)
    index = [result['nuclear_names'].index(atom) for atom in gradient_atoms]
    result['atgradient'] = np.full((len(result['nuclear_names']), 3), np.nan)
    result['atgradient'][index] = gradient_mix[:, 1:].astype(float)
    # check number of perturbations
    perturbation_check = {'GTO': 0, 'GIAO': 3, 'CGST': 6}
    if result['num_perturbations'] != perturbation_check[result['keywords']]:
        raise ValueError("Number of perturbations is not equal to 0, 3 or 6.")
    if result['keywords'] not in ['GTO', 'GIAO', 'CGST']:
        raise ValueError("The keywords should be one out of GTO, GIAO and CGST.")
    return result


def parse_wfx(lit: LineIterator, required_tags: list = None) -> dict:
    """Load data in all sections existing in the given WFX file LineIterator."""
    data = {}
    while True:
        # get a new line
        try:
            line = next(lit).strip()
        except StopIteration:
            break

        # read mo primitive coefficients
        if line == "<Molecular Orbital Primitive Coefficients>":
            section = []
            section_start = "<Molecular Orbital Primitive Coefficients>"
            section_end = "</Molecular Orbital Primitive Coefficients>"
            line = next(lit).strip()
            while line != section_end:
                # skip mo number section
                if line == '<MO Number>':
                    for _ in range(3):
                        line = next(lit).strip()
                section.append(line)
                line = next(lit).strip()

        # read rest of the sections; this skips sections without a closing tag
        if line.startswith("<") and not line.startswith("</"):
            section = []
            section_start = line
            section_end = line[:1] + "/" + line[1:]
        elif line.startswith("</"):
            if line != section_end:
                lit.error("Expected line {0} but got {1}".format(section_end, line))
            data[section_start] = section
        else:
            section.append(line)

    # check required section tags
    if required_tags is not None:
        for section_tag in required_tags:
            if section_tag not in data.keys():
                raise IOError(f'The {section_tag} section is missing!')
    return data


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


def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_data_wfx(lit)

    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(data['icenters'], data['type'], data['exponents'])
    # Re-order the mo coefficients.
    mo_coefficients = data['mo_coeff'][permutation]
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
    mo_count = np.arange(data['mo_coeff'].shape[1])
    norb = mo_coefficients.shape[1]
    # make the wavefunction
    if data['mo_occ'].max() > 1.0:
        # closed-shell system
        mo = MolecularOrbitals(
            'restricted', norb, norb,
            data['mo_occ'], data['mo_coeff'], data['mo_energy'], None)
    else:
        # open-shell system
        # counting the number of alpha orbitals
        norba = 1
        while (norba < mo_coefficients.shape[1]
               and data['mo_energy'][norba] >= data['mo_energy'][norba - 1]
               and mo_count[norba] == mo_count[norba - 1] + 1):
            norba += 1
        mo = MolecularOrbitals(
            'unrestricted', norba, norb - norba,
            data['mo_occ'], data['mo_coeff'], data['mo_energy'], None)

    extra_items = ['keywords', 'model_name', 'num_perturbations', 'num_core_electrons',
                   'spin_multi', 'virial_ratio', 'nuc_viral', 'full_virial_ratio', 'mo_spin']
    extra = {item: data.setdefault(item, None) for item in extra_items}

    result = {
        'title': data['title'],
        'atcoords': data['atcoords'],
        'atnums': data['atnums'],
        'obasis': obasis,
        'mo': mo,
        'energy': data['energy'],
        'atgradient': data['atgradient'],
        'extra': extra,
    }
    return result
