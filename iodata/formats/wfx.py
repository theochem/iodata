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
"""AIM/AIMAll WFX file format.

See http://aim.tkgristmill.com/wfxformat.html
"""

import warnings

import numpy as np

from ..utils import LineIterator
from ..docstrings import document_load_one

from .wfn import build_obasis, build_mo


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
        '<Electronic Spin Multiplicity>': 'spin_multi',
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
        '<MO Numbers>': 'mo_numbers',  # This is constructed in parse_wfx.
    }
    labels_array_float = {
        '<Nuclear Cartesian Coordinates>': 'atcoords',
        '<Nuclear Charges>': 'nuclear_charge',
        '<Primitive Exponents>': 'exponents',
        '<Molecular Orbital Energies>': 'mo_energy',
        '<Molecular Orbital Occupation Numbers>': 'mo_occ',
        '<Molecular Orbital Primitive Coefficients>': 'mo_coeff',
    }
    labels_other = {
        '<Nuclear Names>': 'nuclear_names',
        '<Molecular Orbital Spin Types>': 'mo_spin',
        '<Nuclear Cartesian Energy Gradients>': 'nuclear_gradient',
    }

    # list of required section tags
    required_tags = (
        list(labels_str) + list(labels_int) + list(labels_float)
        + list(labels_array_int) + list(labels_array_float) + list(labels_other)
    )
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
        lit.error("Number of perturbations is not equal to 0, 3 or 6.")
    if result['keywords'] not in ['GTO', 'GIAO', 'CGST']:
        lit.error("The keywords should be one out of GTO, GIAO and CGST.")
    return result


def parse_wfx(lit: LineIterator, required_tags: list = None) -> dict:
    """Load data in all sections existing in the given WFX file LineIterator."""
    data = {}
    mo_start = "<Molecular Orbital Primitive Coefficients>"
    section_start = None
    while True:
        # get a new line
        try:
            line = next(lit).strip()
        except StopIteration:
            break

        if section_start is None and line.startswith("<"):
            section = []
            section_start = line
            data[section_start] = section
            section_end = line[:1] + "/" + line[1:]
            # Special handling of MO coeffs
            if section_start == mo_start:
                mo_numbers = []
                data['<MO Numbers>'] = mo_numbers
        elif section_start is not None and line.startswith("</"):
            # read rest of the sections; this skips sections without a closing tag
            if line != section_end:
                lit.error("Expecting line {} but got {}.".format(section_end, line))
            section_start = None
        elif section_start == mo_start and line == '<MO Number>':
            # Special handling of MO coeffs: read mo number
            mo_numbers.append(next(lit).strip())
            next(lit)  # skip '</MO Number>'
        else:
            section.append(line)

    # check if last section was closed
    if section_start is not None:
        lit.error("Section {} is not closed at end of file.".format(section_start))
    # check required section tags
    if required_tags is not None:
        for section_tag in required_tags:
            if section_tag not in data.keys():
                lit.error(f'Section {section_tag} is missing.')
    return data


@document_load_one("WFX", ['atcoords', 'atgradient', 'atnums', 'energy',
                           'exrtra', 'mo', 'obasis', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_data_wfx(lit)
    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(
        data['centers'] - 1, data['types'] - 1, data['exponents'], lit)
    # Build the molecular orbitals
    mo = build_mo(data['mo_coeff'], data['mo_occ'], data['mo_energy'],
                  data['mo_numbers'], obasis, permutation)
    # Store WFX-specific data
    extra_labels = ['keywords', 'model_name', 'num_perturbations', 'num_core_electrons',
                    'spin_multi', 'virial_ratio', 'nuc_viral', 'full_virial_ratio', 'mo_spin']
    extra = {label: data.get(label, None) for label in extra_labels}

    return {
        'atcoords': data['atcoords'],
        'atgradient': data['atgradient'],
        'atnums': data['atnums'],
        'energy': data['energy'],
        'extra': extra,
        'mo': mo,
        'obasis': obasis,
        'title': data['title'],
    }
