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
# pragma pylint: disable=wrong-import-order,invalid-name
"""Module for handling AIM/AIMAll WFX file format."""

import re

import numpy as np

from typing import Tuple, List, TextIO, Dict

__all__ = ['load_wfx_low']

patterns = ['*.wfx']


def load_wfx_low(filename: str) -> Tuple:
    """Load data from a WFX file into arrays."""
    def helper_section(f_content: TextIO, start: str, end: str,
                       line_break: bool = False) -> Dict:
        """Extract the information based on the given name."""
        section = re.findall(start + '\n\\s+(.*?)\n' + end, f_content,
                             re.DOTALL)
        section = [i.strip() for i in section]
        if line_break:
            section = [i.split() for i in section]

        return section

    def helper_str(f_content: TextIO) -> Dict:
        """Compute the string type values."""
        str_label = {
            'title': ['<Title>', '</Title>'],
            'keywords': ['<Keywords>', '</Keywords>'],
            'model_name': ['<Model>', '</Model>']
        }

        dict_str = {}
        for key, val in str_label.items():
            str_info = helper_section(f_content=f_content, start=val[0],
                                      end=val[1])
            if len(str_info) != 0:
                dict_str[key] = str_info[0]
            else:
                dict_str[key] = None

        return dict_str

    def helper_int(f_content: TextIO) -> Dict:
        """Compute the init type values."""
        int_label = {
            'num_atoms': ['<Number of Nuclei>', '</Number of Nuclei>'],
            'num_primitives': ['<Number of Primitives>',
                               '</Number of Primitives>'],
            'num_occ_mo': ['<Number of Occupied Molecular Orbitals>',
                           '</Number of Occupied Molecular Orbitals>'],
            'num_perturbations': ['<Number of Perturbations>',
                                  '</Number of Perturbations>'],
            'charge': ['<Net Charge>', '</Net Charge>'],
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
            int_info = helper_section(f_content=f_content,
                                      start=val[0],
                                      end=val[1])
            if len(int_info) != 0:
                dict_int[key] = np.array(helper_section(f_content=f_content,
                                                        start=val[0],
                                                        end=val[1])[0],
                                         dtype=int)
            else:
                dict_int[key] = np.array(None)

        return dict_int

    def helper_float(f_content: TextIO) -> Dict:
        """Compute the float type values."""
        float_label = {
            'energy': ['<Energy = T + Vne + Vee + Vnn>',
                       '</Energy = T + Vne + Vee + Vnn>'],
            'virial_ratio': ['<Virial Ratio (-V/T)>', '</Virial Ratio (-V/T)>'],
            'nuclear_viral': ['<Nuclear Virial of Energy-Gradient-Based '
                              'Forces on Nuclei, W>',
                              '</Nuclear Virial of Energy-Gradient-Based '
                              'Forces on Nuclei, W>'],
            'full_viral_ratio': ['<Full Virial Ratio, -(V - W)/T>',
                                 '</Full Virial Ratio, -(V - W)/T>']
        }
        dict_float = {}
        for key, val in float_label.items():
            if f_content.find(val[0]) > 0:
                float_info = f_content[f_content.find(
                    val[0]) + len(val[0]) + 1: f_content.find(val[1])]
                dict_float[key] = np.array(float_info, dtype=float)
            elif f_content.find(val[0]) == -1:
                dict_float[key] = np.array(None)
            else:
                continue

        return dict_float

    def energy_gradient(f_content: TextIO) -> Tuple[np.ndarray, np.ndarray]:
        gradient_list = helper_section(
            f_content=f_content,
            start='<Nuclear Cartesian Energy Gradients>',
            end='</Nuclear Cartesian Energy Gradients>',
            line_break=True)
        # build structured array
        gradient_mix = np.array(gradient_list[0]).reshape(-1, 4)
        gradient_atoms = gradient_mix[:, 0].astype(np.unicode_)
        gradient = gradient_mix[:, 1:].astype(float)
        return gradient_atoms, gradient

    def helper_mo(f_content: TextIO, num_primitives: int) \
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
        return mo_count, mo_coefficients

    def check_tag(f_content: str):
        tags_header = re.findall(r'<(?!/)(.*?)>', f_content)
        tags_tail = re.findall(r'</(.*?)>', f_content)
        # Check if header or tail tags match
        # head and tail tags of Molecular Orbital Primitive Coefficients are
        # not matched paired because there are MO between
        assert ('Molecular Orbital Primitive Coefficients' in tags_header) and \
               ('Molecular Orbital Primitive Coefficients' in tags_tail), \
            "Molecular Orbital Primitive Coefficients tags are not shown in " \
            "WFX inputfile pairwise or both are missing."
        # check others
        tags_header_check = [i for i in tags_header
                             if i != 'Molecular Orbital Primitive Coefficients']
        tags_tail_check = [i for i in tags_tail
                           if i != 'Molecular Orbital Primitive Coefficients']
        for tag_header, tag_tail in zip(tags_header_check, tags_tail_check):
            assert (tag_header == tag_tail), \
                "Tag header %s and tail %s do not match." \
                % (tag_header, tag_tail)
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
            err_str = ', '.join(diff)
            err_str += 'are/is required but not present in the WFX file.'
            raise AssertionError(err_str)

    with open(filename) as f:
        fc = f.read()
        # Check tag
        check_tag(f_content=fc)
        # string type properties
        title, keywords, model_name = helper_str(f_content=fc).values()
        # Check keywords
        assert (keywords in ['GTO', 'GIAO', 'CGST']), \
            "The keywords should be one out of GTO, GIAO and CGST."

        # int type properties
        num_atoms, num_primitives, num_occ_mo, num_perturbations, charge, \
            num_electrons, num_alpha_electron, num_beta_electron, num_spin_multi \
            = helper_int(f_content=fc).values()
        # Check number of perturbations, num_perturbations
        perturbation_check = {'GTO': 0, 'GIAO': 3, 'CGST': 6}
        assert (num_perturbations == perturbation_check[keywords]), \
            "Numbmer of perturbations is not equal to 0, 3 or 6."
        # float type properties
        energy, virial_ratio, nuclear_viral, full_viral_ratio = \
            helper_float(f_content=fc).values()
        # list type properties
        atom_names = np.array([i.split() for i in
                               helper_section(f_content=fc,
                                              start='<Nuclear Names>',
                                              end='</Nuclear Names>')],
                              dtype=np.unicode_)
        atom_numbers = np.array(helper_section(f_content=fc,
                                               start='<Atomic Numbers>',
                                               end='</Atomic Numbers>',
                                               line_break=True), dtype=int)
        mo_spin_type = np.array(helper_section(
            f_content=fc,
            start='<Molecular Orbital Spin Types>',
            end='</Molecular Orbital Spin Types>',
            line_break=True), dtype=np.unicode_).reshape(-1, 1)
        coordinates = np.array(
            helper_section(
                f_content=fc, start='<Nuclear Cartesian Coordinates>',
                end='</Nuclear Cartesian Coordinates>', line_break=True),
            dtype=np.float).reshape(-1, 3)
        centers = np.array(helper_section(
            f_content=fc, start='<Primitive Centers>',
            end='</Primitive Centers>', line_break=True), dtype=np.int)
        # primitive types
        primitives_types = np.array(helper_section(
            f_content=fc, start='<Primitive Types>',
            end='</Primitive Types>', line_break=True), dtype=np.int)
        # primitives exponents
        exponent = np.array(helper_section(
            f_content=fc, start='<Primitive Exponents>',
            end='</Primitive Exponents>', line_break=True), dtype=np.float)
        # molecular orbital
        mo_occ = np.array(helper_section(
            f_content=fc, line_break=True,
            start='<Molecular Orbital Occupation Numbers>',
            end='</Molecular Orbital Occupation Numbers>'), dtype=np.float)
        mo_energy = np.array(helper_section(
            f_content=fc, line_break=True,
            start='<Molecular Orbital Energies>',
            end='</Molecular Orbital Energies>'), dtype=np.float)
        # energy gradient
        gradient_atoms, gradient = energy_gradient(f_content=fc)
        # molecular orbital
        mo_count, mo_coefficients = helper_mo(f_content=fc,
                                              num_primitives=num_primitives)

    return \
        title, keywords, model_name, atom_names, num_atoms, num_primitives, \
        num_occ_mo, num_perturbations, num_electrons, num_alpha_electron, \
        num_beta_electron, num_spin_multi, charge, energy, \
        virial_ratio, nuclear_viral, full_viral_ratio, mo_count, \
        atom_numbers, mo_spin_type, coordinates, centers, \
        primitives_types, exponent, mo_occ, mo_energy, gradient_atoms, \
        gradient, mo_coefficients
