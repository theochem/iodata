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

from typing import Tuple, List, TextIO, Dict, Union

from .overlap import init_scales
from .periodic import sym2num
from .utils import MolecularOrbitals

__all__ = ['load_wfx_low', 'get_permutation_orbital',
           'get_mask', 'load']

patterns = ['*.wfx']


def load_wfx_low(filename: str) -> Tuple:
    """Load data from a WFX file into arrays.

    Parameters
    ----------
    filename
        The filename of the wfx file.
    """

    # TODO: Add ECP and EDF support
    def helper_section(f_content: TextIO, start: str, end: str,
                       line_break: bool = False) -> Dict:
        """Extract the information based on the given name."""
        section = re.findall(start + '\n\\s+(.*?)\n' + end, f_content,
                             re.DOTALL)
        section = [i.strip() for i in section]
        if line_break:
            section = section[0].splitlines()

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
                dict_int[key] = int(helper_section(f_content=f_content,
                                                   start=val[0],
                                                   end=val[1])[0])
            else:
                dict_int[key] = None

        return dict_int

    with open(filename) as f:
        fc = f.read()
        # Check tag
        # check_tag(f_content=fc)
        # string type variables
        title, keywords, model_name = helper_str(f_content=fc).values()

        # int type variables
        num_atoms, num_primitives, num_occ_mo, num_perturbations, \
        num_electrons, num_alpha_electron, num_beta_electron, num_spin_multi \
            = helper_int(f_content=fc).values()

    return \
        title, keywords, model_name, num_atoms, num_primitives, \
        num_occ_mo, num_perturbations, num_electrons, num_alpha_electron, \
        num_beta_electron, num_spin_multi
