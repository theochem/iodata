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
                       line_break: bool = False) -> List:
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

        return dict_str.values()

    with open(filename) as f:
        fc = f.read()
        # Check tag
        # check_tag(f_content=fc)
        # string type variables
        title, keywords, model_name = helper_str(f_content=fc)

    return title, keywords, model_name
