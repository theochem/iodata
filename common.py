# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
'''Shared routines in the IO package'''


import numpy as np
from horton.gbasis.cext import gob_cart_normalization


__all__ = ['renorm_helper', 'get_orca_signs']


def renorm_helper(con_coeff, alpha, shell_type, reverse=False):
    '''Fix an unnormalized contraction coefficient from the molden/orca file.

       **Arguments:**

       con_coeff
            The original contraction coefficient

       alpha
            The Gaussian exponent

       shell_type
            The shell_type as an integer. Must be one of the following: 0=s,
            1=p, -2=d, -3=f, -4=g.

       **Optional arguments:**

       reverse
            By default, the conversion is good for reading molden or orca wfn
            files. The opposite conversion can be done by setting this option to
            True. This is useful for writing molden or orca wfn files.
    '''
    if shell_type == 0:
        scale = gob_cart_normalization(alpha, np.array([0,0,0]))
    elif shell_type == 1:
        scale = gob_cart_normalization(alpha, np.array([1,0,0]))
    elif shell_type == -2:
        scale = gob_cart_normalization(alpha, np.array([1,1,0]))
    elif shell_type == -3:
        scale = gob_cart_normalization(alpha, np.array([1,1,1]))
    elif shell_type == -4:
        scale = gob_cart_normalization(alpha, np.array([2,1,1]))
    else:
        raise NotImplementedError('MKL Normalization conventions beyond G are not known. Please notify Toon.Verstraelen@UGent.be.')
    if reverse:
        return con_coeff*scale
    else:
        return con_coeff/scale


def get_orca_signs(obasis):
    '''Correct for different sign conventions for Gaussian basis functions used in ORCA.

       **Arguments:**

       obasis
            An instance GOBasis.
    '''
    sign_rules = {
      -4: np.array([1,1,1,1,1,-1,-1,-1,-1]),
      -3: np.array([1,1,1,1,1,-1,-1]),
      -2: np.array([1,1,1,1,1]),
       0: np.array([1]),
       1: np.array([1,1,1]),
    }
    signs = []
    for shell_type in obasis.shell_types:
        signs.extend(sign_rules[shell_type])
    return np.array(signs, dtype=int)
