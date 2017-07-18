# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --

import numpy as np


__all__ = ['set_four_index_element']

angstrom = 1.0e-10 / 0.5291772083e-10


def str_to_shell_types(s, pure=False):
    """Convert a string into a list of contraction types"""
    if pure:
        d = {'s': 0, 'p': 1, 'd': -2, 'f': -3, 'g': -4, 'h': -5, 'i': -6}
    else:
        d = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    return [d[c] for c in s.lower()]


def shell_type_to_str(shell_type):
    """Convert a shell type into a character"""
    return {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i'}[abs(shell_type)]


def set_four_index_element(four_index_object, i, j, k, l, value):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation

    Parameters
    ----------
    four_index_object : np.ndarray, shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
        The four-index object
    i, j, k, l: int
        The indices to assign to.
    value : float
        The value of the matrix element to store.
    """
    four_index_object[i, j, k, l] = value
    four_index_object[j, i, l, k] = value
    four_index_object[k, j, i, l] = value
    four_index_object[i, l, k, j] = value
    four_index_object[k, l, i, j] = value
    four_index_object[l, k, j, i] = value
    four_index_object[j, k, l, i] = value
    four_index_object[l, i, j, k] = value


def shells_to_nbasis(shell_types):
    nbasis_shell = [2 * i + 1 for i in shell_types]
    return sum(nbasis_shell)


def get_shell_nbasis(shell):
    if shell > 0:  # Cartesian
        return (shell + 1) * (shell + 2) / 2
    elif shell == -1:
        raise ValueError
    else:  # Pure
        return -2 * shell + 1


def gob_cart_normalization(alpha, n):
    # sqrt(pow(4.0*alpha, n[0]+n[1]+n[2])*pow(2.0*alpha/M_PI, 1.5)
    #        /(fac2(2*n[0]-1)*fac2(2*n[1]-1)*fac2(2*n[2]-1)));

    vfac2 = np.vectorize(fac2)
    return np.sqrt((4*alpha)**sum(n) * (2*alpha/np.pi)**1.5 / np.prod(vfac2(2*n-1)))


def fac2(n):
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result
