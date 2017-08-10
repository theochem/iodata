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
"""Mulliken partitioning"""


import numpy as np

from .. overlap import get_shell_nbasis, compute_overlap


__all__ = ['partition_mulliken', 'get_mulliken_operators']


def partition_mulliken(operator, obasis, index):
    """Fill in the mulliken operator in the first argument.

    Parameters
    ----------
    operator : np.ndarray, shape=(nbasis, nbasis), dtype=float
        A Two index operator to which the Mulliken mask is applied.
    obasis : GOBasis
        The localized orbital basis for which the Mulliken operator is to be constructed.
    index : int
        The index of the atom (center) for which the Mulliken operator needs to be
        constructed.

    This routine implies that the first ``natom`` centers in the obasis corresponds to the
    atoms in the system.
    """
    mask = np.zeros_like(operator, dtype=bool)
    begin = 0
    for ishell in range(obasis["shell_types"].size):
        end = begin + get_shell_nbasis(obasis["shell_types"][ishell])
        if obasis["shell_map"][ishell] != index:
            mask[begin:end] = True
        begin = end
    operator[mask] = 0.0
    operator[:] = 0.5*(operator + operator.T)


def get_mulliken_operators(obasis):
    """Return a list of Mulliken operators for the given obasis."""
    operators = []
    olp = compute_overlap(**obasis)
    for icenter in range(obasis["centers"].shape[0]):
        operator = olp.copy()
        partition_mulliken(operator, obasis, icenter)
        operators.append(operator)
    return operators
