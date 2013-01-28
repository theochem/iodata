# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
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


import numpy as np
from horton import *


__all__ = ['compute_mulliken']


def compute_mulliken(sys):
    basis_count = np.zeros(sys.natom, int)
    for ishell in xrange(sys.obasis.nshell):
        basis_count[sys.obasis.shell_map[ishell]] += get_shell_nbasis(sys.obasis.shell_types[ishell])

    begin = 0
    dm = sys.wfn.dm_full
    pops = []
    for i in xrange(sys.natom):
        end = begin + basis_count[i]
        pop = sys.get_overlap().copy()
        pop._array[:begin,:begin] = 0.0
        pop._array[:begin,end:] = 0.0
        pop._array[end:,:begin] = 0.0
        pop._array[end:,end:] = 0.0
        pop._array[begin:end,:begin] *= 0.5
        pop._array[begin:end,end:] *= 0.5
        pop._array[end:,begin:end] *= 0.5
        pop._array[:begin,begin:end] *= 0.5
        pops.append(pop.expectation_value(dm))
        begin = end
    return sys.numbers - np.array(pops)
