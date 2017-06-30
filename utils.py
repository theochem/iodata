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


__all__ = ['set_four_index_element']


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
