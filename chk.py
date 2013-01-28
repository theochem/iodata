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


import h5py as h5


__all__ = ['load_checkpoint']


def load_checkpoint(filename, lf):
    """Load constructor arguments from a Horton checkpoint file

       **Arguments:**

       filename
            This is the file name of an HDF5 Horton checkpoint file. It may also
            be an open h5.File object.

       lf
            A LinalgFactory instance.
    """
    result = {}
    if isinstance(filename, basestring):
        chk = h5.File(filename,'r')
        do_close = True
    else:
        chk = filename
        do_close = False
    from horton.checkpoint import register
    for field in register.itervalues():
        att = field.read(chk, lf)
        d = result
        name = field.att_name
        if field.key is not None:
            d = result.setdefault(field.att_name, {})
            name = field.key
        if att is not None:
            d[name] = att
    if do_close:
        chk.close()
    result['chk'] = filename
    return result
