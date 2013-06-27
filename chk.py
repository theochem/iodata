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
'''Horton Checkpoint File Format'''


import h5py as h5

from horton.checkpoint import attribute_register


__all__ = ['load_checkpoint']


def load_checkpoint(filename, lf):
    """Load constructor arguments from a Horton checkpoint file

       **Arguments:**

       filename
            This is the file name of an HDF5 Horton checkpoint file. It may also
            be an open h5.Group object.

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

    for field in attribute_register.itervalues():
        att = field.read(chk, lf)
        d = result
        name = field.att_name
        if field.key is not None:
            d = result.setdefault(field.att_name, {})
            name = field.key
        if att is not None:
            if field.tags is None:
                d[name] = att
            else:
                d[name] = att, field.tags

    if do_close:
        chk.close()
    result['chk'] = filename
    return result


def dump_checkpoint(filename, system):
    '''Dump the system in a Horton checkpoint file.

       **Arguments:**

       filename
            This is the file name of an HDF5 Horton checkpoint file. It may also
            be an open h5.Group object.

       system
            A System instance.
    '''
    if isinstance(filename, basestring):
        chk = h5.File(filename, 'w')
        do_close = True
    else:
        chk = filename
        do_close = False

    for field_name, field in attribute_register.iteritems():
        field.write(chk, system)

    if do_close:
        chk.close()
