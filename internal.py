# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
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
'''HORTON internal file format'''


import numpy as np, h5py as h5
from horton.io.lockedh5 import LockedH5File


__all__ = ['load_h5', 'dump_h5']


def load_h5(item):
    '''Load a (HORTON) object from an h5py File/Group

       **Arguments:**

       item
            A HD5 Dataset or group, or a filename of an HDF5 file
    '''
    if isinstance(item, basestring):
        with LockedH5File(item, 'r') as f:
            return load_h5(f)
    elif isinstance(item, h5.Dataset):
        if len(item.shape) > 0:
            # convert to a numpy array
            return np.array(item)
        else:
            # convert to a scalar
            return item[()]
    elif isinstance(item, h5.Group):
        class_name = item.attrs.get('class')
        if class_name is None:
            # assuming that an entire dictionary must be read.
            result = {}
            for key, subitem in item.iteritems():
                result[key] = load_h5(subitem)
            return result
        else:
            # special constructor. the class is found with the imp module
            cls = __import__('horton', fromlist=[class_name]).__dict__[class_name]
            return cls.from_hdf5(item)


def dump_h5(grp, data):
    '''Dump a (HORTON) object to a HDF5 file.

       grp
            A HDF5 group or a filename of a new HDF5 file.

       data
            The object to be written. This can be a dictionary of objects or
            an instance of a HORTON class that has a ``to_hdf5`` method. The
            dictionary my contain numpy arrays
    '''
    if isinstance(grp, basestring):
        with LockedH5File(grp, 'w') as f:
            dump_h5(f, data)
    elif isinstance(data, dict):
        for key, value in data.iteritems():
            # Simply overwrite old data
            if key in grp:
                del grp[key]
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, np.ndarray) or isinstance(value, basestring):
                grp[key] = value
            else:
                subgrp = grp.require_group(key)
                dump_h5(subgrp, value)
    else:
        # clear the group if anything was present
        for key in grp.keys():
            del grp[key]
        for key in grp.attrs.keys():
            del grp.attrs[key]
        data.to_hdf5(grp)
        # The following is needed to create object of the right type when
        # reading from the checkpoint:
        grp.attrs['class'] = data.__class__.__name__
