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
'''H5 file with lock'''


import h5py as h5, fcntl, time


__all__ = ['LockedH5File']


class LockedH5File(h5.File):
    def __init__(self, *args, **kwargs):
        '''Open an HDF5 file exclusively using flock (works only with sec2 driver)

           Except for the following two optional arguments, all arguments and
           keyword arguments are passed on to the h5.File constructor:

           count
                The number of attempts to open the file.

           wait
                The maximum number of seconds to wait between two attempts to
                open the file. [default=10]

           Two processes on the same machine typically can not open the same
           HDF5 file for writing. The second one will get an IOError because the
           HDF5 library tries to detect such cases. When the h5.File constructor
           raises no IOError, fcntl.flock is used to obtain a non-blocking lock:
           shared when mode=='r', exclusive in all other cases. This may also
           raise an IOError when two processes on different machines try to
           aquire incompatible locks. Whenever an IOError is raised, this
           constructor will wait for some time and try again, hoping that
           the other process has finished reading/writing and closed the file.
           Several attempts are made before finally giving up.

           This class guarantees that only one process is writing to the HDF5
           file (while no other processes are reading). Multiple processes may
           still read from the same HDF5 file.
        '''
        count = kwargs.pop('count', 10)
        wait = kwargs.pop('wait', 10.0)
        # first try to open the file
        for irep in xrange(count):
            try:
                h5.File.__init__(self, *args, **kwargs)
                break # When this line is reached, it worked.
            except IOError, e:
                if irep == count-1:
                    # giving up
                    raise
                else:
                    time.sleep(wait)
        # then try to get a lock
        for irep in xrange(count):
            try:
                if self.driver != 'sec2': # only works for sec2
                    raise ValueError('LockedH5File only works with HDF5 sec2 driver.')
                fd = self.fid.get_vfd_handle()
                if fd == 0:
                    raise IOError('Zero file descriptor')
                if self.mode == 'r':
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break # When this line is reached, it worked.
            except IOError, e:
                if irep == count-1:
                    # giving up
                    raise
                else:
                    time.sleep(wait)
