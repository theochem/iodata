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
'''Dump FCIDUMP file to disc using the standard Molpro format of FCIDUMP.
   One- and two-electron integrals are stored in chemical notation!.
'''


__all__ = ['dump_fci']


def dump_fci(lf, one, two, ecore, exps, filename='./FCIDUMP', **kwargs):
    '''Write one- and two-electron integrals using Molpro file conventions.
       Works only for restricted wave function and no active space (all orbitals
       are active)

       **Arguments:**

       one/two
            One and four-index integrals in the AO basis.

       exps
            The AO/MO expansion coefficients. An Expansion instance.

       filename
            The filename of the fcidump file. Default "FCIDUMP".
    '''
    name = ["nel", "ms2", "norb",]
    for name, value in kwargs.items():
        if name not in names:
            raise ValueError("Unknown keyword argument %s" % name)
    nel = kwargs.get('nel', 0)
    ms2 = kwargs.get('ms2', 0)
    norb = kwargs.get('norb', one.nbasis)

    two_mo = lf.create_four_index()
    two_mo.apply_four_index_transform_tensordot(two, exps)
    one_mo = lf.create_two_index()
    one_mo.apply_2index_trans(one, exps)

    with open(filename, 'w') as f:
        print >> f, ' &FCI NORB= %i,NELEC=%i,MS2= %i,' %(norb, nel, ms2)
        print >> f, '  ORBSYM= '+",".join(str(1) for v in range(norb))+","
        print >> f, '  ISYM=1'
        print >> f, ' &END'

        for i in range(norb):
            for j in range(0,i+1):
                for k in range(norb):
                    for l in range(0,k+1):
                        print >> f, '%16.12f %4i %4i %4i %4i' %(two_mo.get_element(i,k,j,l), (i+1), (j+1), (k+1), (l+1))
        for i in range(norb):
            for j in range(0,i+1):
                print >> f, '%16.12f %4i %4i %4i %4i' %(one_mo.get_element(i,j), (i+1), (j+1), 0, 0)

        print >> f, '%16.12f %4i %4i %4i %4i' %(ecore, 0, 0, 0, 0)
