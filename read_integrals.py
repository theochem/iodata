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


__all__ = ['read_integrals']


def read_integrals(lf, filename='./FCIDUMP'):
    '''Read one- and two-electron integrals using Molpro file conventions.
       Works only for restricted wave functions.

       **Arguments:**

       lf
            A linear algebra factory.

       filename
            The filename of the fcidump file. Default "FCIDUMP".
    '''
    one_mo = lf.create_two_index()
    two_mo = lf.create_four_index()
    arrays = []
    with open(filename) as f:
        for line in f.readlines():
            arrays.append((line.split()))
    i = -1
    while True:
        i = i+1
        if arrays[i][0]=="&END":
            while True:
                i = i+1
                if arrays[i][4] == '0':
                    if arrays[i][1] == '0':
                        coreenergy = float(arrays[i][0])
                        return one_mo, two_mo, coreenergy
                    ii = int(arrays[i][1])-1
                    ij = int(arrays[i][2])-1
                    one_mo.set_element(ii,ij,float(arrays[i][0]))
                else:
                    ii = int(arrays[i][1])-1
                    ij = int(arrays[i][2])-1
                    ik = int(arrays[i][3])-1
                    il = int(arrays[i][4])-1
                    two_mo.set_element(ii,ik,ij,il,float(arrays[i][0]))
            break
    return one_mo, two_mo, coreenergy
