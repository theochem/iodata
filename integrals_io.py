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
   One- and two-electron integrals are stored in chemists' notation!.
'''

from horton.orbital_utils import transform_integrals
from horton.utils import check_type, check_options


__all__ = ['integrals_from_file', 'integrals_to_file']


def integrals_from_file(lf, filename='./FCIDUMP'):
    '''Read one- and two-electron integrals using Molpro file conventions.
       Works only for restricted wavefunctions. Number of basis functions in
       integrals must be equal to number of basis functions in lf

       **Arguments:**

       lf
            A linear algebra factory. Must be of type DenseLinalgFactory.

       **Optional arguments:**

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
        #
        # Get to first line containing integrals. Ignore everything above
        #
        i = i+1
        #
        # Molpro file format changed in version 2012
        #
        if arrays[i][0]=="&END" or arrays[i][0]=="/END" or arrays[i][0]=="/":
            while True:
                i = i+1
                if i>=len(arrays):
                    raise ValueError('Reached end of file while reading integrals. File %s might be incomplete or corrupted. Please check for errors.' %filename)
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


def integrals_to_file(lf, one, two, ecore, orb, filename='./FCIDUMP', **kwargs):
    '''Write one- and two-electron integrals using Molpro file conventions.
       If ncore/nactive are specified, the Hamiltonian within the active space
       is written to file.

       Works only for restricted wavefunctions.

       **Arguments:**

       one/two
            One and two-electron integrals.

       orb
            The MO expansion coefficients. An Expansion instance. If None,
            integrals are not transformed into new basis.

       **Optional arguments:**

       filename
            The filename of the fcidump file. Default "FCIDUMP".

       **Keyword arguments:**

       nel
            The number of active electrons (int) (default 0)

       ncore
            The number of frozen core orbitals (int) (default 0)

       ms2
            The spin multiplicity (int) (default 0)

       nactive
            The number of active orbitals (int) (default nbasis)

       indextrans
            4-index transformation (str). One of ``tensordot``, ``einsum``
    '''
    names = []
    def _helper(x,y):
        names.append(x)
        return kwargs.get(x,y)
    nel = _helper('nel', 0)
    ncore = _helper('ncore', 0)
    ms2 = _helper('ms2', 0.0)
    nactive = _helper('nactive', one.nbasis)
    indextrans = _helper('indextrans', 'tensordot')
    for name, value in kwargs.items():
        if name not in names:
            raise ValueError("Unknown keyword argument %s" % name)

    #
    # Check type/option of keyword arguments
    #
    check_type('nel', nel, int)
    check_type('ncore', ncore, int)
    check_type('nactive', nactive, int)
    check_type('ms2', ms2, int, float)
    check_options('indextrans', indextrans, 'tensordot', 'einsum')
    if nel < 0 or ncore < 0 or ms2 < 0 or nactive < 0:
        raise ValueError('nel, ncore, ms2, and nactive must be larger equal 0!')
    if nactive+ncore > one.nbasis:
        raise ValueError('More active orbitals than basis functions!')

    if orb:
        #
        # No need to check orb. This is done in transform_integrals function
        #
        one_mo, two_mo = transform_integrals(one, two, indextrans, orb)
    else:
        one_mo = [one]
        two_mo = [two]

    #
    # Account for core electrons
    #
    norb = one.nbasis
    core_ = 0.0
    if ncore > 0:
        core_ += one_mo[0].trace(0, ncore, 0, ncore)*2.0
        tmp = two_mo[0].slice_to_two('abab->ab', None, 2.0, True, 0, ncore, 0, ncore, 0, ncore, 0, ncore)
        core_ += tmp.sum()
        #
        # exchange part:
        #
        tmp = two_mo[0].slice_to_two('abba->ab', None,-1.0, True, 0, ncore, 0, ncore, 0, ncore, 0, ncore)
        core_ += tmp.sum()

        one_mo_ = one_mo[0].new()
        two_mo[0].contract_to_two('abcb->ac', one_mo_, 2.0, True, 0, norb, 0, ncore, 0, norb, 0, ncore)
        #
        # exchange part:
        #
        two_mo[0].contract_to_two('abbc->ac', one_mo_,-1.0, False, 0, norb, 0, ncore, 0, ncore, 0, norb)
        one_mo[0].iadd(one_mo_, 1.0)

    with open(filename, 'w') as f:
        print >> f, ' &FCI NORB=%i,NELEC=%i,MS2=%i,' %(nactive, nel, ms2)
        print >> f, '  ORBSYM= '+",".join(str(1) for v in range(nactive))+","
        print >> f, '  ISYM=1'
        print >> f, ' &END'

        for i in range(ncore,(ncore+nactive)):
            for j in range(ncore,i+1):
                for k in range(ncore,(ncore+nactive)):
                    for l in range(ncore,k+1):
                        if (i+1)*(j+1) >= (k+1)*(l+1):
                            print >> f, '%16.12f %4i %4i %4i %4i' %(two_mo[0].get_element(i,k,j,l), (i+1-ncore), (j+1-ncore), (k+1-ncore), (l+1-ncore))
        for i in range(ncore, (ncore+nactive)):
            for j in range(ncore,i+1):
                print >> f, '%16.12f %4i %4i %4i %4i' %(one_mo[0].get_element(i,j), (i+1-ncore), (j+1-ncore), 0, 0)

        print >> f, '%16.12f %4i %4i %4i %4i' %(ecore+core_, 0, 0, 0, 0)
