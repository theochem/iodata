# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Module for handling MOLPRO file formal."""


from typing import Dict

import numpy as np

from ..utils import set_four_index_element, LineIterator


__all__ = ['load', 'dump']


patterns = ['*FCIDUMP*']


def load(lit: LineIterator) -> Dict:
    """Load one- and two-electron integrals from a MOLPRO 2012 FCIDUMP file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``nelec``, ``ms2``, ``one_mo``, ``two_mo`` & ``core_energy``
        keys and their corresponding values.

    Notes
    -----
    1. This function works only for restricted wave-functions.
    2. One- and two-electron integrals are stored in chemists' notation in an FCIDUMP file,
       while HORTON internally uses Physicist's notation.
    3. Keep in mind that the FCIDUMP format changed in MOLPRO 2012, so files generated with
       older versions are not supported.

    """
    # check header
    line = next(lit)
    if not line.startswith(' &FCI NORB='):
        lit.error('Incorrect file header')

    # read info from header
    words = line[5:].split(',')
    header_info = {}
    for word in words:
        if word.count('=') == 1:
            key, value = word.split('=')
            header_info[key.strip()] = value.strip()
    nbasis = int(header_info['NORB'])
    nelec = int(header_info['NELEC'])
    ms2 = int(header_info['MS2'])

    # skip rest of header
    for line in lit:
        words = line.split()
        if words[0] == "&END" or words[0] == "/END" or words[0] == "/":
            break

    # read the integrals
    one_mo = np.zeros((nbasis, nbasis))
    two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis))
    core_energy = 0.0

    for line in lit:
        words = line.split()
        if len(words) != 5:
            lit.error('Expecting 5 fields on each data line in FCIDUMP')
        value = float(words[0])
        if words[3] != '0':
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            ik = int(words[3]) - 1
            il = int(words[4]) - 1
            # Uncomment the following line if you want to assert that the
            # FCIDUMP file does not contain duplicate 4-index entries.
            # assert two_mo.get_element(ii,ik,ij,il) == 0.0
            set_four_index_element(two_mo, ii, ik, ij, il, value)
        elif words[1] != '0':
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            one_mo[ii, ij] = value
            one_mo[ij, ii] = value
        else:
            core_energy = value

    return {
        'nelec': nelec,
        'ms2': ms2,
        'one_mo': one_mo,
        'two_mo': two_mo,
        'core_energy': core_energy,
    }


def dump(filename: str, data: 'IOData'):
    """Write one- and two-electron integrals into a MOLPRO 2012 FCIDUMP file format.

    Parameters
    ----------
    filename : str
        The MOLPRO 2012 FCIDUMP filename.
    data : IOData
        An IOData instance which must contain ``one_mo`` & ``two_mo`` attributes.
        It may contain ``core_energy``, ``nelec`` and ``ms`` attributes.

    Notes
    -----
    1. This function works only for restricted wave-functions.
    2. One- and two-electron integrals are stored in chemists' notation in an FCIDUMP file,
       while HORTON internally uses Physicist's notation.
    3. Keep in mind that the FCIDUMP format changed in MOLPRO 2012, so files generated with
       older versions are not supported.

    """
    with open(filename, 'w') as f:
        one_mo = data.one_mo
        two_mo = data.two_mo
        nactive = one_mo.shape[0]
        core_energy = getattr(data, 'core_energy', 0.0)
        nelec = getattr(data, 'nelec', 0)
        ms2 = getattr(data, 'ms2', 0)

        # Write header
        print(f' &FCI NORB={nactive:d},NELEC={nelec:d},MS2={ms2:d},', file=f)
        print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
        print('  ISYM=1', file=f)
        print(' &END', file=f)

        # Write integrals and core energy
        for i in range(nactive):  # pylint: disable=too-many-nested-blocks
            for j in range(i + 1):
                for k in range(nactive):
                    for l in range(k + 1):
                        if (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l:
                            value = two_mo[i, k, j, l]
                            if value != 0.0:
                                print(f'{value:23.16e} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}', file=f)
        for i in range(nactive):
            for j in range(i + 1):
                value = one_mo[i, j]
                if value != 0.0:
                    print(f'{value:23.16e} {i+1:4d} {j+1:4d} {0:4d} {0:4d}', file=f)
        if core_energy != 0.0:
            print(f'{core_energy:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}', file=f)
