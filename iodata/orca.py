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
# pragma pylint: disable=wrong-import-order,invalid-name
"""Module for handling GAUSSIAN LOG file format."""


import numpy as np

from typing import Dict, TextIO

from .utils import set_four_index_element


__all__ = ['load']


patterns = ['*.out']


def load(filename: str) -> Dict:
    """Load several results from an orca output file.

    Parameters
    ----------
    filename : str
        The ORCA LOG filename.

    Returns
    -------
    out : dict
        Output dictionary may contain ``olp``, ``kin``, ``na`` and/or ``er`` keys and
        corresponding values.

    Notes
    -----

    """
    with open(filename) as f:
        result = {}
        for line in f:
            # Get the total number of atoms
            if line.startswith('CARTESIAN COORDINATES (ANGSTROEM)'):
                natom = _helper_number_atoms(f)
            # Every Cartesian coordinates found are replaced with the old ones
            # to maintain the ones from the final SCF iteration in e.g. optimization run
            if line.startswith('CARTESIAN COORDINATES (A.U.)'):
                result['numbers'], result['coordinates'] = _helper_geometry(f, natom)
            # The final SCF energy is obtained
            if line.startswith('FINAL SINGLE POINT ENERGY'):
                words =line.split()
                result['energy'] = float(words[4])
            # read also the dipole moment (commented out until key is in iodata)
            #if line.startswith('Total Dipole Moment'):
            #    dipole = np.zeros(3)
            #    dipole[0] = float(words[5])
            #    dipole[1] = float(words[6])
            #    dipole[2] = float(words[7])
            #    result['dipole'] = dipole
        return result

def _helper_number_atoms(f: TextIO) -> int:
    """Load list of coordinates from an ORCA output file format.

        Parameters
        ----------
        f
            A ORCA file object (in read mode).

        Returns
        -------
        out: int
            Total number of atoms.

    """
    # skip the dashed line
    next(f)
    natom = 0
    # Add until an empty line is found
    while next(f).strip() != '':
        natom += 1
    return natom



def _helper_geometry(f: TextIO, natom: int) -> (int, np.ndarray):
    """Load coordinates form a ORCA output file format.

    Parameters
    ----------
    f
        A ORCA file object (in read mode).

    Returns
    -------
    out: int
        The atomic numbers
        array_like
        The coordinates in an array of size (natom, 3).

    """
    coordinates = np.zeros((natom, 3))
    numbers = np.zeros(natom)
    # skip the dashed line
    next(f)
    # skip the titles in table
    next(f)
    # read in the atomic number and coordinates in a.u.
    for i in range(natom):
        words = next(f).split()
        numbers[i] = int(float(words[2]))
        coordinates[i,0] = float(words[5])
        coordinates[i,1] = float(words[6])
        coordinates[i,2] = float(words[7])
    return (numbers, coordinates)
    # read in the atomic number and coordinates in a.u.
    for i in range(natom):
        words = next(f).split()
        numbers[i] = int(words[2])
        coordinates[i,0] = float(words[5])
        coordinates[i,1] = float(words[6])
        coordinates[i,2] = float(words[7])
    return (numbers, coordinates)
