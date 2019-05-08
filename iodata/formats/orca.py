# IODATA is an input and output module for quantum chemistry.
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
# --
"""Module for handling ORCA OUT file format."""


from typing import TextIO, Tuple

import numpy as np

from ..utils import LineIterator


__all__ = []


patterns = ['*.out']


def load(lit: LineIterator) -> dict:
    """Load several results from an orca output file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary may contain ``atnums``, ``atcoords``, and ``total_energy`` and
        corresponding values.

    """
    result = {}
    while True:
        try:
            line = next(lit)
        except StopIteration:
            # Read until the end of the file.
            break
        # Get the total number of atoms
        if line.startswith('CARTESIAN COORDINATES (ANGSTROEM)'):
            natom = _helper_number_atoms(lit)
        # Every Cartesian coordinates found are replaced with the old ones
        # to maintain the ones from the final SCF iteration in e.g. optimization run
        if line.startswith('CARTESIAN COORDINATES (A.U.)'):
            result['atnums'], result['atcoords'] = _helper_geometry(lit, natom)
        # The final SCF energy is obtained
        if line.startswith('FINAL SINGLE POINT ENERGY'):
            words = line.split()
            result['energy'] = float(words[4])
        # read also the dipole moment (commented out until key is in iodata)
        # if line.startswith('Total Dipole Moment'):
        #    dipole = np.zeros(3)
        #    dipole[0] = float(words[5])
        #    dipole[1] = float(words[6])
        #    dipole[2] = float(words[7])
        #    result['dipole'] = dipole
    return result


def _helper_number_atoms(lit: LineIterator) -> int:
    """Load list of coordinates from an ORCA output file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    natom: int
       Total number of atoms.

    """
    # skip the dashed line
    next(lit)
    natom = 0
    # Add until an empty line is found
    while next(lit).strip() != '':
        natom += 1
    return natom


def _helper_geometry(lit: TextIO, natom: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load coordinates form a ORCA output file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    atnums: int
        The atomic numbers.
    atcoords: array_like
        The atcoords in an array of size (natom, 3).

    """
    atcoords = np.zeros((natom, 3))
    atnums = np.zeros(natom)
    # skip the dashed line
    next(lit)
    # skip the titles in table
    next(lit)
    # read in the atomic number and coordinates in a.u.
    for i in range(natom):
        words = next(lit).split()
        atnums[i] = int(float(words[2]))
        atcoords[i, 0] = float(words[5])
        atcoords[i, 1] = float(words[6])
        atcoords[i, 2] = float(words[7])
    return atnums, atcoords
