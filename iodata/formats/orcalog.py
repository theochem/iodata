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
"""Orca output file format."""


from typing import TextIO, Tuple

import numpy as np

from ..docstrings import document_load_one
from ..utils import LineIterator


__all__ = []


PATTERNS = ['*.out']


@document_load_one("Orca output", ['atcoords', 'atnums', 'energy', 'moments', 'extra'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
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
        # Read the energies of each SCF cycle in iodata.extra
        if line.startswith('SCF ITERATIONS'):
            scf_energies = _helper_scf_energies(lit)
            result['extra'] = {'scf_energies': scf_energies}
        # The final SCF energy is obtained
        if line.startswith('FINAL SINGLE POINT ENERGY'):
            words = line.split()
            result['energy'] = float(words[4])
        # Read also the dipole moment
        if line.startswith('Total Dipole Moment'):
            words = line.split()
            dipole = np.array([float(words[4]), float(words[5]), float(words[6])])
            result['moments'] = {(1, 'c'): dipole}
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


def _helper_scf_energies(lit: TextIO) -> Tuple[np.ndarray, np.ndarray]:
    """Load energies from each SCF cycle from a ORCA output file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    energies: array_like
        The energies of each scf cycle in 1D-array.

    """
    energies = []
    line = next(lit)
    # read the next line until blank line
    while line.strip() != '':
        words = line.split()
        if words[0].isdigit():
            energies.append(float(words[1]))
        line = next(lit)
    return np.asarray(energies)
