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
"""GAMESS punch file format."""


import numpy as np

from ..docstrings import document_load_one
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.dat']


def _read_data(lit: LineIterator) -> tuple:
    """Extract ``title``, ``symmetry`` and ``symbols`` from the punch file."""
    title = next(lit).strip()
    symmetry = next(lit).split()[0]
    # The dat file only contains symmetry-unique atoms, so we would be incapable of
    # supporting non-C1 symmetry without significant additional coding.
    if symmetry != "C1":
        raise NotImplementedError(f"Only C1 symmetry is supported. Got {symmetry}")
    symbols = []
    line = True
    while line != " $END      \n":
        line = next(lit)
        if line[0] != " ":
            symbols.append(line.split()[0])
    return title, symmetry, symbols


def _read_coordinates(lit: LineIterator, result: dict) -> tuple:
    """Extract ``numbers`` and ``coordinates`` from the punch file."""
    for i in range(2):
        next(lit)
    natom = len(result["symbols"])
    # if the data are already read before, just overwrite them
    numbers = result.get("atnums")
    if numbers is None:
        numbers = np.zeros(natom, int)
        result["atnums"] = np.zeros(natom, int)

    coordinates = result.get("atcoords")
    if coordinates is None:
        coordinates = np.zeros((natom, 3), float)
    for i in range(natom):
        words = next(lit).split()
        numbers[i] = int(float(words[1]))
        coordinates[i] = np.array([float(elem) for elem in words[2:5]]) * angstrom
    return numbers, coordinates


def _read_energy(lit: LineIterator, result: dict) -> tuple:
    """Extract ``energy`` and ``gradient`` from the punch file."""
    energy = float(next(lit).split()[1])
    natom = len(result["symbols"])
    # if the data are already read before, just overwrite them
    gradient = result.get("gradient")
    if gradient is None:
        gradient = np.zeros((natom, 3), float)
    for i in range(natom):
        words = next(lit).split()
        gradient[i] = words[2:5]
    return energy, gradient


def _read_hessian(lit: LineIterator, result: dict) -> np.ndarray:
    """Extract ``hessian`` from the punch file."""
    # check that $HESS is not already parsed
    if "athessian" in result:
        lit.error("Cannot parse $HESS twice! Make sure approximate hessian is not being parsed!")
    next(lit)
    natom = len(result["symbols"])
    hessian = np.zeros((3 * natom, 3 * natom), float)
    tmp = hessian.ravel()
    counter = 0
    while True:
        line = next(lit)
        if line == " $END\n":
            break
        line = line[5:-1]
        for j in range(len(line) // 15):
            tmp[counter] = float(line[j * 15:(j + 1) * 15])
            counter += 1
    return hessian


def _read_masses(lit: LineIterator, result: dict) -> np.ndarray:
    """Extract ``masses`` from the punch file."""
    natom = len(result["symbols"])
    masses = np.zeros(natom, float)
    counter = 0
    while counter < natom:
        words = next(lit).split()
        for word in words:
            masses[counter] = float(word)
            counter += 1
    return masses


@document_load_one("PUNCH", ['title', 'energy', 'grot', 'atgradient', 'athessian', 'atmasses',
                             'atnums', 'atcoords'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    result = dict()
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        if line == "$DATA\n":
            result["title"], result["g_rot"], result["symbols"] = _read_data(lit)
        elif line == " COORDINATES OF SYMMETRY UNIQUE ATOMS (ANGS)\n":
            result["atnums"], result["atcoords"] = _read_coordinates(lit, result)
        elif line == " $GRAD\n":
            result["energy"], result["atgradient"] = _read_energy(lit, result)
        elif line == "CAUTION, APPROXIMATE HESSIAN!\n":
            # prevent the approximate Hessian from being parsed
            while line != " $END\n":
                line = next(lit)
        elif line == " $HESS\n":
            result["athessian"] = _read_hessian(lit, result)
        elif line == "ATOMIC MASSES\n":
            result["atmasses"] = _read_masses(lit, result)
    result.pop("symbols")
    return result
