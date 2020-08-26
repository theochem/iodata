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


PATTERNS = ['*.gamessout']


def _read_data(gms) -> tuple:
    """Extract ``title``, ``symmetry`` and ``symbols`` from the punch file."""
    title = next(gms).strip()
    symmetry = next(gms).split()[0]
    if symmetry != "C1":
        raise NotImplementedError("Only C1 symmetry is supported.")
    symbols = []
    line = True
    while line != " $END      \n":
        line = next(gms)
        if line[0] != " ":
            symbols.append(line.split()[0])
    return title, symmetry, symbols


def _read_coordinates(gms, res) -> tuple:
    """Extract ``numbers`` and ``coordinates`` from the punch file."""
    for i in range(2):
        next(gms)
    N = len(res["symbols"])
    # if the data are already read before, just overwrite them
    numbers = res.get("atnums")
    if numbers is None:
        numbers = np.zeros(N, int)
        res["atnums"] = numbers

    coordinates = res.get("atcoords")
    if coordinates is None:
        coordinates = np.zeros((N, 3), float)
    for i in range(N):
        words = next(gms).split()
        numbers[i] = int(float(words[1]))
        coordinates[i, 0] = float(words[2]) * angstrom
        coordinates[i, 1] = float(words[3]) * angstrom
        coordinates[i, 2] = float(words[4]) * angstrom
    return numbers, coordinates


def _read_energy(gms, res) -> tuple:
    """Extract ``energy`` and ``gradient`` from the punch file."""
    energy = float(next(gms).split()[1])
    N = len(res["symbols"])
    # if the data are already read before, just overwrite them
    gradient = res.get("gradient")
    if gradient is None:
        gradient = np.zeros((N, 3), float)
    for i in range(N):
        words = next(gms).split()
        gradient[i, 0] = float(words[2])
        gradient[i, 1] = float(words[3])
        gradient[i, 2] = float(words[4])
    return energy, gradient


def _read_hessian(gms, res) -> np.ndarray:
    """Extract ``hessian`` from the punch file."""
    assert "hessian" not in res
    line = next(gms)
    N = len(res["symbols"])
    hessian = np.zeros((3 * N, 3 * N), float)
    tmp = hessian.ravel()
    counter = 0
    while True:
        line = next(gms)
        if line == " $END\n":
            break
        line = line[5:-1]
        for j in range(len(line) // 15):
            tmp[counter] = float(line[j * 15:(j + 1) * 15])
            counter += 1
    return hessian


def _read_masses(gms, res):
    """Extract ``masses`` from the punch file."""
    N = len(res["symbols"])
    masses = np.zeros(N, float)
    counter = 0
    while counter < N:
        words = next(gms).split()
        for word in words:
            masses[counter] = float(word)
            counter += 1
    return masses


@document_load_one("PUNCH", ['title', 'energy', 'grot', 'atgradient', 'athessian', 'atmasses'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    res = dict()
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        if line == "$DATA\n":
            res["title"], res["g_rot"], res["symbols"] = _read_data(lit)
        elif line == " COORDINATES OF SYMMETRY UNIQUE ATOMS (ANGS)\n":
            res["atnums"], res["atcoords"] = _read_coordinates(lit, res)
        elif line == " $GRAD\n":
            res["energy"], res["atgradient"] = _read_energy(lit, res)
        elif line == "CAUTION, APPROXIMATE HESSIAN!\n":
            # prevent the approximate Hessian from being parsed
            while line != " $END\n":
                line = next(lit)
        elif line == " $HESS\n":
            res["athessian"] = _read_hessian(lit, res)
        elif line == "ATOMIC MASSES\n":
            res["atmasses"] = _read_masses(lit, res)
    res.pop("symbols")
    return res
