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
"""Module for handling GAUSSIAN LOG file format."""


import numpy as np

from ..utils import set_four_index_element, LineIterator


__all__ = []


patterns = ['*.log']


def load(lit: LineIterator) -> dict:
    """Load several two- and four-index operators from a GAUSSIAN09 LOG file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary may contain ``olp``, ``kin``, ``na`` and/or ``er`` keys and
        corresponding values.

    Notes
    -----
    1. The following two-index operators are loaded if present:
       overlap, kinetic, nuclear attraction.
    2. The following four-index operator is loaded if present: electrostatic repulsion.
       In order to make all these matrices present in the GAUSSIAN LOG file, the following
       commands must be used in the GAUSSIAN INPUT file:
       ```scf(conventional) iop(3/33=5) extralinks=l316 iop(3/27=999)```

    """
    # First get the line with the number of orbital basis functions
    for line in lit:
        if line.startswith('    NBasis ='):
            nbasis = int(line[12:18])
            break

    # Then load the two- and four-index operators. This part is written such
    # that it does not make any assumptions about the order in which these
    # operators are printed.
    result = {}
    while True:
        line = next(lit)
        if line.startswith(" Normal termination of Gaussian"):
            break
        elif line.startswith(' *** Overlap ***'):
            result['olp'] = _load_twoindex_g09(lit, nbasis)
        elif line.startswith(' *** Kinetic Energy ***'):
            result['kin'] = _load_twoindex_g09(lit, nbasis)
        elif line.startswith(' ***** Potential Energy *****'):
            result['na'] = _load_twoindex_g09(lit, nbasis)
        elif line.startswith(' *** Dumping Two-Electron integrals ***'):
            result['er'] = _load_fourindex_g09(lit, nbasis)
    return result


def _load_twoindex_g09(lit: LineIterator, nbasis: int) -> np.ndarray:
    """Load a two-index operator from a GAUSSIAN LOG file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    nbasis
        The number of atomic orbital basis functions.

    Returns
    -------
    out: array_like
        The output (nbasis, nbasis) array of operator.

    """
    result = np.zeros((nbasis, nbasis))
    block_counter = 0
    while block_counter < nbasis:
        # skip the header line
        next(lit)
        # determine the number of rows in this part
        nrow = nbasis - block_counter
        for i in range(nrow):
            words = next(lit).split()[1:]
            for j, word in enumerate(words):
                value = float(word.replace('D', 'E'))
                result[i + block_counter, j + block_counter] = value
                result[j + block_counter, i + block_counter] = value
        block_counter += 5
    return result


def _load_fourindex_g09(lit: LineIterator, nbasis: int) -> np.ndarray:
    """Load a four-index operator from a GAUSSIAN LOG file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    nbasis
        The number of atomic orbital basis functions.

    Returns
    -------
    out: array_like
        The (nbasis, nbasis, nbasis, nbasis) array of operator.

    """
    result = np.zeros((nbasis, nbasis, nbasis, nbasis))
    # Skip first six lines
    for i in range(6):
        next(lit)
    # Start reading elements until a line is encountered that does not start
    # with ' I='
    for line in lit:
        if not line.startswith(' I='):
            break
        # print line[3:7], line[9:13], line[15:19], line[21:25], line[28:].replace('D', 'E')
        i = int(line[3:7]) - 1
        j = int(line[9:13]) - 1
        k = int(line[15:19]) - 1
        l = int(line[21:25]) - 1
        value = float(line[29:].replace('D', 'E'))
        # Gaussian uses the chemists notation for the 4-center indexes. HORTON
        # uses the physicists notation.
        set_four_index_element(result, i, k, j, l, value)
    return result
