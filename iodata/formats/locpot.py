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
"""VASP 5 LOCPOT file format.

This format is used by `VASP 5.X <https://www.vasp.at/>`_ and
`VESTA <http://jp-minerals.org/vesta/en/>`_.

Note that even though the ``CHGCAR`` and ``LOCPOT`` files look very similar, they require
different conversions to atomic units.
"""


from ..docstrings import document_load_one
from ..utils import electronvolt, LineIterator
from .chgcar import _load_vasp_grid


__all__ = []


PATTERNS = ['LOCPOT*']


@document_load_one("VASP 5 LOCPOT", ['atcoords', 'atnums', 'cellvecs', 'cube', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    result = _load_vasp_grid(lit)
    # convert locpot to atomic units
    result['cube'].data[:] *= electronvolt
    return result
