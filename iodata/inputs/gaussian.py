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
"""Gaussian Input Module."""

from typing import Callable, Optional, TextIO

from ..docstrings import document_write_input
from ..iodata import IOData
from ..periodic import num2sym
from ..utils import angstrom
from .common import write_input_base

__all__ = []


default_template = """\
#n {lot}/{obasis_name} {run_type}

{title}

{charge} {spinmult}
{geometry}

"""


def default_atom_line(data: IOData, iatom: int):
    """Format atom line for Gaussian input."""
    symbol = num2sym[data.atnums[iatom]]
    atcoord = data.atcoords[iatom] / angstrom
    return f"{symbol:3s} {atcoord[0]:10.6f} {atcoord[1]:10.6f} {atcoord[2]:10.6f}"


@document_write_input(
    "GAUSSIAN",
    ["atnums", "atcoords"],
    ["title", "run_type", "lot", "obasis_name", "spinmult", "charge"],
)
def write_input(
    fh: TextIO,
    data: IOData,
    template: Optional[str] = None,
    atom_line: Optional[Callable] = None,
    **kwargs,
):
    """Do not edit this docstring. It will be overwritten."""
    # Fill in some Gaussian-specific defaults and field names.
    if template is None:
        template = default_template
    if atom_line is None:
        atom_line = default_atom_line
    gaussian_keywords = {
        "energy": "sp",
        "energy_force": "force",
        "opt": "opt",
        "scan": "scan",
        "freq": "freq",
    }
    fields = {
        "lot": data.lot or "hf",
        "obasis_name": data.obasis_name or "sto-3g",
        "run_type": gaussian_keywords[(data.run_type or "energy").lower()],
    }
    # User-specifield fields have priority, may overwrite default ones.
    fields.update(kwargs)
    write_input_base(fh, data, template, atom_line, fields)
