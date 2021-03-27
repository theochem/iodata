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


from typing import TextIO

from .common import populate_fields

from ..docstrings import document_write_input
from ..iodata import IOData
from ..periodic import num2sym


__all__ = []


default_template = """\
#n {lot}/{obasis_name} {run_type}

{title}

{charge} {spinmult}
{geometry}

"""


@document_write_input("GAUSSIAN", ['atnums', 'atcoords'],
                      ['title', 'run_type', 'lot', 'obasis_name', 'spinmult', 'charge'])
def write_input(f: TextIO, data: IOData, template: str = None, **kwargs):
    """Do not edit this docstring. It will be overwritten."""
    # initialize a dictionary with fields to replace in the template
    fields = populate_fields(data)
    # set format-specific defaults
    fields["lot"] = data.lot if data.lot is not None else 'hf'
    fields["obasis_name"] = data.obasis_name if data.obasis_name is not None else 'sto-3g'
    # convert run type to Gaussian keywords
    run_types = {"energy": "sp", "energy_force": "force", "opt": "opt", "scan": "scan",
                 "freq": "freq"}
    fields["run_type"] = run_types[fields["run_type"].lower()]
    # generate geometry (in angstrom)
    geometry = []
    for num, coord in zip(fields["atnums"], fields["atcoords"]):
        geometry.append(f"{num2sym[num]:3} {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}")
    fields["geometry"] = "\n".join(geometry)
    # get template
    if template is None:
        template = default_template
    # populate files & write input
    fields.update(kwargs)
    print(template.format(**fields), file=f)
