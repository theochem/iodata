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
"""Input and Output Module."""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.post0"
    __version_tuple__ = (0, 0, 0, "a-dev")


from .api import dump_many, dump_one, load_many, load_one, write_input
from .iodata import IOData

__all__ = ("IOData", "dump_many", "dump_one", "load_many", "load_one", "write_input")
