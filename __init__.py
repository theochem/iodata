# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
'''Input and output routines.

   All input routines begin with ``load_``. All output routines begin with
   ``dump_``.

   This package also contains a smart routine, ``load_system_args``, which makes
   it easier to load molecular data from various file formats. It uses to
   extension of the filename to figure out what the file format is.
'''


from horton.io.common import *
from horton.io.cube import *
from horton.io.gaussian import *
from horton.io.molden import *
from horton.io.molekel import *
from horton.io.smart import *
from horton.io.xyz import *
