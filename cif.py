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


from horton import angstrom, deg, periodic


def dump_cif(filename, system):
    if system.cell is None or system.cell.nvec != 3:
        raise ValueError('The CIF format only supports 3D periodic systems.')
    with open(filename, 'w') as f:
        print >> f, 'data_foobar'
        print >> f, '_symmetry_space_group_name_H-M       \'P1\''
        print >> f, '_audit_creation_method            \'Horton\''
        print >> f, '_symmetry_Int_Tables_number       1'
        print >> f, '_symmetry_cell_setting            triclinic'
        print >> f, 'loop_'
        print >> f, '_symmetry_equiv_pos_as_xyz'
        print >> f, '  x,y,z'
        lengths, angles = system.cell.parameters
        print >> f, '_cell_length_a     %12.6f' % (lengths[0]/angstrom)
        print >> f, '_cell_length_b     %12.6f' % (lengths[1]/angstrom)
        print >> f, '_cell_length_c     %12.6f' % (lengths[2]/angstrom)
        print >> f, '_cell_angle_alpha  %12.6f' % (angles[0]/deg)
        print >> f, '_cell_angle_beta   %12.6f' % (angles[1]/deg)
        print >> f, '_cell_angle_gamma  %12.6f' % (angles[2]/deg)
        print >> f, 'loop_'
        print >> f, '_atom_site_label'
        print >> f, '_atom_site_type_symbol'
        print >> f, '_atom_site_fract_x'
        print >> f, '_atom_site_fract_y'
        print >> f, '_atom_site_fract_z'
        for i in xrange(system.natom):
            fx, fy, fz = system.cell.to_frac(system.coordinates[i])
            symbol = periodic[system.numbers[i]].symbol
            label = symbol+str(i+1)
            print >> f, '%10s %3s % 12.6f % 12.6f % 12.6f' % (label, symbol, fx, fy, fz)
