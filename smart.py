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


import h5py as h5, os


__all__ = ['load_system_args', 'dump_system']


# TODO: let all load functions return a dictionary

def load_system_args(filename, lf):
    '''Load a molecular data from a file.

       **Argument:**

       filename
            The file to load the geometry from

       lf
            A LinalgFactory instance.

       This routine uses the extension of the filename to determine the file
       format. It returns a dictionary with constructor arguments for the System
       class.
    '''
    if isinstance(filename, h5.File) or filename.endswith('.h5'):
        from horton.io.chk import load_checkpoint
        return load_checkpoint(filename, lf)
    if filename.endswith('.xyz'):
        from horton.io.xyz import load_geom_xyz
        coordinates, numbers = load_geom_xyz(filename)
        return {'coordinates': coordinates, 'numbers': numbers}
    elif filename.endswith('.fchk'):
        from horton.io.gaussian import load_fchk
        return load_fchk(filename, lf)
    elif filename.endswith('.log'):
        from horton.io.gaussian import load_operators_g09
        overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(filename, lf)
        operators = {}
        if overlap is not None:
            operators['olp'] = overlap
        if kinetic is not None:
            operators['kin'] = kinetic
        if nuclear_attraction is not None:
            operators['na'] = nuclear_attraction
        if electronic_repulsion is not None:
            operators['er'] = electronic_repulsion
        return {'operators': operators}
    elif filename.endswith('.mkl'):
        from horton.io.molekel import load_mkl
        coordinates, numbers, obasis, wfn, signs = load_mkl(filename, lf)
        return {
            'coordinates': coordinates, 'numbers': numbers, 'obasis': obasis,
            'wfn': wfn, 'signs': signs,
        }
    elif filename.endswith('.molden.input'):
        from horton.io.molden import load_molden
        coordinates, numbers, obasis, wfn, signs = load_molden(filename, lf)
        return {
            'coordinates': coordinates, 'numbers': numbers, 'obasis': obasis,
            'wfn': wfn, 'signs': signs,
        }
    elif filename.endswith('.cube'):
        from horton.io.cube import load_cube
        return load_cube(filename)
    elif os.path.basename(filename)[:6] in ['CHGCAR', 'AECCAR']:
        from horton.io.vasp import load_chgcar
        return load_chgcar(filename)
    elif os.path.basename(filename)[:6] == 'LOCPOT':
        from horton.io.vasp import load_locpot
        return load_locpot(filename)
    elif filename.endswith('.cp2k.out'):
        from horton.io.cp2k import load_atom_cp2k
        return load_atom_cp2k(filename, lf)
    else:
        raise ValueError('Unknown file format for reading: %s' % filename)


def dump_system(filename, system):
    if filename.endswith('.cube'):
        from horton.io.cube import dump_cube
        dump_cube(filename, system)
    else:
        raise ValueError('Unknown file format for writing: %s' % filename)
