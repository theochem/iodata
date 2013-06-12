# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
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

       For each filename extension, there is a specialized load_xxx function
       that returns a dictionary with arguments for the constructor of the
       ``System`` class, except for the ``chk`` and ``lf`` arguments. Two
       additional variables may be added in the returned dictionary:
       ``permutation`` and ``signs``. These will be used to reorder and change
       the signs of all matrix elements related to the basis functions. The
       cache argument always has to be a dictionary (and not yet a Cache
       instance).
    '''
    if isinstance(filename, h5.Group) or filename.endswith('.h5'):
        from horton.io.chk import load_checkpoint
        return load_checkpoint(filename, lf)
    if filename.endswith('.xyz'):
        from horton.io.xyz import load_geom_xyz
        return load_geom_xyz(filename)
    elif filename.endswith('.fchk'):
        from horton.io.gaussian import load_fchk
        return load_fchk(filename, lf)
    elif filename.endswith('.log'):
        from horton.io.gaussian import load_operators_g09
        return load_operators_g09(filename, lf)
    elif filename.endswith('.mkl'):
        from horton.io.molekel import load_mkl
        return load_mkl(filename, lf)
    elif filename.endswith('.molden.input'):
        from horton.io.molden import load_molden
        return load_molden(filename, lf)
    elif filename.endswith('.cube'):
        from horton.io.cube import load_cube
        return load_cube(filename)
    elif os.path.basename(filename).startswith('POSCAR'):
        from horton.io.vasp import load_poscar
        return load_poscar(filename)
    elif os.path.basename(filename)[:6] in ['CHGCAR', 'AECCAR']:
        from horton.io.vasp import load_chgcar
        return load_chgcar(filename)
    elif os.path.basename(filename).startswith('LOCPOT'):
        from horton.io.vasp import load_locpot
        return load_locpot(filename)
    elif filename.endswith('.cp2k.out'):
        from horton.io.cp2k import load_atom_cp2k
        return load_atom_cp2k(filename, lf)
    elif filename.endswith('.cif'):
        from horton.io.cif import load_cif
        return load_cif(filename, lf)
    else:
        raise ValueError('Unknown file format for reading: %s' % filename)


def dump_system(filename, system):
    if isinstance(filename, h5.Group) or filename.endswith('.h5'):
        from horton.io.chk import dump_checkpoint
        dump_checkpoint(filename, system)
    elif filename.endswith('.xyz'):
        from horton.io.xyz import dump_xyz
        dump_xyz(filename, system)
    elif filename.endswith('.cube'):
        from horton.io.cube import dump_cube
        dump_cube(filename, system)
    elif filename.endswith('.cif'):
        from horton.io.cif import dump_cif
        dump_cif(filename, system)
    elif os.path.basename(filename).startswith('POSCAR'):
        from horton.io.vasp import dump_poscar
        return dump_poscar(filename, system)
    else:
        raise ValueError('Unknown file format for writing: %s' % filename)
