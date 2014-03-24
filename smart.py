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
'''Input/output dispatcher for different file formats

   The ``load_smart`` and ``dump_smart`` functions read/write molecule data
   from/to a file. The format is deduced from the prefix or extension of the
   filename. ``load_smart`` returns a dictionary of data while ``dump_smart``
   has such a dictionary as argument. These dictionaries may contain the
   following keys:

   cell
        A Cell object that describes the (generally triclinic) periodic
        boundary conditions.

   coordinates
        A (N, 3) float numpy array with Cartesian coordinates of the atoms.

   cube_data
        Data on a uniform grid (defined by ugrid).

   energy
        The total energy (electronic+nn) of the molecule

   er
        The electron repulsion two-body operator

   esp_charges
        Charges fitted to the electrostatic potential

   grid
        An integration grid (usually a UniformGrid instance)

   kin
        The kinetic energy operator

   lf
        A LinalgFactory instance.

   mulliken_charges
        Mulliken AIM charges

   na
        The nuclear attraction operator

   npa_charges
        Natural charges

   numbers
        A (N,) int numpy vector with the atomic numbers.

   obasis
        An instance of the GOBasis class.

   olp
        The overlap operator

   permutation
        The permutation applied to the basis functions.

   pseudo_numbers
        The core charges of the pseudo potential, if applicable

   signs
        The sign changes applied to the basis functions.

   symmetry
        An instance of the Symmetry class, describing the geometric symmetry.

   links
        A mapping between the atoms in the primitive unit and the
        crystallographic unit.

   wfn
        A WFN object.
'''


from horton.matrix import DenseLinalgFactory
import h5py as h5, os


__all__ = ['load_smart', 'dump_smart']


def load_smart(*filenames, **kwargs):
    '''Load a molecular data from a file.

       **Arguments:**

       filename1, filename2, ...
            The files to load molecule data from. When multiple files are given,
            data from the first file is overwritten by data from the second,
            etc. When one file contains sign and permutation changes for the
            orbital basis, these changes will be applied to data from all other
            files.

       **Optional arguments:**

       lf
            A LinalgFactory instance. DenseLinalgFactory is used as default.

       This routine uses the extension or prefix of the filename to determine the file
       format. It returns a dictionary with data loaded from the file.

       For each file format, a specialized load_xxx function is called that
       returns a dictionary with data from the file.
    '''
    result = {}

    lf = kwargs.pop('lf', None)
    if lf is None:
        lf = DenseLinalgFactory()
    if len(kwargs) > 0:
        raise TypeError('Keyword argument(s) not supported: %s' % lf.keys())
    result['lf'] = lf

    for filename in filenames:
        if isinstance(filename, h5.Group) or filename.endswith('.h5'):
            from horton.io.internal import load_h5
            result.update(load_h5(filename, lf))
        elif filename.endswith('.xyz'):
            from horton.io.xyz import load_xyz
            result.update(load_xyz(filename))
        elif filename.endswith('.fchk'):
            from horton.io.gaussian import load_fchk
            result.update(load_fchk(filename, lf))
        elif filename.endswith('.log'):
            from horton.io.gaussian import load_operators_g09
            result.update(load_operators_g09(filename, lf))
        elif filename.endswith('.mkl'):
            from horton.io.molekel import load_mkl
            result.update(load_mkl(filename, lf))
        elif filename.endswith('.molden.input'):
            from horton.io.molden import load_molden
            result.update(load_molden(filename, lf))
        elif filename.endswith('.cube'):
            from horton.io.cube import load_cube
            result.update(load_cube(filename))
        elif filename.endswith('.wfn'):
            from horton.io.wfn import load_wfn
            result.update(load_wfn(filename, lf))
        elif os.path.basename(filename).startswith('POSCAR'):
            from horton.io.vasp import load_poscar
            result.update(load_poscar(filename))
        elif os.path.basename(filename)[:6] in ['CHGCAR', 'AECCAR']:
            from horton.io.vasp import load_chgcar
            result.update(load_chgcar(filename))
        elif os.path.basename(filename).startswith('LOCPOT'):
            from horton.io.vasp import load_locpot
            result.update(load_locpot(filename))
        elif filename.endswith('.cp2k.out'):
            from horton.io.cp2k import load_atom_cp2k
            result.update(load_atom_cp2k(filename, lf))
        elif filename.endswith('.cif'):
            from horton.io.cif import load_cif
            result.update(load_cif(filename, lf))
        else:
            raise ValueError('Unknown file format for reading: %s' % filename)

    # Apply changes in orbital permutation and sign conventions
    if 'permutation' in result:
        for key in 'olp', 'kin', 'na', 'er', 'wfn':
            if key in result:
                result[key].apply_basis_permutation(result['permutation'])
        del result['permutation']
    if 'signs' in result:
        for key in 'olp', 'kin', 'na', 'er', 'wfn':
            if key in result:
                result[key].apply_basis_signs(result['signs'])
        del result['signs']
    return result


def dump_smart(filename, data):
    '''Write molecule data to a file

       **Arguments:**

       filename
            The file to load the geometry from

       data
            A dictionary containing all the data. When some elements of the
            dictionary are not supported by the file format, they will be
            ignored. When for a given format, required elements are missing from
            the dictionary, an error is raised.

       This routine uses the extension or prefix of the filename to determine
       the file format. For each file format, a specialized dump_xxx function is
       called that does the real work.
    '''
    if isinstance(filename, h5.Group) or filename.endswith('.h5'):
        from horton.io.internal import dump_h5
        if 'lf' in data:
            data = data.copy()
            data.pop('lf')
        dump_h5(filename, data)
    elif filename.endswith('.xyz'):
        from horton.io.xyz import dump_xyz
        dump_xyz(filename, data)
    elif filename.endswith('.cube'):
        from horton.io.cube import dump_cube
        dump_cube(filename, data)
    elif filename.endswith('.cif'):
        from horton.io.cif import dump_cif
        dump_cif(filename, data)
    elif filename.endswith('.molden.input'):
        from horton.io.molden import dump_molden
        dump_molden(filename, data)
    elif os.path.basename(filename).startswith('POSCAR'):
        from horton.io.vasp import dump_poscar
        return dump_poscar(filename, data)
    else:
        raise ValueError('Unknown file format for writing: %s' % filename)
