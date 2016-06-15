# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
'''Crystalographic Information File format'''



import shlex, numpy as np

from horton.units import angstrom, deg
from horton.periodic import periodic
from horton.cext import Cell
from horton.symmetry import Symmetry


__all__ = ['dump_cif', 'iter_equiv_pos_terms', 'equiv_pos_to_generator', 'load_cif']


def dump_cif(filename, data):
    '''Write a CIF file

       **Arguments:**

       filename
            The name of the new CIF file.

       data
            An IOData instance. Must contain: ``cell``, ``coordinates``,
            ``numbers``. May contain ``title``.

    '''
    if data.cell.nvec != 3:
        raise TypeError('The CIF format only supports 3D periodic systems.')
    with open(filename, 'w') as f:
        print >> f, 'data_%s' % getattr(data, 'title', 'Created with HORTON').replace(' ', '_')
        print >> f, '_symmetry_space_group_name_H-M       \'P1\''
        print >> f, '_audit_creation_method            \'HORTON\''
        print >> f, '_symmetry_Int_Tables_number       1'
        print >> f, '_symmetry_cell_setting            triclinic'
        print >> f, 'loop_'
        print >> f, '_symmetry_equiv_pos_as_xyz'
        print >> f, '  x,y,z'
        lengths, angles = data.cell.parameters
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
        for i in xrange(data.natom):
            fx, fy, fz = data.cell.to_frac(data.coordinates[i])
            symbol = periodic[data.numbers[i]].symbol
            label = symbol+str(i+1)
            print >> f, '%10s %3s % 12.6f % 12.6f % 12.6f' % (label, symbol, fx, fy, fz)


class IterRelevantCIFLines(object):
    '''A wrapper that reads lines from the CIF file.

       Irrelevant lines are ignored and a rewind method is present such that
       one can easily 'undo' a line read.
    '''
    def __init__(self, f):
        '''
           **Arguments:**

           f
                An open file object.
        '''
        self.f = f
        self.cache = []
        self.in_comment = False

    def iter(self):
        '''Return an iterator'''
        return self

    def rewind(self, line):
        '''Undo the last line read (can be repeated)

           **Arguments:**

           line
                The line that was last read.
        '''

        self.cache.append(line)

    def next(self):
        '''Get the next line from the file (or from the undo stack).'''
        if len(self.cache) == 0:
            while True:
                line = self.f.next()
                line = line[:line.find('#')].strip()
                if self.in_comment:
                    if line == ';':
                        self.in_comment = False
                else:
                    if line == ';':
                        self.in_comment = True
                    elif len(line) > 0:
                        return line
        else:
            return self.cache.pop(-1)


def _interpret_cif_value(value_str):
    '''Parse a string from a CIF file'''
    if value_str[0] == '\'':
        assert value_str[-1] == '\''
        return value_str[1:-1]

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    try:
        return float(value_str[:value_str.find('(')])
    except ValueError:
        pass

    return value_str


def _load_cif_table(irl):
    '''Load an entire table from a CIF file

       **Arguments:**

       irl
            An instance of IterRelevantCIFLines
    '''
    # first read headings
    headings = []
    while True:
        line = irl.next()
        if line.startswith('_'):
            headings.append(line[1:])
        else:
            irl.rewind(line)
            break

    # Read in the data
    rows = []
    while True:
        try:
            line = irl.next()
        except StopIteration:
            break
        if line.startswith('_') or line.startswith('loop_'):
            irl.rewind(line)
            break
        else:
            words = shlex.split(line)
            rows.append([_interpret_cif_value(word) for word in words])

    return headings, rows



def _load_cif_low(fn_cif):
    '''Read the title and all the field arrays from the CIF file.

       **Arguments:**

       filename
            The name of the CIF file.

       **Returns:**

       title
            The title of the CIF file.

       fields
            A dictionary with all the data from the CIF file. Tables are cut
            into columns with each one having a corresponding item in the
            dictionary. The data as kept in the same units as in the original
            CIF file.
    '''
    title = None
    fields = {}
    tables = []
    with open(fn_cif) as f:
        irl = IterRelevantCIFLines(f)
        title = irl.next()
        assert title.startswith('data_')
        title = title[5:]
        while True:
            try:
                line = irl.next()
            except StopIteration:
                break

            if len(line) == 0:
                continue
            elif line.startswith('_'):
                words = shlex.split(line)
                if len(words) == 2:
                    value = _interpret_cif_value(words[1])
                    key = words[0][1:]
                    fields[key] = value
            elif line.startswith('loop_'):
                tables.append(_load_cif_table(irl))
            # Just ignore lines that are not fit
            #else:
            #    print line
            #    raise NotImplementedError

    # convert tables to extra fields
    for header, rows in tables:
        for index, key in enumerate(header):
            data = np.array([row[index] for row in rows])
            fields[key] = data

    return title, fields


def iter_equiv_pos_terms(comp):
    '''Iterate over the terms in a component from the equiv_pos_as_xyz table

       **Arguments:**

       comp
            A string with one or more terms, e.g. 'x', '-y', 'y+1/2', ...

       **Yields:** 2-tuples of the form (sign, sybol). For example, the input
       'y-1/2' will yield to 2-tuples: (+1, 'y'), (-1, '1/2')
    '''
    # remove all white space first:
    comp = comp.replace(' ','')
    while len(comp) > 0:
        sign = 1-2*(comp[0]=='-')
        if comp[0] in '+-':
            comp = comp[1:]
        pos_p = comp.find('+')
        pos_m = comp.find('-')
        if pos_p < 0: pos_p += len(comp)+1
        if pos_m < 0: pos_m += len(comp)+1
        end = min(pos_p, pos_m)
        yield sign, comp[:end]
        comp = comp[end:]


def equiv_pos_to_generator(s):
    '''Convert a equiv_pos_as_xyz string to a generator matrix

       **Arguments:**

       s
            A single line from the equiv_pos_as_xyz table

       **Returns:** a (3,4) array where the first three columns contain the
       rotation matrix and the last column is the translation vector.
    '''
    g = np.zeros((3, 4), float)
    for index, comp in enumerate(s.split(',')):
        for sign, term in iter_equiv_pos_terms(comp):
            if term in 'xyz':
                g[index,'xyz'.find(term)] = sign
            else:
                nom, denom = term.split('/')
                g[index,3] = sign*float(nom)/float(denom)

    return g


def load_cif(filename, lf):
    '''Load a CIF file

       **Arguments:**

       filename
            The name of an existing CIF file

       lf
            An instance of the LinalgFactory class. (Ignored here)

       **Returns** a dictionary with: ``title``, ``coordinates``, ``numbers``,
       ``symmetry``, ``links``, ``cell``.
    '''
    title, fields = _load_cif_low(filename)

    name = fields.get('symmetry_Int_Tables_number', 'None')

    generators = [equiv_pos_to_generator(s) for s in fields['symmetry_equiv_pos_as_xyz']]

    x = fields['atom_site_fract_x'].reshape(-1, 1)
    y = fields['atom_site_fract_y'].reshape(-1, 1)
    z = fields['atom_site_fract_z'].reshape(-1, 1)
    prim_fracs = np.hstack((x, y, z))

    prim_numbers = np.array([periodic[symbol].number for symbol in fields['atom_site_type_symbol']])

    lengths = np.array([fields['cell_length_a'], fields['cell_length_b'], fields['cell_length_c']])*angstrom
    angles = np.array([fields['cell_angle_alpha'], fields['cell_angle_beta'], fields['cell_angle_gamma']])*deg
    cell = Cell.from_parameters(lengths, angles)

    prim_labels = fields['atom_site_label']

    symmetry = Symmetry(name, generators, prim_fracs, prim_numbers, cell, prim_labels)

    coordinates, numbers, links = symmetry.generate()

    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers,
        'symmetry': symmetry,
        'links': links,
        'cell': cell,
    }
