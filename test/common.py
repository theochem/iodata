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
#pylint: skip-file


import numpy as np

from horton.cext import compute_nucnuc
from horton.meanfield.hamiltonian import Hamiltonian
from horton.meanfield.observable import OneBodyTerm, DirectTerm, ExchangeTerm
from horton.matrix import DenseOneBody, DenseTwoBody
from horton.part.mulliken import get_mulliken_operators
from horton.test.common import compare_wfns


__all__ = ['compute_mulliken_charges']


def compute_mulliken_charges(obasis, lf, numbers, wfn):
    operators = get_mulliken_operators(obasis, lf)
    populations = np.array([operator.expectation_value(wfn.dm_full) for operator in operators])
    return numbers - np.array(populations)


def compute_hf_energy(mol):
    olp = mol.obasis.compute_overlap(mol.lf)
    kin = mol.obasis.compute_kinetic(mol.lf)
    nai = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, mol.lf)
    er = mol.obasis.compute_electron_repulsion(mol.lf)
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        OneBodyTerm(kin, mol.wfn, 'kin'),
        DirectTerm(er, mol.wfn, 'hartree'),
        ExchangeTerm(er, mol.wfn, 'x_hf'),
        OneBodyTerm(nai, mol.wfn, 'ne'),
    ]
    ham = Hamiltonian(terms, external)
    return ham.compute()
