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

from horton.system import System
from horton.cext import compute_nucnuc
from horton.meanfield.hamiltonian import Hamiltonian
from horton.meanfield.observable import OneBodyTerm, DirectTerm, ExchangeTerm
from horton.matrix import DenseOneBody, DenseTwoBody
from horton.part.mulliken import get_mulliken_operators
from horton.test.common import compare_wfns


__all__ = ['compute_mulliken_charges', 'compare_mol']


def compute_mulliken_charges(obasis, lf, numbers, wfn):
    operators = get_mulliken_operators(obasis, lf)
    populations = np.array([operator.expectation_value(wfn.dm_full) for operator in operators])
    return numbers - np.array(populations)


def compare_mol(mol1, mol2):
    assert (mol1.numbers == mol2.numbers).all()
    assert (mol1.coordinates == mol2.coordinates).all()
    # orbital basis
    if mol1.obasis is not None:
        assert (mol1.obasis.centers == mol2.obasis.centers).all()
        assert (mol1.obasis.shell_map == mol2.obasis.shell_map).all()
        assert (mol1.obasis.nprims == mol2.obasis.nprims).all()
        assert (mol1.obasis.shell_types == mol2.obasis.shell_types).all()
        assert (mol1.obasis.alphas == mol2.obasis.alphas).all()
        assert (mol1.obasis.con_coeffs == mol2.obasis.con_coeffs).all()
    else:
        assert mol2.obasis is None
    # wfn
    compare_wfns(mol1.wfn, mol2.wfn)
    # operators
    for key in 'olp', 'kin', 'na', 'er':
        if hasattr(mol1, key):
            assert hasattr(mol2, key)
            compare_operator(getattr(mol1, key), getattr(mol2, key))
        else:
            assert not hasattr(mol2, key)


def compare_operator(op1, op2):
    # TODO: move this to horton.test.common after System class is removed.
    if isinstance(op1, DenseOneBody) or isinstance(op1, DenseTwoBody):
        assert isinstance(op2, op1.__class__)
        assert op1.nbasis == op2.nbasis
        assert (op1._array == op2._array).all()
    else:
        raise NotImplementedError


def compute_hf_energy(mol):
    # TODO: this has to be updated after rewrite of Hamiltonian class without
    # System class
    sys = System(
        mol.coordinates, mol.numbers, mol.obasis, wfn=mol.wfn,
        lf=mol.lf
    )
    kin = sys.get_kinetic()
    nai = sys.get_nuclear_attraction()
    er = sys.get_electron_repulsion()
    external = {'nn': compute_nucnuc(sys.coordinates, sys.numbers)}
    terms = [
        OneBodyTerm(kin, sys.lf, sys.wfn, 'kin'),
        DirectTerm(er, sys.lf, sys.wfn),
        ExchangeTerm(er, sys.lf, sys.wfn),
        OneBodyTerm(nai, sys.lf, sys.wfn, 'ne'),
    ]
    ham = Hamiltonian(sys, terms, external)
    return ham.compute()
