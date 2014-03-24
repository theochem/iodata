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
from horton.meanfield.hamiltonian import Hamiltonian
from horton.meanfield.builtin import HartreeFockExchange
from horton.matrix import DenseOneBody, DenseTwoBody
from horton.part.mulliken import get_mulliken_operators
from horton.test.common import compare_wfns

__all__ = ['compute_mulliken_charges', 'compare_data']


def compute_mulliken_charges(obasis, lf, numbers, wfn):
    operators = get_mulliken_operators(obasis, lf)
    populations = np.array([operator.expectation_value(wfn.dm_full) for operator in operators])
    return numbers - np.array(populations)


def compare_data(data1, data2):
    assert (data1['numbers'] == data2['numbers']).all()
    assert (data1['coordinates'] == data2['coordinates']).all()
    # orbital basis
    if data1['obasis'] is not None:
        assert (data1['obasis'].centers == data2['obasis'].centers).all()
        assert (data1['obasis'].shell_map == data2['obasis'].shell_map).all()
        assert (data1['obasis'].nprims == data2['obasis'].nprims).all()
        assert (data1['obasis'].shell_types == data2['obasis'].shell_types).all()
        assert (data1['obasis'].alphas == data2['obasis'].alphas).all()
        assert (data1['obasis'].con_coeffs == data2['obasis'].con_coeffs).all()
    else:
        assert data2['obasis'] is None
    # wfn
    compare_wfns(data1['wfn'], data2['wfn'])
    # operators
    for key in 'olp', 'kin', 'na', 'er':
        if key in data1:
            assert key in data2
            compare_operator(data1[key], data2[key])
        else:
            assert key not in data2


def compare_operator(op1, op2):
    # TODO: move this to horton.test.common after System class is removed.
    if isinstance(op1, DenseOneBody) or isinstance(op1, DenseTwoBody):
        assert isinstance(op2, op1.__class__)
        assert op1.nbasis == op2.nbasis
        assert (op1._array == op2._array).all()
    else:
        raise NotImplementedError


def compute_hf_energy(data):
    # TODO: this has to be updated after rewrite of Hamiltonian class without
    # System class
    sys = System(
        data['coordinates'], data['numbers'], data['obasis'], wfn=data['wfn'],
        lf=data['lf']
    )
    ham = Hamiltonian(sys, [HartreeFockExchange()])
    return ham.compute()
