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


import numpy as np

from horton.cext import compute_nucnuc
from horton.meanfield.hamiltonian import REffHam, \
    UEffHam
from horton.meanfield.observable import RTwoIndexTerm, \
    RDirectTerm, RExchangeTerm, UTwoIndexTerm, \
    UDirectTerm, UExchangeTerm
from horton.part.mulliken import get_mulliken_operators


__all__ = ['compute_mulliken_charges']


def compute_mulliken_charges(obasis, lf, numbers, dm):
    operators = get_mulliken_operators(obasis, lf)
    populations = np.array([operator.contract_two('ab,ab', dm) for operator in operators])
    return numbers - np.array(populations)


def compute_hf_energy(mol):
    olp = mol.obasis.compute_overlap(mol.lf)
    kin = mol.obasis.compute_kinetic(mol.lf)
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, mol.lf)
    er = mol.obasis.compute_electron_repulsion(mol.lf)
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    if hasattr(mol, 'exp_beta'):
        # assuming unrestricted
        terms = [
            UTwoIndexTerm(kin, 'kin'),
            UDirectTerm(er, 'hartree'),
            UExchangeTerm(er, 'x_hf'),
            UTwoIndexTerm(na, 'ne'),
        ]
        ham = UEffHam(terms, external)
        dm_alpha = mol.exp_alpha.to_dm()
        dm_beta = mol.exp_beta.to_dm()
        ham.reset(dm_alpha, dm_beta)
    else:
        # assuming restricted
        terms = [
            RTwoIndexTerm(kin, 'kin'),
            RDirectTerm(er, 'hartree'),
            RExchangeTerm(er, 'x_hf'),
            RTwoIndexTerm(na, 'ne'),
        ]
        ham = REffHam(terms, external)
        dm_alpha = mol.exp_alpha.to_dm()
        ham.reset(dm_alpha)
    return ham.compute_energy()
