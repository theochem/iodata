# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Unit tests for iodata.obasis."""


import attr
import numpy as np
from numpy.testing import assert_equal
import pytest

from ..basis import (angmom_sti, angmom_its, Shell, MolecularBasis,
                     convert_convention_shell, convert_conventions,
                     iter_cart_alphabet, HORTON2_CONVENTIONS, CCA_CONVENTIONS)
from ..formats.cp2klog import CONVENTIONS as CP2K_CONVENTIONS


def test_angmom_sti():
    assert angmom_sti('s') == 0
    assert angmom_sti('p') == 1
    assert angmom_sti('f') == 3
    assert angmom_sti(['s']) == [0]
    assert angmom_sti(['s', 's']) == [0, 0]
    assert angmom_sti(['s', 's', 's']) == [0, 0, 0]
    assert angmom_sti(['p']) == [1]
    assert angmom_sti(['s', 'p']) == [0, 1]
    assert angmom_sti(['s', 'p', 'p']) == [0, 1, 1]
    assert angmom_sti(['s', 'p', 'p', 'd', 'd', 's', 'f', 'i']) == \
        [0, 1, 1, 2, 2, 0, 3, 6]
    assert angmom_sti(['e', 't', 'k']) == [24, 14, 7]


def test_angmom_sti_uppercase():
    assert angmom_sti('S') == 0
    assert angmom_sti('D') == 2
    assert angmom_sti('g') == 4
    assert angmom_sti(['P']) == [1]
    assert angmom_sti(['F', 'f']) == [3, 3]
    assert angmom_sti(['n', 'N', 'N']) == [10, 10, 10]
    assert angmom_sti(['D', 'O']) == [2, 11]
    assert angmom_sti(['S', 'p', 'P', 'D', 's', 'I']) == [0, 1, 1, 2, 0, 6]
    assert angmom_sti(['E', 'T', 'k']) == [24, 14, 7]


def test_angmom_its():
    assert angmom_its(0) == 's'
    assert angmom_its(1) == 'p'
    assert angmom_its(2) == 'd'
    assert angmom_its(3) == 'f'
    assert angmom_its(24) == 'e'
    assert angmom_its([0, 1, 3]) == ['s', 'p', 'f']
    with pytest.raises(ValueError):
        angmom_its(-1)
    with pytest.raises(ValueError):
        angmom_its([-4])
    with pytest.raises(ValueError):
        angmom_its([1, -5])
    with pytest.raises(ValueError):
        angmom_its([0, 5, -2, 3, 3, 1])


def test_shell_info_propertes():
    shells = [
        Shell(0, [0], ['c'], np.zeros(6), np.zeros((6, 1))),
        Shell(0, [0, 1], ['c', 'c'], np.zeros(3), np.zeros((3, 2))),
        Shell(0, [0, 1], ['c', 'c'], np.zeros(1), np.zeros((1, 2))),
        Shell(0, [2], ['p'], np.zeros(2), np.zeros((2, 1))),
        Shell(0, [2, 3, 4], ['c', 'p', 'p'], np.zeros(1), np.zeros((1, 3)))]

    assert shells[0].nbasis == 1
    assert shells[1].nbasis == 4
    assert shells[2].nbasis == 4
    assert shells[3].nbasis == 5
    assert shells[4].nbasis == 6 + 7 + 9
    assert shells[0].nprim == 6
    assert shells[1].nprim == 3
    assert shells[2].nprim == 1
    assert shells[3].nprim == 2
    assert shells[4].nprim == 1
    assert shells[0].ncon == 1
    assert shells[1].ncon == 2
    assert shells[2].ncon == 2
    assert shells[3].ncon == 1
    assert shells[4].ncon == 3
    obasis = MolecularBasis(
        shells,
        {(0, 'c'): ['s'],
         (1, 'c'): ['x', 'z', '-y'],
         (2, 'p'): ['dc0', 'dc1', '-ds1', 'dc2', '-ds2']},
        'L2')
    assert obasis.nbasis == 1 + 4 + 4 + 5 + 6 + 7 + 9


def test_shell_validators():
    # The following line constructs a Shell instance with valid arguments.
    # It should not raise a TypeError.
    shell = Shell(0, [0, 0], ['c', 'c'], np.zeros(6), np.zeros((6, 2)))
    # Rerun the validators as a double check.
    attr.validate(shell)
    # Tests with invalid constructor arguments.
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ['c', 'c'], np.zeros(6), np.zeros((6, 2, 2)))
    with pytest.raises(TypeError):
        Shell(0, [0], ['c'], np.zeros(6), np.zeros(6,))
    with pytest.raises(TypeError):
        Shell(0, [0], ['c'], np.zeros((6, 2)), np.zeros((6, 1)))
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ['c', 'c'], np.zeros((6, 2)), np.zeros((6, 1)))
    with pytest.raises(TypeError):
        Shell(0, [0], ['c', 'c'], np.zeros(6), np.zeros((6, 2)))
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ['c'], np.zeros(6), np.zeros((6, 2)))


def test_shell_exceptions():
    Shell(0, [0, 0, 0], ['e', 'e', 'e'], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [0, 0, 0], ['e', 'e', 'e'], np.zeros(6), np.zeros((6, 3))).nbasis
    Shell(0, [0, 0, 0], ['p', 'p', 'p'], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [0, 0, 0], ['p', 'p', 'p'], np.zeros(6), np.zeros((6, 3))).nbasis
    Shell(0, [1, 1, 1], ['p', 'p', 'p'], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [1, 1, 1], ['p', 'p', 'p'], np.zeros(6), np.zeros((6, 3))).nbasis


def test_nbasis1():
    obasis = MolecularBasis([
        Shell(0, [0], ['c'], np.zeros(16), np.zeros((16, 1))),
        Shell(0, [1], ['c'], np.zeros(16), np.zeros((16, 1))),
        Shell(0, [2], ['p'], np.zeros(16), np.zeros((16, 1))),
    ], CP2K_CONVENTIONS, 'L2')
    assert obasis.nbasis == 9


def test_get_segmented():
    obasis0 = MolecularBasis([
        Shell(0, [0, 1], ['c', 'c'], np.random.uniform(0, 1, 5),
              np.random.uniform(-1, 1, (5, 2))),
        Shell(1, [2, 3], ['p', 'p'], np.random.uniform(0, 1, 7),
              np.random.uniform(-1, 1, (7, 2))),
    ], CP2K_CONVENTIONS, 'L2')
    assert obasis0.nbasis == 16
    obasis1 = obasis0.get_segmented()
    assert len(obasis1.shells) == 4
    assert obasis1.nbasis == 16
    # shell 0
    shell0 = obasis1.shells[0]
    assert shell0.icenter == 0
    assert_equal(shell0.angmoms, [0])
    assert shell0.kinds == ['c']
    assert_equal(shell0.exponents, obasis0.shells[0].exponents)
    assert_equal(shell0.coeffs, obasis0.shells[0].coeffs[:, :1])
    # shell 1
    shell1 = obasis1.shells[1]
    assert shell1.icenter == 0
    assert_equal(shell1.angmoms, [1])
    assert shell1.kinds == ['c']
    assert_equal(shell1.exponents, obasis0.shells[0].exponents)
    assert_equal(shell1.coeffs, obasis0.shells[0].coeffs[:, 1:])
    # shell 2
    shell2 = obasis1.shells[2]
    assert shell2.icenter == 1
    assert_equal(shell2.angmoms, [2])
    assert shell2.kinds == ['p']
    assert_equal(shell2.exponents, obasis0.shells[1].exponents)
    assert_equal(shell2.coeffs, obasis0.shells[1].coeffs[:, :1])
    # shell 0
    shell3 = obasis1.shells[3]
    assert shell3.icenter == 1
    assert_equal(shell3.angmoms, [3])
    assert shell3.kinds == ['p']
    assert_equal(shell3.exponents, obasis0.shells[1].exponents)
    assert_equal(shell3.coeffs, obasis0.shells[1].coeffs[:, 1:])


def test_convert_convention_shell():
    assert convert_convention_shell('abc', 'cba') == ([2, 1, 0], [1, 1, 1])
    assert convert_convention_shell(['a', 'b', 'c'], ['c', 'b', 'a']) == ([2, 1, 0], [1, 1, 1])

    permutation, signs = convert_convention_shell(['-a', 'b', 'c'], ['c', 'b', 'a'])
    assert permutation == [2, 1, 0]
    assert signs == [1, 1, -1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([3, 2, -1])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = convert_convention_shell(['-a', 'b', 'c'], ['c', 'b', 'a'], True)
    assert_equal(vec2[permutation] * signs, vec1)

    permutation, signs = convert_convention_shell(['a', 'b', 'c'], ['-c', 'b', 'a'])
    assert permutation == [2, 1, 0]
    assert signs == [-1, 1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([-3, 2, 1])
    assert_equal(vec1[permutation] * signs, vec2)

    permutation, signs = convert_convention_shell(['a', '-b', '-c'], ['-c', 'b', 'a'])
    assert permutation == [2, 1, 0]
    assert signs == [1, -1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([3, -2, 1])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = convert_convention_shell(['a', '-b', '-c'], ['-c', 'b', 'a'], True)
    assert_equal(vec2[permutation] * signs, vec1)

    permutation, signs = convert_convention_shell(['fo', 'ba', 'sp'], ['fo', '-sp', 'ba'])
    assert permutation == [0, 2, 1]
    assert signs == [1, -1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, -3, 2])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = convert_convention_shell(['fo', 'ba', 'sp'], ['fo', '-sp', 'ba'], True)
    assert_equal(vec2[permutation] * signs, vec1)


def test_convert_convention_obasis():
    obasis = MolecularBasis(
        [Shell(0, [0], ['c'], np.zeros(3), np.zeros((3, 1))),
         Shell(0, [0, 1], ['c', 'c'], np.zeros(3), np.zeros((3, 2))),
         Shell(0, [0, 1], ['c', 'c'], np.zeros(3), np.zeros((3, 2))),
         Shell(0, [2], ['p'], np.zeros(3), np.zeros((3, 1)))],
        {(0, 'c'): ['s'],
         (1, 'c'): ['x', 'z', '-y'],
         (2, 'p'): ['dc0', 'dc1', '-ds1', 'dc2', '-ds2']},
        'L2')
    new_convention = {(0, 'c'): ['-s'],
                      (1, 'c'): ['x', 'y', 'z'],
                      (2, 'p'): ['dc2', 'dc1', 'dc0', 'ds1', 'ds2']}
    permutation, signs = convert_conventions(obasis, new_convention)
    assert_equal(permutation, [0, 1, 2, 4, 3, 5, 6, 8, 7, 12, 10, 9, 11, 13])
    assert_equal(signs, [-1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1])
    vec1 = np.random.uniform(-1, 1, obasis.nbasis)
    vec2 = vec1[permutation] * signs
    permutation, signs = convert_conventions(obasis, new_convention, reverse=True)
    vec3 = vec2[permutation] * signs
    assert_equal(vec1, vec3)


def test_convert_exceptions():
    with pytest.raises(TypeError):
        convert_convention_shell('abc', 'cb')
    with pytest.raises(TypeError):
        convert_convention_shell('abc', 'cbb')
    with pytest.raises(TypeError):
        convert_convention_shell('aba', 'cba')
    with pytest.raises(TypeError):
        convert_convention_shell(['a', 'b', 'c'], ['a', 'b', 'd'])
    with pytest.raises(TypeError):
        convert_convention_shell(['a', 'b', 'c'], ['a', 'b', '-d'])


def test_iter_cart_alphabet():
    assert np.array(list(iter_cart_alphabet(0))).tolist() == [[0, 0, 0]]
    assert np.array(list(iter_cart_alphabet(1))).tolist() == [
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.array(list(iter_cart_alphabet(2))).tolist() == [
        [2, 0, 0], [1, 1, 0], [1, 0, 1],
        [0, 2, 0], [0, 1, 1], [0, 0, 2]]


def test_conventions():
    for angmom in range(25):
        assert HORTON2_CONVENTIONS[(angmom, 'c')] == CCA_CONVENTIONS[(angmom, 'c')]
    assert HORTON2_CONVENTIONS[(0, 'c')] == ['1']
    assert HORTON2_CONVENTIONS[(1, 'c')] == ['x', 'y', 'z']
    assert HORTON2_CONVENTIONS[(2, 'c')] == ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    assert (0, 'p') not in HORTON2_CONVENTIONS
    assert (0, 'p') not in CCA_CONVENTIONS
    assert (1, 'p') not in HORTON2_CONVENTIONS
    assert (1, 'p') not in CCA_CONVENTIONS
    assert HORTON2_CONVENTIONS[(2, 'p')] == ['c0', 'c1', 's1', 'c2', 's2']
    assert CCA_CONVENTIONS[(2, 'p')] == ['s2', 's1', 'c0', 'c1', 'c2']
    assert HORTON2_CONVENTIONS[(3, 'p')] == ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
    assert CCA_CONVENTIONS[(3, 'p')] == ['s3', 's2', 's1', 'c0', 'c1', 'c2', 'c3']
