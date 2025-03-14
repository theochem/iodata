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
"""Unit tests for iodata.convert."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..basis import MolecularBasis, Shell
from ..convert import (
    CCA_CONVENTIONS,
    HORTON2_CONVENTIONS,
    _convert_convention_shell,
    convert_conventions,
    convert_to_segmented,
    convert_to_unrestricted,
    iter_cart_alphabet,
)
from ..formats.cp2klog import CONVENTIONS as CP2K_CONVENTIONS
from ..orbitals import MolecularOrbitals


def test_convert_convention_shell():
    assert _convert_convention_shell("abc", "cba") == ([2, 1, 0], [1, 1, 1])
    assert _convert_convention_shell(["a", "b", "c"], ["c", "b", "a"]) == ([2, 1, 0], [1, 1, 1])
    assert _convert_convention_shell([], []) == ([], [])

    permutation, signs = _convert_convention_shell(["-a", "b", "c"], ["c", "b", "a"])
    assert permutation == [2, 1, 0]
    assert signs == [1, 1, -1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([3, 2, -1])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = _convert_convention_shell(["-a", "b", "c"], ["c", "b", "a"], True)
    assert_equal(vec2[permutation] * signs, vec1)

    permutation, signs = _convert_convention_shell(["a", "b", "c"], ["-c", "b", "a"])
    assert permutation == [2, 1, 0]
    assert signs == [-1, 1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([-3, 2, 1])
    assert_equal(vec1[permutation] * signs, vec2)

    permutation, signs = _convert_convention_shell(["a", "-b", "-c"], ["-c", "b", "a"])
    assert permutation == [2, 1, 0]
    assert signs == [1, -1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([3, -2, 1])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = _convert_convention_shell(["a", "-b", "-c"], ["-c", "b", "a"], True)
    assert_equal(vec2[permutation] * signs, vec1)

    permutation, signs = _convert_convention_shell(["fo", "ba", "sp"], ["fo", "-sp", "ba"])
    assert permutation == [0, 2, 1]
    assert signs == [1, -1, 1]
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, -3, 2])
    assert_equal(vec1[permutation] * signs, vec2)
    permutation, signs = _convert_convention_shell(["fo", "ba", "sp"], ["fo", "-sp", "ba"], True)
    assert_equal(vec2[permutation] * signs, vec1)


@pytest.fixture
def obasis() -> MolecularBasis:
    return MolecularBasis(
        [
            Shell(0, [0], ["c"], np.zeros(3), np.zeros((3, 1))),
            Shell(0, [0, 1], ["c", "c"], np.zeros(3), np.zeros((3, 2))),
            Shell(0, [0, 1], ["c", "c"], np.zeros(3), np.zeros((3, 2))),
            Shell(0, [2], ["p"], np.zeros(3), np.zeros((3, 1))),
        ],
        {
            (0, "c"): ["s"],
            (1, "c"): ["x", "z", "-y"],
            (2, "p"): ["dc0", "dc1", "-ds1", "dc2", "-ds2"],
        },
        "L2",
    )


def test_convert_convention_obasis(obasis: MolecularBasis):
    new_convention = {
        (0, "c"): ["-s"],
        (1, "c"): ["x", "y", "z"],
        (2, "p"): ["dc2", "dc1", "dc0", "ds1", "ds2"],
    }
    permutation, signs = convert_conventions(obasis, new_convention)
    assert_equal(permutation, [0, 1, 2, 4, 3, 5, 6, 8, 7, 12, 10, 9, 11, 13])
    assert_equal(signs, [-1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1])
    rng = np.random.default_rng(1)
    vec1 = rng.uniform(-1, 1, obasis.nbasis)
    vec2 = vec1[permutation] * signs
    permutation, signs = convert_conventions(obasis, new_convention, reverse=True)
    vec3 = vec2[permutation] * signs
    assert_equal(vec1, vec3)


def test_convert_convention_obasis_empty(obasis: MolecularBasis):
    with pytest.raises(KeyError):
        convert_conventions(obasis, {})


def test_convert_convention_obasis_invalid(obasis: MolecularBasis):
    new_convention = {
        (0, "c"): ["-s"],
        (1, "c"): ["a", "b", "c"],
        (2, "p"): ["dc2", "dc1", "dc0", "ds1", "ds2"],
    }
    with pytest.raises(ValueError):
        convert_conventions(obasis, new_convention)


def test_convert_convention_obasis_duplicate(obasis: MolecularBasis):
    new_convention = {
        (0, "c"): ["-s"],
        (1, "c"): ["x", "y", "y"],
        (2, "p"): ["dc2", "dc1", "dc0", "ds1", "ds2"],
    }
    with pytest.raises(ValueError):
        convert_conventions(obasis, new_convention)


def test_convert_exceptions():
    with pytest.raises(ValueError):
        _convert_convention_shell("abc", "cb")
    with pytest.raises(ValueError):
        _convert_convention_shell("abc", "cbb")
    with pytest.raises(ValueError):
        _convert_convention_shell("aba", "cba")
    with pytest.raises(ValueError):
        _convert_convention_shell(["a", "b", "c"], ["a", "b", "d"])
    with pytest.raises(ValueError):
        _convert_convention_shell(["a", "b", "c"], ["a", "b", "-d"])
    with pytest.raises(ValueError):
        _convert_convention_shell([1, 2, 3], "cb")


def test_iter_cart_alphabet():
    assert np.array(list(iter_cart_alphabet(0))).tolist() == [[0, 0, 0]]
    assert np.array(list(iter_cart_alphabet(1))).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.array(list(iter_cart_alphabet(2))).tolist() == [
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]
    for angmom in -1, -2, -3:
        with pytest.raises(ValueError):
            list(iter_cart_alphabet(angmom))


def test_conventions():
    for angmom in range(25):
        assert HORTON2_CONVENTIONS[(angmom, "c")] == CCA_CONVENTIONS[(angmom, "c")]
    assert HORTON2_CONVENTIONS[(0, "c")] == ["1"]
    assert HORTON2_CONVENTIONS[(1, "c")] == ["x", "y", "z"]
    assert HORTON2_CONVENTIONS[(2, "c")] == ["xx", "xy", "xz", "yy", "yz", "zz"]
    assert (0, "p") not in HORTON2_CONVENTIONS
    assert (0, "p") not in CCA_CONVENTIONS
    assert (1, "p") not in HORTON2_CONVENTIONS
    assert (1, "p") not in CCA_CONVENTIONS
    assert HORTON2_CONVENTIONS[(2, "p")] == ["c0", "c1", "s1", "c2", "s2"]
    assert CCA_CONVENTIONS[(2, "p")] == ["s2", "s1", "c0", "c1", "c2"]
    assert HORTON2_CONVENTIONS[(3, "p")] == ["c0", "c1", "s1", "c2", "s2", "c3", "s3"]
    assert CCA_CONVENTIONS[(3, "p")] == ["s3", "s2", "s1", "c0", "c1", "c2", "c3"]


def test_convert_to_unrestricted_generalized():
    with pytest.raises(ValueError):
        convert_to_unrestricted(MolecularOrbitals("generalized", None, None))


def test_convert_to_unrestricted_pass_through():
    mo1 = MolecularOrbitals("unrestricted", 5, 3, occs=[1, 1, 0, 0, 0, 1, 0, 0])
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is mo2


def test_convert_to_unrestricted_minimal():
    mo1 = MolecularOrbitals("restricted", 5, 5)
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is not mo2
    assert mo2.kind == "unrestricted"
    assert mo2.norba == 5
    assert mo2.norbb == 5
    assert mo2.coeffs is None
    assert mo2.occs is None
    assert mo2.coeffs is None
    assert mo2.energies is None
    assert mo2.irreps is None


def test_convert_to_unrestricted_aminusb():
    mo1 = MolecularOrbitals(
        "restricted",
        5,
        5,
        occs=np.array([2.0, 0.8, 0.2, 0.0, 0.0]),
        occs_aminusb=np.array([0.0, 0.2, 0.2, 0.0, 0.0]),
    )
    assert_allclose(mo1.spinpol, 0.4)
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is not mo2
    assert mo2.kind == "unrestricted"
    assert_allclose(mo2.occsa, [1.0, 0.5, 0.2, 0.0, 0.0])
    assert_allclose(mo2.occsb, [1.0, 0.3, 0.0, 0.0, 0.0])
    assert_allclose(mo2.spinpol, 0.4)


def test_convert_to_unrestricted_occ_integer():
    mo1 = MolecularOrbitals(
        "restricted",
        5,
        5,
        occs=np.array([2.0, 1.0, 1.0, 0.0, 0.0]),
    )
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is not mo2
    assert mo2.kind == "unrestricted"
    assert_allclose(mo2.occsa, [1.0, 1.0, 1.0, 0.0, 0.0])
    assert_allclose(mo2.occsb, [1.0, 0.0, 0.0, 0.0, 0.0])


def test_convert_to_unrestricted_occ_float():
    mo1 = MolecularOrbitals(
        "restricted",
        5,
        5,
        occs=np.array([2.0, 1.6, 1.0, 0.0, 0.0]),
    )
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is not mo2
    assert mo2.kind == "unrestricted"
    assert_allclose(mo2.occsa, [1.0, 0.8, 0.5, 0.0, 0.0])
    assert_allclose(mo2.occsb, mo2.occsa)


def test_convert_to_unrestricted_full():
    rng = np.random.default_rng(42)
    mo1 = MolecularOrbitals(
        "restricted",
        5,
        5,
        occs=rng.uniform(0, 1, 5),
        coeffs=rng.uniform(0, 1, (8, 5)),
        energies=rng.uniform(-1, 0, 5),
        irreps=["A"] * 5,
    )
    mo2 = convert_to_unrestricted(mo1)
    assert mo1 is not mo2
    assert mo2.kind == "unrestricted"
    assert_allclose(mo2.occsa, mo1.occsa)
    assert_allclose(mo2.occsb, mo1.occsb)
    assert_allclose(mo2.coeffsa, mo1.coeffsa)
    assert_allclose(mo2.coeffsb, mo1.coeffsb)
    assert_allclose(mo2.energiesa, mo1.energiesa)
    assert_allclose(mo2.energiesb, mo1.energiesb)
    assert_equal(mo2.irrepsa, mo1.irrepsa)
    assert_equal(mo2.irrepsb, mo1.irrepsb)


def test_convert_to_segmented():
    rng = np.random.default_rng(1)
    obasis0 = MolecularBasis(
        [
            Shell(0, [0, 1], ["c", "c"], rng.uniform(0, 1, 5), rng.uniform(-1, 1, (5, 2))),
            Shell(1, [2, 3], ["p", "p"], rng.uniform(0, 1, 7), rng.uniform(-1, 1, (7, 2))),
        ],
        CP2K_CONVENTIONS,
        "L2",
    )
    assert obasis0.nbasis == 16
    obasis1 = convert_to_segmented(obasis0)
    assert len(obasis1.shells) == 4
    assert obasis1.nbasis == 16
    # shell 0
    shell0 = obasis1.shells[0]
    assert shell0.icenter == 0
    assert_equal(shell0.angmoms, [0])
    assert_equal(shell0.kinds, ["c"])
    assert_equal(shell0.exponents, obasis0.shells[0].exponents)
    assert_equal(shell0.coeffs, obasis0.shells[0].coeffs[:, :1])
    # shell 1
    shell1 = obasis1.shells[1]
    assert shell1.icenter == 0
    assert_equal(shell1.angmoms, [1])
    assert_equal(shell1.kinds, ["c"])
    assert_equal(shell1.exponents, obasis0.shells[0].exponents)
    assert_equal(shell1.coeffs, obasis0.shells[0].coeffs[:, 1:])
    # shell 2
    shell2 = obasis1.shells[2]
    assert shell2.icenter == 1
    assert_equal(shell2.angmoms, [2])
    assert_equal(shell2.kinds, ["p"])
    assert_equal(shell2.exponents, obasis0.shells[1].exponents)
    assert_equal(shell2.coeffs, obasis0.shells[1].coeffs[:, :1])
    # shell 0
    shell3 = obasis1.shells[3]
    assert shell3.icenter == 1
    assert_equal(shell3.angmoms, [3])
    assert_equal(shell3.kinds, ["p"])
    assert_equal(shell3.exponents, obasis0.shells[1].exponents)
    assert_equal(shell3.coeffs, obasis0.shells[1].coeffs[:, 1:])


def test_convert_to_segmented_sp():
    rng = np.random.default_rng(1)
    obasis0 = MolecularBasis(
        [
            Shell(0, [0, 1], ["c", "c"], rng.uniform(0, 1, 5), rng.uniform(-1, 1, (5, 2))),
            Shell(1, [2, 3], ["p", "p"], rng.uniform(0, 1, 7), rng.uniform(-1, 1, (7, 2))),
        ],
        HORTON2_CONVENTIONS,
        "L2",
    )
    obasis1 = convert_to_segmented(obasis0, keep_sp=True)
    assert len(obasis1.shells) == 3
    assert obasis1.nbasis == 16
    # shell 0
    shell0 = obasis1.shells[0]
    assert shell0.icenter == 0
    assert_equal(shell0.angmoms, [0, 1])
    assert_equal(shell0.kinds, ["c", "c"])
    assert_equal(shell0.exponents, obasis0.shells[0].exponents)
    assert_equal(shell0.coeffs, obasis0.shells[0].coeffs)
    # shell 1
    shell2 = obasis1.shells[1]
    assert shell2.icenter == 1
    assert_equal(shell2.angmoms, [2])
    assert_equal(shell2.kinds, ["p"])
    assert_equal(shell2.exponents, obasis0.shells[1].exponents)
    assert_equal(shell2.coeffs, obasis0.shells[1].coeffs[:, :1])
    # shell 3
    shell3 = obasis1.shells[2]
    assert shell3.icenter == 1
    assert_equal(shell3.angmoms, [3])
    assert_equal(shell3.kinds, ["p"])
    assert_equal(shell3.exponents, obasis0.shells[1].exponents)
    assert_equal(shell3.coeffs, obasis0.shells[1].coeffs[:, 1:])


def test_convert_to_segmented_empty():
    obasis0 = MolecularBasis([], HORTON2_CONVENTIONS, "L2")
    obasis1 = convert_to_segmented(obasis0, keep_sp=False)
    assert len(obasis1.shells) == 0
    obasis2 = convert_to_segmented(obasis0, keep_sp=True)
    assert len(obasis2.shells) == 0
