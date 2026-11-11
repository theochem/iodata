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

import pickle

import attrs
import numpy as np
import pytest

from ..basis import (
    BasisFunction,
    FrozenConventions,
    MolecularBasis,
    Shell,
    angmom_its,
    angmom_sti,
)
from ..convert import HORTON2_CONVENTIONS
from ..formats import cp2klog, fchk, molden, mwfn, wfn
from ..formats.cp2klog import CONVENTIONS as CP2K_CONVENTIONS


def _two_center_basis():
    """Build a basis with two centers, a generalized SP contraction and a pure shell."""
    return MolecularBasis(
        [
            Shell(0, [0], ["c"], np.zeros(3), np.zeros((3, 1))),
            Shell(0, [1], ["c"], np.zeros(3), np.zeros((3, 1))),
            Shell(1, [0, 1], ["c", "c"], np.zeros(3), np.zeros((3, 2))),
            Shell(1, [2], ["p"], np.zeros(3), np.zeros((3, 1))),
        ],
        CP2K_CONVENTIONS,
        "L2",
    )


def test_get_basis_functions_matches_nbasis():
    obasis = _two_center_basis()
    functions = obasis.get_basis_functions()
    assert obasis.nbasis == 1 + 3 + 4 + 5
    assert len(functions) == obasis.nbasis
    assert [function.index for function in functions] == list(range(obasis.nbasis))


def test_get_basis_functions_order():
    functions = _two_center_basis().get_basis_functions()
    assert [function.label for function in functions] == [
        "1",
        "x",
        "y",
        "z",
        "1",
        "x",
        "y",
        "z",
        "c0",
        "c1",
        "s1",
        "c2",
        "s2",
    ]
    assert [function.icenter for function in functions] == [0] * 4 + [1] * 9
    assert [function.angmom for function in functions] == [0] + [1] * 3 + [0] + [1] * 3 + [2] * 5
    assert [function.kind for function in functions] == ["c"] * 8 + ["p"] * 5
    # the SP shell is one shell holding two generalized contractions
    assert [function.icon for function in functions[4:8]] == [0, 1, 1, 1]
    assert [function.ishell for function in functions[4:8]] == [2, 2, 2, 2]


def test_basis_function_is_an_immutable_attrs_class():
    function = _two_center_basis().get_basis_functions()[8]
    assert function == BasisFunction(8, 3, 0, 1, 2, "p", "c0", 1)
    assert attrs.has(BasisFunction)
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        function.label = "other"
    # The values have builtin types, not NumPy types.
    for name in "index", "ishell", "icon", "icenter", "angmom", "sign":
        assert type(getattr(function, name)) is int
    assert type(function.kind) is str
    assert type(function.label) is str


def test_sign_convention_is_tracked():
    # ORCA's pure f convention, from the MolecularBasis docstring, flips two signs
    obasis = MolecularBasis(
        [Shell(0, [3], ["p"], np.zeros(3), np.zeros((3, 1)))],
        {(3, "p"): ("c0", "c1", "s1", "c2", "s2", "-c3", "-s3")},
        "L2",
    )
    functions = obasis.get_basis_functions()
    # the minus is stripped from the label and reported through sign instead
    assert [function.label for function in functions] == ["c0", "c1", "s1", "c2", "s2", "c3", "s3"]
    assert [function.sign for function in functions] == [1, 1, 1, 1, 1, -1, -1]
    # so a sign-flipped function is found by its plain label
    index = obasis.index_basis_functions()
    assert index[0, 3, "c3"] == [5]
    assert (0, 3, "-c3") not in index
    assert functions[5].sign == -1


def test_index_basis_functions_default_keys():
    obasis = _two_center_basis()
    index = obasis.index_basis_functions()
    # The default keys are the center, the angular momentum and the label.
    assert index[0, 0, "1"] == [0]
    assert index[0, 1, "x"] == [1]
    assert index[1, 0, "1"] == [4]
    assert index[1, 1, "x"] == [5]
    assert index[1, 2, "s2"] == [12]
    # Every function is in exactly one group, with the functions in the order of the basis.
    assert sorted(i for indices in index.values() for i in indices) == list(range(obasis.nbasis))
    for indices in index.values():
        assert indices == sorted(indices)


def test_index_basis_functions_round_trip():
    obasis = _two_center_basis()
    functions = obasis.get_basis_functions()
    index = obasis.index_basis_functions()
    seen = {}
    for function in functions:
        key = (function.icenter, function.angmom, function.label)
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        assert index[key][occurrence] == function.index


def test_index_basis_functions_occurrence():
    # Two shells with the same angular momentum on one center give a label twice.
    obasis = MolecularBasis(
        [
            Shell(0, [1], ["c"], np.zeros(3), np.zeros((3, 1))),
            Shell(0, [1], ["c"], np.zeros(2), np.zeros((2, 1))),
            Shell(1, [1], ["c"], np.zeros(2), np.zeros((2, 1))),
        ],
        HORTON2_CONVENTIONS,
        "L2",
    )
    index = obasis.index_basis_functions()
    assert index[0, 1, "x"] == [0, 3]
    assert index[0, 1, "z"] == [2, 5]
    assert index[1, 1, "x"] == [6]


@pytest.mark.parametrize(
    ("keys", "key", "expected"),
    [
        # The angular momentum is not part of the key, so the labels of different angular momenta
        # at the same center would be grouped (there are none here).
        (("icenter", "label"), (1, "x"), [5]),
        # All the functions of a center and a kind
        (("icenter", "kind"), (1, "p"), [8, 9, 10, 11, 12]),
        # No center, so the functions of the centers are grouped.
        (("label",), ("x",), [1, 5]),
        (("angmom", "label"), (1, "y"), [2, 6]),
        # All the attributes
        (("ishell", "icon"), (2, 1), [5, 6, 7]),
        # No keys, one group with all the functions
        ((), (), list(range(13))),
    ],
)
def test_index_basis_functions_custom_keys(keys, key, expected):
    index = _two_center_basis().index_basis_functions(keys)
    assert index[key] == expected


def test_index_basis_functions_invalid_keys():
    obasis = _two_center_basis()
    with pytest.raises(ValueError, match="icentre"):
        obasis.index_basis_functions(("icentre", "label"))
    with pytest.raises(ValueError, match="BasisFunction"):
        obasis.index_basis_functions(("label", "nonsense"))


def test_basis_functions_empty():
    obasis = MolecularBasis([], HORTON2_CONVENTIONS, "L2")
    assert obasis.get_basis_functions() == []
    assert obasis.index_basis_functions() == {}


def test_angmom_sti():
    assert angmom_sti("s") == 0
    assert angmom_sti("p") == 1
    assert angmom_sti("f") == 3
    assert angmom_sti(["s"]) == [0]
    assert angmom_sti(["s", "s"]) == [0, 0]
    assert angmom_sti(["s", "s", "s"]) == [0, 0, 0]
    assert angmom_sti(["p"]) == [1]
    assert angmom_sti(["s", "p"]) == [0, 1]
    assert angmom_sti(["s", "p", "p"]) == [0, 1, 1]
    assert angmom_sti(["s", "p", "p", "d", "d", "s", "f", "i"]) == [0, 1, 1, 2, 2, 0, 3, 6]
    assert angmom_sti(["e", "t", "k"]) == [24, 14, 7]


def test_angmom_sti_uppercase():
    assert angmom_sti("S") == 0
    assert angmom_sti("D") == 2
    assert angmom_sti("g") == 4
    assert angmom_sti(["P"]) == [1]
    assert angmom_sti(["F", "f"]) == [3, 3]
    assert angmom_sti(["n", "N", "N"]) == [10, 10, 10]
    assert angmom_sti(["D", "O"]) == [2, 11]
    assert angmom_sti(["S", "p", "P", "D", "s", "I"]) == [0, 1, 1, 2, 0, 6]
    assert angmom_sti(["E", "T", "k"]) == [24, 14, 7]


def test_angmom_its():
    assert angmom_its(0) == "s"
    assert angmom_its(1) == "p"
    assert angmom_its(2) == "d"
    assert angmom_its(3) == "f"
    assert angmom_its(24) == "e"
    assert angmom_its([0, 1, 3]) == ["s", "p", "f"]
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
        Shell(0, [0], ["c"], np.zeros(6), np.zeros((6, 1))),
        Shell(0, [0, 1], ["c", "c"], np.zeros(3), np.zeros((3, 2))),
        Shell(0, [0, 1], ["c", "c"], np.zeros(1), np.zeros((1, 2))),
        Shell(0, [2], ["p"], np.zeros(2), np.zeros((2, 1))),
        Shell(0, [2, 3, 4], ["c", "p", "p"], np.zeros(1), np.zeros((1, 3))),
    ]

    assert shells[0].nbasis == 1
    assert shells[1].nbasis == 4
    assert shells[2].nbasis == 4
    assert shells[3].nbasis == 5
    assert shells[4].nbasis == 6 + 7 + 9
    assert shells[0].nexp == 6
    assert shells[1].nexp == 3
    assert shells[2].nexp == 1
    assert shells[3].nexp == 2
    assert shells[4].nexp == 1
    assert shells[0].ncon == 1
    assert shells[1].ncon == 2
    assert shells[2].ncon == 2
    assert shells[3].ncon == 1
    assert shells[4].ncon == 3
    obasis = MolecularBasis(
        shells,
        {
            (0, "c"): ("s",),
            (1, "c"): ("x", "z", "-y"),
            (2, "p"): ("dc0", "dc1", "-ds1", "dc2", "-ds2"),
        },
        "L2",
    )
    assert obasis.nbasis == 1 + 4 + 4 + 5 + 6 + 7 + 9


def test_shell_validators():
    # The following line constructs a Shell instance with valid arguments.
    # It should not raise a TypeError.
    shell = Shell(0, [0, 0], ["c", "c"], np.zeros(6), np.zeros((6, 2)))
    # Rerun the validators as a double check.
    attrs.validate(shell)
    # Tests with invalid constructor arguments.
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ["c", "c"], np.zeros(6), np.zeros((6, 2, 2)))
    with pytest.raises(TypeError):
        Shell(
            0,
            [0],
            ["c"],
            np.zeros(6),
            np.zeros(
                6,
            ),
        )
    with pytest.raises(TypeError):
        Shell(0, [0], ["c"], np.zeros((6, 2)), np.zeros((6, 1)))
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ["c", "c"], np.zeros((6, 2)), np.zeros((6, 1)))
    with pytest.raises(TypeError):
        Shell(0, [0], ["c", "c"], np.zeros(6), np.zeros((6, 2)))
    with pytest.raises(TypeError):
        Shell(0, [0, 0], ["c"], np.zeros(6), np.zeros((6, 2)))


def test_shell_exceptions():
    Shell(0, [0, 0, 0], ["e", "e", "e"], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [0, 0, 0], ["e", "e", "e"], np.zeros(6), np.zeros((6, 3))).nbasis
    Shell(0, [0, 0, 0], ["p", "p", "p"], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [0, 0, 0], ["p", "p", "p"], np.zeros(6), np.zeros((6, 3))).nbasis
    Shell(0, [1, 1, 1], ["p", "p", "p"], np.zeros(6), np.zeros((6, 3)))
    with pytest.raises(TypeError):
        _ = Shell(0, [1, 1, 1], ["p", "p", "p"], np.zeros(6), np.zeros((6, 3))).nbasis


def test_nbasis1():
    obasis = MolecularBasis(
        [
            Shell(0, [0], ["c"], np.zeros(16), np.zeros((16, 1))),
            Shell(0, [1], ["c"], np.zeros(16), np.zeros((16, 1))),
            Shell(0, [2], ["p"], np.zeros(16), np.zeros((16, 1))),
        ],
        CP2K_CONVENTIONS,
        "L2",
    )
    assert obasis.nbasis == 9


def test_frozen_conventions_copy():
    conventions = {(0, "c"): ["1"], (1, "c"): ["x", "y", "z"]}
    frozen = FrozenConventions(conventions)
    conventions[(1, "c")].reverse()
    conventions[(2, "p")] = ["c0", "c1", "s1", "c2", "s2"]
    assert frozen[(1, "c")] == ("x", "y", "z")
    assert (2, "p") not in frozen


def test_frozen_conventions_immutable():
    frozen = FrozenConventions({(1, "c"): ("x", "y", "z")})
    with pytest.raises(TypeError):
        frozen[(0, "c")] = ("1",)
    with pytest.raises(TypeError):
        del frozen[(1, "c")]
    with pytest.raises(TypeError):
        frozen[(1, "c")][0] = "z"


def test_frozen_conventions_mapping():
    frozen = FrozenConventions({(0, "c"): ("1",), (1, "c"): ("x", "y", "z")})
    assert len(frozen) == 2
    assert list(frozen) == [(0, "c"), (1, "c")]
    assert (1, "c") in frozen
    assert (2, "c") not in frozen
    assert frozen.get((2, "c")) is None
    with pytest.raises(KeyError):
        _ = frozen[(2, "c")]
    assert dict(frozen) == {(0, "c"): ("1",), (1, "c"): ("x", "y", "z")}
    assert frozen == FrozenConventions({(0, "c"): ("1",), (1, "c"): ("x", "y", "z")})
    assert frozen != FrozenConventions({(0, "c"): ("1",)})
    assert repr(frozen) == "FrozenConventions({(0, 'c'): ('1',), (1, 'c'): ('x', 'y', 'z')})"


def test_frozen_conventions_pickle():
    frozen = FrozenConventions(CP2K_CONVENTIONS)
    assert pickle.loads(pickle.dumps(frozen)) == frozen
    obasis = MolecularBasis([], CP2K_CONVENTIONS, "L2")
    assert pickle.loads(pickle.dumps(obasis)).conventions == CP2K_CONVENTIONS


@pytest.mark.parametrize("module", [cp2klog, fchk, molden, mwfn, wfn])
def test_format_conventions_frozen(module):
    assert isinstance(module.CONVENTIONS, FrozenConventions)


def test_molecular_basis_conventions():
    conventions = {(0, "c"): ["1"], (1, "c"): ["x", "y", "z"]}
    obasis = MolecularBasis([], conventions, "L2")
    assert isinstance(obasis.conventions, FrozenConventions)
    conventions[(1, "c")].reverse()
    assert obasis.conventions[(1, "c")] == ("x", "y", "z")
    with pytest.raises(TypeError):
        obasis.conventions[(0, "p")] = ["c0"]


def test_molecular_basis_change_conventions():
    # Two basis sets that share conventions, like basis sets loaded from the same file format.
    obasis0 = MolecularBasis([], CP2K_CONVENTIONS, "L2")
    obasis1 = MolecularBasis([], CP2K_CONVENTIONS, "L2")
    # The way to change the conventions given in the docstring of MolecularBasis.conventions
    obasis0.conventions = {**obasis0.conventions, (0, "p"): ("c0",)}
    assert isinstance(obasis0.conventions, FrozenConventions)
    assert obasis0.conventions[(0, "p")] == ("c0",)
    assert (0, "p") not in obasis1.conventions
    assert (0, "p") not in CP2K_CONVENTIONS
    obasis2 = attrs.evolve(obasis1, conventions={(1, "c"): ("z", "y", "x")})
    assert obasis2.conventions == FrozenConventions({(1, "c"): ("z", "y", "x")})
    assert obasis1.conventions == CP2K_CONVENTIONS
