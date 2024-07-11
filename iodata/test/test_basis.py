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

import attrs
import numpy as np
import pytest

from ..basis import (
    MolecularBasis,
    Shell,
    angmom_its,
    angmom_sti,
)
from ..formats.cp2klog import CONVENTIONS as CP2K_CONVENTIONS


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
            (0, "c"): ["s"],
            (1, "c"): ["x", "z", "-y"],
            (2, "p"): ["dc0", "dc1", "-ds1", "dc2", "-ds2"],
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
