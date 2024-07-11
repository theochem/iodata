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
"""Unit tests for iodata.prepare."""

from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_one, load_one
from ..basis import MolecularBasis, Shell
from ..convert import HORTON2_CONVENTIONS
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..prepare import prepare_segmented, prepare_unrestricted_aminusb
from ..utils import PrepareDumpError, PrepareDumpWarning


def test_unrestricted_aminusb_no_mo():
    data = IOData()
    with pytest.raises(ValueError):
        prepare_unrestricted_aminusb(data, False, "foo.wfn", "wfn")


def test_unrestricted_aminusb_generaized():
    data = IOData(mo=MolecularOrbitals("generalized", None, None))
    with pytest.raises(ValueError):
        prepare_unrestricted_aminusb(data, False, "foo.wfn", "wfn")


def test_unrestricted_aminusb_pass_through1():
    data1 = IOData(mo=MolecularOrbitals("unrestricted", 5, 3))
    data2 = prepare_unrestricted_aminusb(data1, False, "foo.wfn", "wfn")
    assert data1 is data2


def test_unrestricted_aminusb_pass_through2():
    data1 = IOData(mo=MolecularOrbitals("restricted", 4, 4, occs=[2, 1, 0, 0]))
    data2 = prepare_unrestricted_aminusb(data1, False, "foo.wfn", "wfn")
    assert data1 is data2


def test_unrestricted_aminusb_pass_error():
    data = IOData(
        mo=MolecularOrbitals(
            "restricted", 4, 4, occs=[2, 1, 0, 0], occs_aminusb=[0.7, 0.3, 0.0, 0.0]
        )
    )
    with pytest.raises(PrepareDumpError):
        prepare_unrestricted_aminusb(data, False, "foo.wfn", "wfn")


def test_unrestricted_aminusb_pass_warning():
    data1 = IOData(
        atnums=[1, 1],
        mo=MolecularOrbitals(
            "restricted", 4, 4, occs=[2, 1, 0, 0], occs_aminusb=[0.7, 0.3, 0.0, 0.0]
        ),
    )
    with pytest.warns(PrepareDumpWarning):
        data2 = prepare_unrestricted_aminusb(data1, True, "foo.wfn", "wfn")
    assert data1 is not data2
    assert data1.atnums is data2.atnums
    assert data1.mo is not data2.mo
    assert data2.mo.kind == "unrestricted"
    assert_equal(data2.mo.occsa, data1.mo.occsa)
    assert_equal(data2.mo.occsb, data1.mo.occsb)


@pytest.mark.parametrize("fmt", ["wfn", "wfx", "molden", "molekel"])
def test_dump_occs_aminusb(tmpdir, fmt):
    # Load a restricted spin-paired wfn and alter it.abs
    with as_file(files("iodata.test.data").joinpath("water_sto3g_hf_g03.fchk")) as fn_fchk:
        data1 = load_one(fn_fchk)
    assert data1.mo.kind == "restricted"
    data1.mo.occs = [2, 2, 2, 1, 1, 1, 0]
    data1.mo.occs_aminusb = [0, 0, 0, 1, 0.7, 0.3, 0]
    assert_allclose(data1.spinpol, data1.mo.spinpol)

    # Dump and load again
    path_foo = Path(tmpdir) / "foo"
    with pytest.raises(PrepareDumpError):
        dump_one(data1, path_foo, fmt=fmt)
    with pytest.warns(PrepareDumpWarning):
        dump_one(data1, path_foo, allow_changes=True, fmt=fmt)
    data2 = load_one(path_foo, fmt=fmt)

    # Check the loaded file
    assert data2.mo.kind == "unrestricted"
    assert_allclose(data2.mo.occsa, data1.mo.occsa)
    assert_allclose(data2.mo.occsb, data1.mo.occsb)


def test_segmented_no_basis():
    data = IOData()
    with pytest.raises(ValueError):
        prepare_segmented(data, False, False, "foo.wfn", "wfn")


def test_segmented_not_generalized():
    data = IOData(
        obasis=MolecularBasis(
            [
                Shell(0, [0], ["c"], [0.5, 0.01], [[0.1], [0.2]]),
                Shell(1, [2], ["p"], [1.1], [[0.3]]),
            ],
            HORTON2_CONVENTIONS,
            "L2",
        )
    )
    assert data is prepare_segmented(data, False, False, "foo.wfn", "wfn")


def test_segmented_generalized():
    rng = np.random.default_rng(1)
    data0 = IOData(
        obasis=MolecularBasis(
            [
                Shell(0, [0, 1], ["c", "c"], rng.uniform(0, 1, 3), rng.uniform(-1, 1, (3, 2))),
                Shell(1, [2, 3], ["p", "p"], rng.uniform(0, 1, 4), rng.uniform(-1, 1, (4, 2))),
            ],
            HORTON2_CONVENTIONS,
            "L2",
        )
    )
    with pytest.raises(PrepareDumpError):
        prepare_segmented(data0, False, False, "foo.wfn", "wfn")
    with pytest.warns(PrepareDumpWarning):
        data1 = prepare_segmented(data0, False, True, "foo.wfn", "wfn")
    assert len(data1.obasis.shells) == 4


def test_segmented_sp():
    rng = np.random.default_rng(1)
    data0 = IOData(
        obasis=MolecularBasis(
            [
                Shell(0, [0, 1], ["c", "c"], rng.uniform(0, 1, 3), rng.uniform(-1, 1, (3, 2))),
                Shell(1, [2, 3], ["p", "p"], rng.uniform(0, 1, 4), rng.uniform(-1, 1, (4, 2))),
            ],
            HORTON2_CONVENTIONS,
            "L2",
        )
    )
    with pytest.raises(PrepareDumpError):
        prepare_segmented(data0, True, False, "foo.wfn", "wfn")
    with pytest.warns(PrepareDumpWarning):
        data1 = prepare_segmented(data0, True, True, "foo.wfn", "wfn")
    assert len(data1.obasis.shells) == 3
