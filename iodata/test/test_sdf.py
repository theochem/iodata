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
"""Test iodata.formats.sdf module."""

import os
from importlib.resources import as_file, files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_many, dump_one, load_many, load_one
from ..iodata import IOData
from ..utils import LoadError, PrepareDumpError, angstrom
from .common import check_dump_multiline_title, truncated_file


def test_sdf_load_one_example():
    # test sdf one structure
    with as_file(files("iodata.test.data").joinpath("example.sdf")) as fn_sdf:
        mol = load_one(str(fn_sdf))
    check_example(mol)


def test_sdf_load_one_formamide():
    # test sdf one structure
    with as_file(files("iodata.test.data").joinpath("formamide.sdf")) as fn_sdf:
        mol = load_one(str(fn_sdf))
    assert mol.title == "713"
    assert mol.natom == 6
    assert len(mol.bonds) == 5
    assert_equal(mol.atnums, [8, 7, 6, 1, 1, 1])
    assert_equal(mol.bonds, [[0, 2, 2], [1, 2, 1], [1, 3, 1], [1, 4, 1], [2, 5, 1]])


def test_sdf_formaterror(tmpdir):
    # test if sdf file has the wrong ending without $$$$
    with (
        as_file(files("iodata.test.data").joinpath("example.sdf")) as fn_test,
        truncated_file(fn_test, 36, 0, tmpdir) as fn,
        pytest.raises(LoadError),
    ):
        load_one(str(fn))


def check_example(mol):
    """Test some things on example file."""
    assert mol.title == "24978498"
    assert mol.natom == 16
    assert len(mol.bonds) == 15
    assert_equal(mol.atnums, [16, 8, 8, 8, 8, 7, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1])
    # check coordinates
    atcoords_ang = mol.atcoords / angstrom
    assert_allclose(atcoords_ang[0], [2.8660, -0.4400, 0.0000])
    assert_allclose(atcoords_ang[1], [5.4641, 1.0600, 0.0000])
    assert_allclose(atcoords_ang[14], [6.0010, 1.3700, 0.0000])
    assert_allclose(atcoords_ang[15], [2.0000, -2.5600, 0.0000])
    assert_equal(mol.bonds[0], [0, 3, 1])
    assert_equal(mol.bonds[4], [2, 8, 2])
    assert_equal(mol.bonds[14], [7, 11, 1])


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an SDF file results in the same data."""
    mol0 = load_one(str(fn))
    # write sdf file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, "test.sdf")
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # check two sdf files
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.0e-5)
    assert_equal(mol0.bonds, mol1.bonds)


def test_load_dump_consistency(tmpdir):
    with as_file(files("iodata.test.data").joinpath("example.sdf")) as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)
    with as_file(files("iodata.test.data").joinpath("formamide.sdf")) as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)
    # The benzene mol2 file has aromatic bonds, which are less common in SDF files.
    with as_file(files("iodata.test.data").joinpath("benzene.mol2")) as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)


def test_fixed_width_fields_touching(tmpdir):
    # In the V2000 format, fields have a fixed width and may touch each other:
    # the numbers of atoms and bonds in the counts line, atom indices in the bond lines
    # and coordinates below -1000.
    natom = 120
    rng = np.random.default_rng(1)
    atcoords = rng.uniform(-2000, 2000, (natom, 3)).round(4) * angstrom
    bonds = np.array([[i, i + 1, 1] for i in range(natom - 1)])
    mol0 = IOData(atnums=np.full(natom, 6), atcoords=atcoords, bonds=bonds, title="chain")
    fn_sdf = os.path.join(tmpdir, "chain.sdf")
    dump_one(mol0, fn_sdf)
    with open(fn_sdf) as f:
        lines = f.read().splitlines()
    assert lines[3].startswith("120119")
    assert any(line.startswith("100101") for line in lines)
    mol1 = load_one(fn_sdf)
    assert_allclose(mol1.atcoords, atcoords, atol=1e-8)
    assert_equal(mol1.atnums, mol0.atnums)
    assert_equal(mol1.bonds, bonds)


@pytest.mark.parametrize(
    "sdf_text",
    [
        # The numbers in the counts line do not follow the column layout.
        "loose\n\n\n2 1 0 0 0 0 0 0 0 0999 V2000\n",
        # The coordinates of an atom are separated by single spaces only.
        "loose\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n0.0 0.0 0.0 C\nM  END\n$$$$\n",
        # Unknown element symbol.
        (
            "loose\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 Xx  0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n$$$$\n"
        ),
        # The bond line does not follow the column layout.
        (
            "loose\n\n\n  2  1  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "1 2 2\nM  END\n$$$$\n"
        ),
    ],
)
def test_not_fixed_width(sdf_text, tmpdir):
    # Fields are read by column, a file that does not follow the layout is rejected.
    fn_sdf = os.path.join(tmpdir, "loose.sdf")
    with open(fn_sdf, "w") as f:
        f.write(sdf_text)
    with pytest.raises(LoadError):
        load_one(fn_sdf)


@pytest.mark.parametrize(("natom", "nbond"), [(1000, 0), (3, 1000)])
def test_dump_too_large(natom, nbond, tmpdir):
    # The counts have three digits in the V2000 format.
    atcoords = np.zeros((natom, 3))
    bonds = np.array([[0, 1, 1]] * nbond) if nbond else None
    mol = IOData(atnums=np.full(natom, 6), atcoords=atcoords, bonds=bonds)
    with pytest.raises(PrepareDumpError):
        dump_one(mol, os.path.join(tmpdir, "large.sdf"))


def test_load_many():
    with as_file(files("iodata.test.data").joinpath("example.sdf")) as fn_sdf:
        mols = list(load_many(str(fn_sdf)))
    assert len(mols) == 2
    check_example(mols[0])
    assert mols[1].title == "24978481"
    assert_equal(mols[1].natom, 21)
    assert_allclose(mols[0].atcoords[0] / angstrom, [2.8660, -0.4400, 0.0000])
    assert_allclose(mols[1].atcoords[1] / angstrom, [1.4030, 1.4030, 0.0000])


def test_load_dump_many_consistency(tmpdir):
    with as_file(files("iodata.test.data").joinpath("example.sdf")) as fn_sdf:
        mols0 = list(load_many(str(fn_sdf)))
    # write sdf file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, "test")
    dump_many(mols0, fn_tmp, fmt="sdf")
    mols1 = list(load_many(fn_tmp, fmt="sdf"))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1, strict=True):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.0e-5)
        assert_equal(mol0.bonds, mol1.bonds)


def test_v2000_check():
    with (
        as_file(files("iodata.test.data").joinpath("molv3000.sdf")) as fn_sdf,
        pytest.raises(LoadError),
    ):
        load_one(fn_sdf)


@pytest.mark.parametrize("fn", ["example.sdf", "formamide.sdf"])
def test_sdf_load_one_whitespace(fn, tmpdir):
    with as_file(files("iodata.test.data").joinpath(fn)) as fn_sdf:
        mol = load_one(fn_sdf)
        with open(fn_sdf) as f:
            lines = f.read().splitlines()
    # No line break after the last line.
    fn_new = os.path.join(tmpdir, "no_final_line_break.sdf")
    with open(fn_new, "w") as f:
        f.write("\n".join(lines))
    mol_new = load_one(fn_new)
    assert_equal(mol_new.atnums, mol.atnums)
    assert_equal(mol_new.bonds, mol.bonds)
    assert_allclose(mol_new.atcoords, mol.atcoords)
    # Spaces at the end of every line.
    fn_new = os.path.join(tmpdir, "trailing_spaces.sdf")
    with open(fn_new, "w") as f:
        f.write("".join(line + "  \n" for line in lines))
    mol_new = load_one(fn_new)
    assert_equal(mol_new.atnums, mol.atnums)
    assert_equal(mol_new.bonds, mol.bonds)
    assert_allclose(mol_new.atcoords, mol.atcoords)


def test_dump_multiline_title(tmpdir):
    with as_file(files("iodata.test.data").joinpath("example.sdf")) as fn:
        mol = load_one(fn)
    check_dump_multiline_title(mol, tmpdir, "sdf")
