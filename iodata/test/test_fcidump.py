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
"""Test iodata.formats.fcidump module."""

import os
from importlib.resources import as_file, files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_one, load_one
from ..utils import LoadError


@pytest.mark.parametrize("fn", ["FCIDUMP.psi4.h2", "FCIDUMP.molpro.h2"])
def test_load_fcidump_empty_lines(fn, tmpdir):
    with as_file(files("iodata.test.data").joinpath(fn)) as path:
        mol = load_one(str(path))
        with open(path) as f:
            lines = f.read().splitlines()
    # Empty lines in the header, between the integrals, before the header
    # and at the end of the file.
    lines.insert(1, "")
    lines.insert(len(lines) - 3, "")
    lines.insert(0, "")
    path_new = os.path.join(tmpdir, "FCIDUMP.test")
    with open(path_new, "w") as f:
        f.write("\n".join(lines) + "\n\n\n  \n")
    mol_new = load_one(path_new, fmt="fcidump")
    assert_equal(mol_new.core_energy, mol.core_energy)
    assert_equal(mol_new.one_ints["core_mo"], mol.one_ints["core_mo"])
    assert_equal(mol_new.two_ints["two_mo"], mol.two_ints["two_mo"])


@pytest.mark.parametrize(
    "header",
    [
        # Everything on one line, with the end marker at the end of the line.
        " &FCI NORB=4,NELEC=2,MS2=0,ORBSYM=1,1,5,5,ISYM=1, &END\n",
        # End marker at the end of the last header line.
        " &FCI NORB=4,NELEC=2,MS2=0,\n  ORBSYM=1,1,5,5,\n  ISYM=1,/\n",
        # Lower case.
        " &fci norb=4,nelec=2,ms2=0,\n  orbsym=1,1,5,5,\n  isym=1,\n &end\n",
        # NELEC and MS2 on the second line.
        " &FCI NORB=4,\n  NELEC=2,MS2=0,\n  ORBSYM=1,1,5,5,\n  ISYM=1,\n /\n",
        # No leading space, values separated by spaces, spaces around the equal signs.
        "&FCI NORB = 4 NELEC = 2 MS2 = 0\n  ORBSYM = 1 1 5 5 ISYM = 1\n/\n",
    ],
)
def test_load_fcidump_header_variants(header, tmpdir):
    with as_file(files("iodata.test.data").joinpath("FCIDUMP.molpro.h2")) as path:
        mol = load_one(str(path))
        with open(path) as f:
            lines = f.read().splitlines(keepends=True)
    # Replace the original header, i.e. the first four lines.
    path_new = os.path.join(tmpdir, "FCIDUMP.test")
    with open(path_new, "w") as f:
        f.write(header)
        f.writelines(lines[4:])
    mol_new = load_one(path_new)
    assert mol_new.nelec == 2
    assert mol_new.spinpol == 0
    assert_equal(mol_new.core_energy, mol.core_energy)
    assert_equal(mol_new.one_ints["core_mo"], mol.one_ints["core_mo"])
    assert_equal(mol_new.two_ints["two_mo"], mol.two_ints["two_mo"])


DATA_LINES = """\
  0.6527679278914691E+00   1   1   1   1
  0.7151043364864863E+00   0   0   0   0
"""


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("", "No FCIDUMP header found"),
        ("\n  \n", "No FCIDUMP header found"),
        (" &XYZ NORB=4,NELEC=2,MS2=0,\n /\n" + DATA_LINES, "Incorrect file header"),
        # Missing end marker
        (" &FCI NORB=4,NELEC=2,MS2=0,\n  ISYM=1,\n" + DATA_LINES, "not terminated"),
        # Missing field and invalid values
        (" &FCI NORB=4,MS2=0,\n /\n" + DATA_LINES, "NELEC"),
        (" &FCI NORB=four,NELEC=2,MS2=0,\n /\n" + DATA_LINES, "NORB"),
        (" &FCI NORB=4,NELEC=2,MS2=0.5,\n /\n" + DATA_LINES, "MS2"),
        # Files without integrals, e.g. truncated after the header
        (" &FCI NORB=4,NELEC=2,MS2=0,\n /\n", "No integrals"),
        (" &FCI NORB=4,NELEC=2,MS2=0,\n /\n\n\n  \n", "No integrals"),
        # Wrong number of fields on a data line
        (" &FCI NORB=4,NELEC=2,MS2=0,\n /\n  0.65   1   1   1\n", "Expecting 5 fields"),
    ],
)
def test_load_fcidump_invalid(content, match, tmpdir):
    path = os.path.join(tmpdir, "FCIDUMP.test")
    with open(path, "w") as f:
        f.write(content)
    with pytest.raises(LoadError, match=match):
        load_one(path)


def test_load_fcidump_psi4_h2():
    with as_file(files("iodata.test.data").joinpath("FCIDUMP.psi4.h2")) as fn:
        mol = load_one(str(fn))
    assert_allclose(mol.core_energy, 0.7151043364864863e00)
    assert_equal(mol.nelec, 2)
    assert_equal(mol.spinpol, 0)
    core_mo = mol.one_ints["core_mo"]
    assert_equal(core_mo.shape, (10, 10))
    assert_allclose(core_mo[0, 0], -0.1251399119550580e01)
    assert_allclose(core_mo[2, 1], 0.9292454365115077e-01)
    assert_allclose(core_mo[1, 2], 0.9292454365115077e-01)
    assert_allclose(core_mo[9, 9], 0.9035054979531029e00)
    two_mo = mol.two_ints["two_mo"]
    assert_allclose(two_mo.shape, (10, 10, 10, 10))
    assert_allclose(two_mo[0, 0, 0, 0], 0.6589928924251115e00)
    # Check physicists' notation and symmetry
    assert_allclose(two_mo[6, 1, 5, 0], 0.5335846565304321e-01)
    assert_allclose(two_mo[5, 1, 6, 0], 0.5335846565304321e-01)
    assert_allclose(two_mo[6, 0, 5, 1], 0.5335846565304321e-01)
    assert_allclose(two_mo[5, 0, 6, 1], 0.5335846565304321e-01)
    assert_allclose(two_mo[1, 6, 0, 5], 0.5335846565304321e-01)
    assert_allclose(two_mo[1, 5, 0, 6], 0.5335846565304321e-01)
    assert_allclose(two_mo[0, 6, 1, 5], 0.5335846565304321e-01)
    assert_allclose(two_mo[0, 5, 1, 6], 0.5335846565304321e-01)
    assert_allclose(two_mo[9, 9, 9, 9], 0.6273759381091796e00)


def test_load_fcidump_molpro_h2():
    with as_file(files("iodata.test.data").joinpath("FCIDUMP.molpro.h2")) as fn:
        mol = load_one(str(fn))
    assert_allclose(mol.core_energy, 0.7151043364864863e00)
    assert_equal(mol.nelec, 2)
    assert_equal(mol.spinpol, 0)
    core_mo = mol.one_ints["core_mo"]
    assert_equal(core_mo.shape, (4, 4))
    assert_allclose(core_mo[0, 0], -0.1245406261597530e01)
    assert_allclose(core_mo[0, 1], -0.1666402467335385e00)
    assert_allclose(core_mo[1, 0], -0.1666402467335385e00)
    assert_allclose(core_mo[3, 3], 0.3216193420753873e00)
    two_mo = mol.two_ints["two_mo"]
    assert_allclose(two_mo.shape, (4, 4, 4, 4))
    assert_allclose(two_mo[0, 0, 0, 0], 0.6527679278914691e00)
    # Check physicists' notation and symmetry
    assert_allclose(two_mo[3, 0, 2, 1], 0.7756042287284058e-01)
    assert_allclose(two_mo[2, 0, 3, 1], 0.7756042287284058e-01)
    assert_allclose(two_mo[3, 1, 2, 0], 0.7756042287284058e-01)
    assert_allclose(two_mo[2, 1, 3, 0], 0.7756042287284058e-01)
    assert_allclose(two_mo[0, 3, 1, 2], 0.7756042287284058e-01)
    assert_allclose(two_mo[0, 2, 1, 3], 0.7756042287284058e-01)
    assert_allclose(two_mo[1, 3, 0, 2], 0.7756042287284058e-01)
    assert_allclose(two_mo[1, 2, 0, 3], 0.7756042287284058e-01)
    assert_allclose(two_mo[3, 3, 3, 3], 0.7484308847738417e00)


def test_dump_load_fcidimp_consistency_ao(tmpdir):
    # Setup IOData
    with as_file(files("iodata.test.data").joinpath("water.xyz")) as fn:
        mol0 = load_one(str(fn))
    mol0.nelec = 10
    mol0.spinpol = 0
    with as_file(files("iodata.test.data").joinpath("psi4_h2_one.npy")) as fn:
        mol0.one_ints = {"core_mo": np.load(str(fn))}
    with as_file(files("iodata.test.data").joinpath("psi4_h2_two.npy")) as fn:
        mol0.two_ints = {"two_mo": np.load(str(fn))}

    # Dump to a file and load it again
    fn_tmp = os.path.join(tmpdir, "tmp.fcidump")
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)

    # Compare results
    assert_equal(mol0.nelec, mol1.nelec)
    assert_equal(mol0.spinpol, mol1.spinpol)
    assert_allclose(mol0.one_ints["core_mo"], mol1.one_ints["core_mo"])
    assert_allclose(mol0.two_ints["two_mo"], mol1.two_ints["two_mo"])
