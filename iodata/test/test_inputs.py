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
"""Test iodata.inputs module."""

import os
from importlib.resources import as_file, files

import numpy as np
import pytest

from ..api import load_one, write_input
from ..iodata import IOData
from ..periodic import num2sym
from ..utils import FileFormatWarning, angstrom


def check_load_input_and_compare(fname: str, fname_expected: str):
    """Load saved input file and compare to expected input file.

    Parameters
    ----------
    fname : str
        Path to input file name to load.
    fname_expected : str
        Path to expected input file to load.

    """
    with open(fname) as ifn:
        content = "".join(ifn.readlines())
    with open(fname_expected) as efn:
        expected = "".join(efn.readlines())
    assert content == expected


def test_input_gaussian_from_xyz(tmpdir):
    # load geometry from xyz file & add level of theory & basis set
    with as_file(files("iodata.test.data").joinpath("water_number.xyz")) as fn:
        mol = load_one(fn)
    mol.nelec = 10
    mol.lot = "ub3lyp"
    mol.obasis_name = "6-31g*"
    # write input in a temporary folder using the user-template
    fname = os.path.join(tmpdir, "input_from_xyz.com")
    template = """\
%chk=gaussian.chk
%mem=3500MB
%nprocs=4
#p {lot}/{obasis_name} opt scf(tight,xqc,fermi) integral(grid=ultrafine) {extra_cmd}

{title} {lot}/{obasis_name} opt-force

0 1
{geometry}

--Link1--
%chk=gaussian.chk
%mem=3500MB
%nprocs=4
#p {lot}/{obasis_name} force guess=read geom=allcheck integral(grid=ultrafine) output=wfn

gaussian.wfn


"""
    write_input(mol, fname, fmt="gaussian", template=template, extra_cmd="nosymmetry")
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_gaussian_h2o_opt_ub3lyp.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)


def test_input_gaussian_from_iodata(tmpdir):
    # make an instance of IOData for HCl anion
    data = {
        "atcoords": np.array([[0.0, 0.0, 0.0], [angstrom, 0.0, 0.0]]),
        "atnums": np.array([1, 17]),
        "nelec": 19,
        "run_type": "opt",
        "spinpol": 1,
    }
    mol = IOData(**data)
    # write input in a temporary file
    fname = os.path.join(tmpdir, "input_from_iodata.com")
    write_input(mol, fname, fmt="gaussian")
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_gaussian_hcl_anion_opt_hf.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)


def test_input_gaussian_from_fchk(tmpdir):
    # load fchk
    with as_file(files("iodata.test.data").joinpath("water_hfs_321g.fchk")) as fn:
        mol = load_one(fn)
    # write input in a temporary file
    fname = os.path.join(tmpdir, "input_from_fchk.in")
    write_input(mol, fname, fmt="gaussian")
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_gaussian_hcl_sp_rhf.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)


def test_input_gaussian_atom_line(tmpdir):
    template = """\
# {lot}/{obasis_name} Counterpoise=2

Counterpoise calculation on {extra[name]}

0,1 0,1 0,1
{geometry}

"""

    def atom_line(data, iatom):
        symbol = num2sym[data.atnums[iatom]]
        atcoord = data.atcoords[iatom] / angstrom
        fid = data.extra["fragment_ids"][iatom]
        return f"{symbol}(Fragment={fid}) {atcoord[0]:10.6f} {atcoord[1]:10.6f} {atcoord[2]:10.6f}"

    with as_file(files("iodata.test.data").joinpath("s66_4114_02WaterMeOH.xyz")) as fn:
        mol = load_one(fn, "extxyz")

    fn_com = os.path.join(tmpdir, "input_bsse.com")
    write_input(mol, fn_com, "gaussian", template, atom_line)
    with as_file(files("iodata.test.data").joinpath("input_gaussian_bsse.com")) as fn_expected:
        check_load_input_and_compare(fn_com, fn_expected)


def test_input_orca_from_xyz(tmpdir):
    # load geometry from xyz file & add level of theory & basis set
    with as_file(files("iodata.test.data").joinpath("water_number.xyz")) as fn:
        mol = load_one(fn)
    mol.nelec = 10
    mol.lot = "B3LYP"
    mol.obasis_name = "def2-SVP"
    # write input in a temporary folder using the user-template
    fname = os.path.join(tmpdir, "input_from_xyz.com")
    template = """\
! {lot} {obasis_name} {grid_stuff} KeepDens
# {title}
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs 4 end
%coords
    CTyp xyz
    Charge {charge}
    Mult {spinmult}
    Units Angs
    coords
{geometry}
    end
end
"""

    def atom_line(data, iatom):
        """Construct custom atom_line with indentation."""
        symbol = num2sym[data.atnums[iatom]]
        atcoord = data.atcoords[iatom] / angstrom
        return f"        {symbol:3s} {atcoord[0]:10.6f} {atcoord[1]:10.6f} {atcoord[2]:10.6f}"

    grid_stuff = "Grid4 TightSCF NOFINALGRID"
    write_input(mol, fname, "orca", template, atom_line, grid_stuff=grid_stuff)
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_orca_h2o_sp_b3lyp.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)


def test_input_orca_from_iodata(tmpdir):
    # make an instance of IOData for HCl anion
    data = {
        "atcoords": np.array([[0.0, 0.0, 0.0], [angstrom, 0.0, 0.0]]),
        "atnums": np.array([1, 17]),
        "nelec": 19,
        "run_type": "opt",
        "spinpol": 1,
    }
    mol = IOData(**data)
    # write input in a temporary file
    fname = os.path.join(tmpdir, "input_from_iodata.com")
    write_input(mol, fname, fmt="orca")
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_orca_hcl_anion_opt_hf.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)


def test_input_orca_from_molden(tmpdir):
    # load orca molden
    with (
        as_file(files("iodata.test.data").joinpath("nh3_orca.molden")) as fn,
        pytest.warns(FileFormatWarning),
    ):
        mol = load_one(fn)
    # write input in a temporary file
    fname = os.path.join(tmpdir, "input_from_molden.in")
    write_input(mol, fname, fmt="orca")
    # compare saved input to expected input
    source = files("iodata.test.data").joinpath("input_orca_nh3_sp_hf.txt")
    with as_file(source) as fname_expected:
        check_load_input_and_compare(fname, fname_expected)
