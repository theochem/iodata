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
"""Unit tests for iodata.__main__."""

import os
import subprocess
import sys
from functools import partial
from importlib.resources import as_file, files
from typing import Optional
from warnings import warn

import pytest
from numpy.testing import assert_allclose, assert_equal

from ..__main__ import convert as convfn
from ..api import load_many, load_one
from ..utils import FileFormatError, PrepareDumpError, PrepareDumpWarning


def _convscript(
    infn: str,
    outfn: str,
    many: bool = False,
    infmt: Optional[str] = None,
    outfmt: Optional[str] = None,
    allow_changes: bool = False,
):
    """Simulate the convert function by calling iodata-convert in a subprocess."""
    args = [sys.executable, "-m", "iodata.__main__", infn, outfn]
    if many:
        args.append("-m")
    if infmt is not None:
        args.append(f"--infmt={infmt}")
    if outfmt is not None:
        args.append(f"--outfmt={outfmt}")
    if allow_changes:
        args.append("-c")
    cp = subprocess.run(args, capture_output=True, check=False, encoding="utf8")
    if cp.returncode == 0:
        if allow_changes and "PrepareDumpWarning" in cp.stderr:
            warn(PrepareDumpWarning(cp.stderr), stacklevel=2)
    else:
        if "PrepareDumpError" in cp.stderr:
            raise PrepareDumpError(cp.stderr)
        if "FileFormatError" in cp.stderr:
            raise FileFormatError(cp.stderr)
        raise RuntimeError(f"Failure not processed.\n{cp.stderr}")


def _check_convert_one(myconvert, tmpdir):
    outfn = os.path.join(tmpdir, "tmp.xyz")
    with as_file(files("iodata.test.data").joinpath("hf_sto3g.fchk")) as infn:
        myconvert(infn, outfn, allow_changes=False)
    iodata = load_one(outfn)
    assert iodata.natom == 2
    assert_equal(iodata.atnums, [9, 1])
    assert_allclose(iodata.atcoords, [[0.0, 0.0, 0.190484394], [0.0, 0.0, -1.71435955]])


def _check_convert_one_changes(myconvert, tmpdir):
    outfn = os.path.join(tmpdir, "tmp.mkl")
    with as_file(files("iodata.test.data").joinpath("hf_sto3g.fchk")) as infn:
        with pytest.raises(PrepareDumpError):
            myconvert(infn, outfn, allow_changes=False)
        assert not os.path.isfile(outfn)
        with pytest.warns(PrepareDumpWarning):
            myconvert(infn, outfn, allow_changes=True)
    iodata = load_one(outfn)
    assert iodata.natom == 2
    assert_equal(iodata.atnums, [9, 1])
    assert_allclose(iodata.atcoords, [[0.0, 0.0, 0.190484394], [0.0, 0.0, -1.71435955]])


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_autofmt(tmpdir, convert):
    myconvert = partial(convfn, many=False, infmt=None, outfmt=None)
    _check_convert_one(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_autofmt_changes(tmpdir, convert):
    myconvert = partial(convert, many=False, infmt=None, outfmt=None)
    _check_convert_one_changes(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_manfmt(tmpdir, convert):
    myconvert = partial(convert, many=False, infmt="fchk", outfmt="xyz")
    _check_convert_one(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_nonexisting_infmt(tmpdir, convert):
    myconvert = partial(convert, many=False, infmt="blablabla", outfmt="xyz")
    with pytest.raises(FileFormatError):
        _check_convert_one(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_nonexisting_outfmt(tmpdir, convert):
    myconvert = partial(convert, many=False, infmt="fchk", outfmt="blablabla")
    with pytest.raises(FileFormatError):
        _check_convert_one(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_one_manfmt_changes(tmpdir, convert):
    myconvert = partial(convert, many=False, infmt="fchk", outfmt="molekel")
    _check_convert_one_changes(myconvert, tmpdir)


def _check_convert_many(myconvert, tmpdir):
    outfn = os.path.join(tmpdir, "tmp.xyz")
    with as_file(files("iodata.test.data").joinpath("peroxide_relaxed_scan.fchk")) as infn:
        myconvert(infn, outfn, allow_changes=False)
    trj = list(load_many(outfn))
    assert len(trj) == 13
    for iodata in trj:
        assert iodata.natom == 4
        assert_equal(iodata.atnums, [8, 8, 1, 1])
    assert_allclose(trj[1].atcoords[3], [-1.85942837, -1.70565735, 0.0], atol=1e-5)
    assert_allclose(trj[5].atcoords[0], [0.0, 1.32466211, 0.0], atol=1e-5)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_many_autofmt(tmpdir, convert):
    myconvert = partial(convert, many=True, infmt=None, outfmt=None)
    _check_convert_many(myconvert, tmpdir)


@pytest.mark.parametrize("convert", [convfn, _convscript])
def test_convert_many_manfmt(tmpdir, convert):
    myconvert = partial(convert, many=True, infmt="fchk", outfmt="xyz")
    _check_convert_many(myconvert, tmpdir)
