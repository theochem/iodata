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
import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform.startswith("win"), reason="TrexIO issues on Windows")
def test_load_dump_consistency(tmp_path):
    # Skip tests if trexio is not installed, but do NOT import it here to avoid segfaults
    if importlib.util.find_spec("trexio") is None:
        pytest.skip("trexio not installed")
    """Check if dumping and loading a TREXIO file results in the same data.

    Runs in a subprocess to avoid segmentation faults caused by conflict
    between pytest execution model and trexio C-extension.
    """
    script = """
import numpy as np
import os
import sys

from iodata import IOData
from iodata.api import load_one, dump_one

atcoords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
atnums = np.array([1, 1])
nelec = 2.0
spinpol = 0
iodata_orig = IOData(atcoords=atcoords, atnums=atnums, nelec=nelec, spinpol=spinpol)

filename = "test.trexio"
if os.path.exists(filename):
    os.remove(filename)

print(f"Dumping to {filename}")
dump_one(iodata_orig, filename, fmt="trexio")

print(f"Loading from {filename}")
iodata_new = load_one(filename, fmt="trexio")

print("Verifying data...")
np.testing.assert_allclose(iodata_new.atcoords, atcoords, err_msg="atcoords mismatch")
np.testing.assert_equal(iodata_new.atnums, atnums, err_msg="atnums mismatch")
np.testing.assert_allclose(
    float(iodata_new.nelec),
    nelec,
    rtol=1.0e-8,
    atol=1.0e-12,
    err_msg=f"nelec mismatch: {iodata_new.nelec} != {nelec}",
)
np.testing.assert_allclose(
    float(iodata_new.charge),
    0.0,
    rtol=1.0e-8,
    atol=1.0e-12,
    err_msg=f"charge mismatch: {iodata_new.charge} != 0.0",
)
assert int(iodata_new.spinpol) == spinpol, (
    f"spinpol mismatch: {iodata_new.spinpol} != {spinpol}"
)

print("Verification passed")
"""
    script_file = tmp_path / "verify_trexio_subprocess.py"
    script_file.write_text(script, encoding="utf-8")

    # Determine project root (assuming this test is in iodata/test/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))

    # Add project root to PYTHONPATH to ensure local iodata code is used
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"

    subprocess.check_call([sys.executable, str(script_file)], cwd=tmp_path, env=env)
