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
"""Unit tests for harmonics.py"""

import sympy as sp
from harmonics import real_regular_solid_harmonics


def test_manual():
    x, y, z = sp.symbols("x y z")
    r2 = x * x + y * y + z * z
    rrsh = real_regular_solid_harmonics(x, y, z, 3, sqrt=sp.sqrt)
    assert rrsh[0] == 1
    assert rrsh[1] == z
    assert rrsh[2] == x
    assert rrsh[3] == y
    assert rrsh[4].expand() == (sp.Rational(3, 2) * z**2 - r2 / 2).expand()
    assert rrsh[5].expand() == (sp.sqrt(3) * x * z)
    assert rrsh[6].expand() == (sp.sqrt(3) * y * z)
    assert rrsh[7].expand() == (sp.sqrt(3) / 2 * (x**2 - y**2)).expand()
    assert rrsh[8].expand() == (sp.sqrt(3) * x * y)
    assert rrsh[9].expand() == (sp.Rational(5, 2) * z**3 - sp.Rational(3, 2) * r2 * z).expand()
    assert (
        rrsh[10].expand()
        == (x / sp.sqrt(6) * (sp.Rational(15, 2) * z**2 - sp.Rational(3, 2) * r2)).expand()
    )
    assert (
        rrsh[11].expand()
        == (y / sp.sqrt(6) * (sp.Rational(15, 2) * z**2 - sp.Rational(3, 2) * r2)).expand()
    )
    assert rrsh[12].expand() == (sp.sqrt(15) * z / 2 * (x**2 - y**2)).expand()
    assert rrsh[13].expand() == (sp.sqrt(15) * x * y * z).expand()
    assert rrsh[14].expand() == (sp.sqrt(10) / 4 * (x**3 - 3 * x * y**2)).expand()
    assert rrsh[15].expand() == (sp.sqrt(10) / 4 * (3 * x**2 * y - y**3)).expand()


def test_library():
    theta = sp.Symbol("theta", real=True)
    phi = sp.Symbol("phi", real=True)
    x, y, z, r = sp.symbols("x y z r", real=True)
    r2 = x * x + y * y + z * z
    ellmax = 5
    rrsh = real_regular_solid_harmonics(x, y, z, ellmax, sqrt=sp.sqrt)

    def _get_cartfn(ell, m):
        """Return the reference result in Cartesian coordinates with SymPy."""
        # Take the complex spherical harmonics from SymPy and write in terms
        # of simple trigoniometric functions.
        ref = sp.Ynm(ell, abs(m), theta, phi)
        ref = ref.expand(func=True)
        ref = ref.subs(sp.exp(sp.I * phi), sp.cos(phi) + sp.I * sp.sin(phi))
        # Take the definition of real functions from Wikipedia, not from
        # SymPy. The latter has an incompatible sign for the sin-like
        # functions.
        if m > 0:
            ref = sp.re(ref) * sp.sqrt(2)
        elif m < 0:
            ref = sp.im(ref) * sp.sqrt(2)
        # Undo the Condon-Shortley phase
        ref *= (-sp.Integer(1)) ** abs(m)
        # Convert to regular solid harmonics
        ref = (sp.sqrt(4 * sp.pi / (2 * ell + 1)) * ref * r**ell).expand()
        # From spherical to Cartesian coordinates
        ref = ref.subs(sp.cos(phi), x / (r * sp.sin(theta)))
        ref = ref.subs(sp.sin(phi), y / (r * sp.sin(theta)))
        ref = ref.subs(sp.cos(theta), z / r)
        ref = ref.subs(sp.sin(theta), sp.sqrt(x * x + y * y) / r)
        ref = ref.subs(r, sp.sqrt(r2)).expand()
        return sp.simplify(ref).expand()

    assert rrsh.pop(0) == 1
    for ell in range(1, ellmax + 1):
        for m in range(ell + 1):
            # cosine line
            assert rrsh.pop(0).expand() == _get_cartfn(ell, m), (ell, m)
            if m > 0:
                # sine like
                assert rrsh.pop(0).expand() == _get_cartfn(ell, -m), (ell, m)
