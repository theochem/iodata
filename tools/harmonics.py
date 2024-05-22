#!/usr/bin/env python3
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
"""Build transformation matrices from Cartesian to pure basis functions."""

import argparse

import numpy as np
import sympy as sp


def main():
    """Print transformation from Cartesian to pure functions."""
    args = parse_args()

    # Build the bare transformation matrices, without normalization
    tfs = get_bare_transforms(args.ellmax)

    if args.norm == "none":
        # No changes needed.
        pass
    elif args.norm == "L2":
        # Update the transformation matrices to transform L2 normalized
        # Cartesian Gaussian primitives into L2 normalized pure ones.
        include_l2_norm(tfs)
    else:
        raise NotImplementedError

    # Print in the selected format
    if args.format == "latex":
        print_latex(tfs)
    elif args.format == "python":
        print_python(tfs)
    else:
        raise NotImplementedError


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="harmonics")
    parser.add_argument("norm", help="Normalization convention", choices=["none", "L2"])
    parser.add_argument("format", help="Output format", choices=["latex", "python"])
    parser.add_argument("ellmax", help="Maximum angular momentum", type=int)
    return parser.parse_args()


def get_bare_transforms(ellmax: int):
    """Build transformation matrices up to ellmax, without normalization constants.

    Parameters
    ----------
    ellmax
        Matrices for ell going from 0 to ellmax (inclusive) are returned.

    Returns
    -------
    tfs
        A list of transformation matrices. Each matrix is a NumPy array of
        SymPy expressions.

    """
    x, y, z = sp.symbols("x y z", real=True)
    rrsh = real_regular_solid_harmonics(x, y, z, ellmax, sqrt=sp.sqrt)
    rrsh.pop(0)
    tfs = [np.array([[sp.Integer(1)]])]
    for ell in range(1, ellmax + 1):
        tf = []
        for _m in range(2 * ell + 1):
            row = []
            poly = sp.Poly(rrsh.pop(0), x, y, z)
            lookup = {
                (int(nx), int(ny), int(nz)): coeff
                for (nx, ny, nz), coeff in zip(poly.monoms(), poly.coeffs())
            }
            for nx, ny, nz in iter_mononomials(ell):
                row.append(sp.sympify(lookup.get((nx, ny, nz), 0)))
            tf.append(row)
        tfs.append(np.array(tf))
    return tfs


def iter_mononomials(ell):
    """Iterate over Cartesian mononomials of given angular momentum."""
    for nx in range(ell, -1, -1):
        for ny in range(ell - nx, -1, -1):
            nz = ell - nx - ny
            yield nx, ny, nz


def real_regular_solid_harmonics(x, y, z, ellmax, sqrt=None):
    """Evaluate the real regular solid harmonics up the ell=ellmax.

    Parameters
    ----------
    x, y and z
        Cartesian coordinates where the function must be evaluated.
    ellmax
        Maximum angular monentum quantum number to consider.
    sqrt
        An alternative sqrt function to use, e.g. sympy.sqrt. The default is
        the one defined in numpy.

    Returns
    -------
    result
        List of real regular solid harmonics up to angular momentum ellmax.
        The order is following HORTON2 conventions, i.e.

        .. code-block::

            result = [
                C00,
                C10, C11, S11,
                C20, C21, S21, C22, S22,
                ...
            ]

    """
    if sqrt is None:
        sqrt = np.sqrt
    result = [1, z, x, y]
    r2 = x * x + y * y + z * z
    offset2 = 0
    offset1 = 1
    for ell in range(2, ellmax + 1):
        # case m = 0 (cosine only)
        p1 = 2 * ell - 1
        p2 = ell - 1
        p3 = ell
        c_l2_0 = result[offset2]  # C_{ell-2, 0}
        c_l1_0 = result[offset1]  # C_{ell-1, 0}
        result.append((p1 * z * c_l1_0 - p2 * r2 * c_l2_0) / p3)
        # case m = 1 ... ell - 2
        for m in range(1, ell - 1):
            p2 = sqrt((ell + m - 1) * (ell - m - 1))
            p3 = sqrt((ell + m) * (ell - m))
            c_l2_m = result[offset2 + 2 * m - 1]  # C_{ell-2, m}
            c_l1_m = result[offset1 + 2 * m - 1]  # C_{ell-1, m}
            result.append((p1 * z * c_l1_m - p2 * r2 * c_l2_m) / p3)
            s_l2_m = result[offset2 + 2 * m]  # S_{ell-2, m}
            s_l1_m = result[offset1 + 2 * m]  # S_{ell-1, m}
            result.append((p1 * z * s_l1_m - p2 * r2 * s_l2_m) / p3)
        # case m = ell - 1
        p4 = sqrt(p1)
        c_l1_m = result[offset1 + 2 * ell - 3]  # C_{ell-1, ell-1}
        result.append(p4 * z * c_l1_m)
        s_l1_m = result[offset1 + 2 * ell - 2]  # S_{ell-1, ell-1}
        result.append(p4 * z * s_l1_m)
        # case m = ell
        p5 = p4 / sqrt(2 * ell)
        result.append(p5 * (x * c_l1_m - y * s_l1_m))
        result.append(p5 * (x * s_l1_m + y * c_l1_m))
        # shift offsets
        offset2 = offset1
        offset1 += p1
    return result


def fac2(n):
    """Compute the double factorial."""
    return np.prod(range(n, 0, -2))


def get_cart_l2_norm(alpha, nx, ny, nz):
    """Compute the norm of a Cartesian gaussian primitive."""
    return sp.sqrt(
        int(fac2(2 * nx - 1) * fac2(2 * ny - 1) * fac2(2 * nz - 1))
        / (2 * alpha / sp.pi) ** sp.Rational(3, 2)
        / (4 * alpha) ** (nx + ny + nz)
    )


def get_pure_l2_norm(alpha, ell):
    """Compute the norm of a pure gaussian primitive."""
    return sp.sqrt(
        int(fac2(2 * ell - 1)) / (2 * alpha / sp.pi) ** sp.Rational(3, 2) / (4 * alpha) ** ell
    )


def include_l2_norm(tfs):
    """Correct the transformation matrices to work for L2 normalized functions."""
    for ell, tf in enumerate(tfs):
        # Multiply each columbn with the norm of a Cartesian function, to go
        # from normalized to unnormalized Cartesian functions, after which the
        # transform is applied. The choice of exponent is irrelevant, as long
        # as it is consistent
        for i, (nx, ny, nz) in enumerate(iter_mononomials(ell)):
            tf[:, i] *= get_cart_l2_norm(1, nx, ny, nz)
        # Divide everything by the norm of a pure function, to get results
        # for normalized pure functions
        tf[:] /= get_pure_l2_norm(1, ell)
        # Run simplify on each element
        for i in range(tf.size):
            tf.flat[i] = sp.simplify(tf.flat[i])


def print_latex(tfs):
    """Print transformation matrices in Latex code."""

    def iter_pure_labels(ell):
        """Iterate over labels for pure functions."""
        yield f"C_{{{ell}0}}"
        for m in range(1, ell + 1):
            yield f"C_{{{ell}{m}}}"
            yield f"S_{{{ell}{m}}}"

    def tostr(v):
        """Format an sympy expression as Latex code."""
        if v == 0:
            return r"\cdot"
        return sp.latex(v)

    for ell, tf in enumerate(tfs):
        npure, ncart = tf.shape
        print(r"\left(\begin{array}{c}")
        print("   ", r" \\ ".join([f"b({label})" for label in iter_pure_labels(ell)]))
        print(r"\end{array}\right)")
        print("    &=")
        print(r"\left(\begin{array}{" + ("c" * ncart) + "}")
        for ipure in range(npure):
            print(
                "   ",
                r" & ".join([tostr(tf[ipure, icart]) for icart in range(ncart)]),
                r"\\",
            )
        print(r"\end{array}\right)")
        print(r"\left(\begin{array}{c}")
        els = []
        for nx, ny, nz in iter_mononomials(ell):
            spoly = "x" * nx + "y" * ny + "z" * nz
            if spoly == "":
                spoly = "1"
            els.append(f"b({spoly})")
        print("   ", r" \\ ".join(els))
        print(r"\end{array}\right)")
        if ell != len(tfs) - 1:
            print(r"\\")


def print_python(tfs):
    """Print transformation matrices in Python code."""

    def tostr(v):
        """Format an sympy expression as a float with sufficient digits."""
        s = repr(v.evalf(17))
        while s[-1] == "0" and "." in s and s[-2] != ".":
            s = s[:-1]
        return s

    for ell, tf in enumerate(tfs):
        npure, ncart = tf.shape
        print(f"tf{ell} = np.array([")
        for ipure in range(npure):
            print(
                "    [{}],".format(", ".join([tostr(tf[ipure, icart]) for icart in range(ncart)]))
            )
        print("])")
    print("tfs = [{}]".format(", ".join(f"tf{ell}" for ell in range(len(tfs)))))


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


if __name__ == "__main__":
    main()
