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
"""Gaussian/GAMESS-US WFN file format.

Only use this format if the program that generated it does not offer any alternatives that
HORTON can load. The WFN format has the disadvantage that it cannot represent contractions
and therefore expands all orbitals into a decontracted basis. This makes the
post-processing less efficient compared to formats that do support contractions of
Gaussian functions.
"""

import functools
import operator
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from ..basis import MolecularBasis, Shell, convert_conventions
from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..overlap import gob_cart_normalization
from ..periodic import num2sym, sym2num
from ..utils import LineIterator, LoadError, PrepareDumpError

__all__ = []


PATTERNS = ["*.wfn"]


# From the AIMALL documentation
# 1 S
# 2 PX
# 3 PY
# 4 PZ
# 5 DXX
# 6 DYY
# 7 DZZ
# 8 DXY
# 9 DXZ
# 10 DYZ
# 11 FXXX
# 12 FYYY
# 13 FZZZ
# 14 FXXY
# 15 FXXZ
# 16 FYYZ
# 17 FXYY
# 18 FXZZ
# 19 FYZZ
# 20 FXYZ
# 21 GXXXX
# 22 GYYYY
# 23 GZZZZ
# 24 GXXXY
# 25 GXXXZ
# 26 GXYYY
# 27 GYYYZ
# 28 GXZZZ
# 29 GYZZZ
# 30 GXXYY
# 31 GXXZZ
# 32 GYYZZ
# 33 GXXYZ
# 34 GXYYZ
# 35 GXYZZ
# 36 HZZZZZ (005)
# 37 HYZZZZ (014)
# 38 HYYZZZ (023)
# 39 HYYYZZ (032)
# 40 HYYYYZ (041)
# 41 HYYYYY (050)
# 42 HXZZZZ (104)
# 43 HXYZZZ (113)
# 44 HXYYZZ (122)
# 45 HXYYYZ (131)
# 46 HXYYYY (140)
# 47 HXXZZZ (203)
# 48 HXXYZZ (212)
# 49 HXXYYZ (221)
# 50 HXXYYY (230)
# 51 HXXXZZ (302)
# 52 HXXXYZ (311)
# 53 HXXXYY (320)
# 54 HXXXXZ (401)
# 55 HXXXXY (410)
# 56 HXXXXX (500)


# fmt: off
CONVENTIONS = {
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyz', 'xyy', 'xzz', 'yzz', 'xyz'],
    (4, 'c'): ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz',
               'yzzz', 'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz'],
    (5, 'c'): ['zzzzz', 'yzzzz', 'yyzzz', 'yyyzz', 'yyyyz', 'yyyyy', 'xzzzz',
               'xyzzz', 'xyyzz', 'xyyyz', 'xyyyy', 'xxzzz', 'xxyzz', 'xxyyz',
               'xxyyy', 'xxxzz', 'xxxyz', 'xxxyy', 'xxxxz', 'xxxxy', 'xxxxx'],
}
# fmt: on

# Definition of primitives in the WFN format. This is the order of the primitive
# types as documented by aimall, used in the field TYPE ASSIGNMENTS.
PRIMITIVE_NAMES = functools.reduce(
    operator.iadd, [CONVENTIONS[(angmom, "c")] for angmom in range(6)], []
)


def _load_helper_num(lit: LineIterator) -> list[int]:
    """Read number of orbitals, primitives and atoms."""
    line = next(lit)
    if not line.startswith("G"):
        raise LoadError("Expecting line to start with 'G'.", lit)
    # FORMAT (16X,I7,13X,I7,11X,I9)
    num_mo = int(line[16:23])
    nprim = int(line[36:43])
    num_atoms = int(line[54:63])
    return num_mo, nprim, num_atoms


def _load_helper_atoms(lit: LineIterator, num_atoms: int) -> tuple[NDArray[int], NDArray[float]]:
    """Read the coordinates of the atoms."""
    atnums = np.empty(num_atoms, int)
    atcoords = np.empty((num_atoms, 3), float)
    for atom in range(num_atoms):
        line = next(lit)
        # FORMAT (A8,16X,3F12.8)
        symbol = line[:8].strip().title()[:2]
        atcoords[atom, 0] = float(line[24:36])
        atcoords[atom, 1] = float(line[36:48])
        atcoords[atom, 2] = float(line[48:60])
        atnum = sym2num.get(symbol)
        # WFN files created with AIMAll have the symbol and index combined in one word.
        if atnum is None:
            atnum = sym2num[symbol[0]]
        atnums[atom] = atnum
    return atnums, atcoords


def _load_helper_section(
    lit: LineIterator, n: int, start: str, skip: int, step: int, dtype: np.dtype
) -> NDArray:
    """Read CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS sections."""
    section = []
    while len(section) < n:
        line = next(lit)
        if not line.startswith(start):
            raise LoadError(f"Expecting line to start with '{start}'.", lit)
        line = line[skip:]
        while len(line) >= step:
            section.append(dtype(line[:step].replace("D", "E")))
            line = line[step:]
    if len(section) != n:
        raise LoadError("Number of elements in section do not match 'n'.", lit)
    return np.array(section, dtype=dtype)


def _load_helper_mo(lit: LineIterator, nprim: int) -> tuple[int, float, float, NDArray[float]]:
    """Read one section of MO information."""
    line = next(lit)
    if not line.startswith("MO"):
        raise LoadError("Expecting line to start with 'MO'.", lit)
    # FORMAT (2X,I5,27X,F13.7,15X,F12.6)
    number = int(line[2:7])
    occ = float(line[34:47])
    energy = float(line[62:74])
    # FORMAT (5E16.8)
    coeffs = _load_helper_section(lit, nprim, "", 0, 16, float)
    return number, occ, energy, coeffs


def _load_helper_energy(lit: LineIterator) -> float:
    """Read energy."""
    line = next(lit)
    while "ENERGY" not in line and line is not None:
        line = next(lit)
    # FORMAT (17X,F20.12)
    # Note: this differs between *.WFN files -- in some files the energy field ends at
    # column 35 instead of column 37 -- so use split() to extract energy
    energy = float(line[17:37].split()[0])
    virial = float(line[55:68])
    return energy, virial


def _load_helper_multiwfn(lit: LineIterator, num_mo: int) -> NDArray[int]:
    """Read MO spin information from MULTIWFN extension."""
    for line in lit:
        if "$MOSPIN $END" in line:
            # FORMAT (40I2)
            return _load_helper_section(lit, num_mo, "", 0, 2, int)
    return np.empty((0,), dtype=int)


def load_wfn_low(lit: LineIterator) -> tuple:
    """Load data from a WFN file into arrays.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    """
    # read sections of wfn file
    title = next(lit).strip()
    num_mo, nprim, num_atoms = _load_helper_num(lit)
    atnums, atcoords = _load_helper_atoms(lit, num_atoms)
    # centers are indexed from zero in HORTON
    icenters = _load_helper_section(lit, nprim, "CENTRE ASSIGNMENTS", 20, 3, int) - 1
    # The type assignments are integer indices for individual basis functions,
    # while in IOData, only the order within shells is fixed by configurable
    # conventions. In principle, the wfn format makes it possible for two
    # shells with the same angular momentum to have a different ordering of
    # the basis functions.
    type_assignments = _load_helper_section(lit, nprim, "TYPE ASSIGNMENTS", 20, 3, int) - 1
    exponent = _load_helper_section(lit, nprim, "EXPONENTS", 10, 14, float)
    mo_numbers = np.empty(num_mo, int)
    mo_occs = np.empty(num_mo, float)
    mo_energies = np.empty(num_mo, float)
    mo_coeffs = np.empty([nprim, num_mo], float)
    for mo in range(num_mo):
        mo_numbers[mo], mo_occs[mo], mo_energies[mo], mo_coeffs[:, mo] = _load_helper_mo(lit, nprim)
    energy, virial = _load_helper_energy(lit)
    mo_spin = _load_helper_multiwfn(lit, num_mo)
    return (
        title,
        atnums,
        atcoords,
        icenters,
        type_assignments,
        exponent,
        mo_numbers,
        mo_occs,
        mo_energies,
        mo_coeffs,
        energy,
        virial,
        mo_spin,
    )


def build_obasis(
    icenters: NDArray[int],
    type_assignments: NDArray[int],
    exponents: NDArray[float],
    lit: LineIterator,
) -> tuple[MolecularBasis, NDArray[int]]:
    """Construct a basis set using the arrays read from a WFN or WFX file.

    Parameters
    ----------
    icenters
        The center indices for all basis functions. shape=(nbasis,). Lowest
        index is zero.
    type_assignments
        Integer codes for basis function names. shape=(nbasis,). Lowest index
        is zero.
    exponents
        The Gaussian exponents of all basis functions. shape=(nbasis,)

    """
    # Build the basis set, keeping track of permutations in case there are
    # deviations from the default ordering of primitives in a WFN file.
    shells = []
    ibasis = 0
    nbasis = len(icenters)
    permutation = np.zeros(nbasis, dtype=int)
    # Loop over all (batches of primitive) basis functions and extract shells.
    while ibasis < nbasis:
        # Determine the angular moment of the shell
        type_assignment = type_assignments[ibasis]
        # multiple different type assignments (codes for individual basis
        # functions) can match one angular momentum.
        angmom = 0 if type_assignment == 0 else len(PRIMITIVE_NAMES[type_assignments[ibasis]])
        # The number of cartesian functions for the current angular momentum
        ncart = len(CONVENTIONS[(angmom, "c")])
        # Determine how many shells are to be read in one batch. E.g. for a
        # contracted p shell, the WFN format contains first all px basis
        # functions, the all py, finally all pz. These need to be regrouped into
        # shells.
        # This pattern can almost be used to reverse-engineer contractions.
        # One should also check (i) if the corresponding mo-coefficients are the
        # same (after fixing them for normalization) and (ii) if the functions
        # are centered on the same atom.
        # For now, this implementation makes no attempt to reverse-engineer
        # contractions, but it can be done.
        ncon = 1  # the contraction length
        if angmom > 0:
            # batches for s-type functions are not necessary and may result in
            # multiple centers being pulled into one batch.
            while (
                ibasis + ncon < len(type_assignments)
                and type_assignments[ibasis + ncon] == type_assignment
            ):
                ncon += 1
        # Check if the type assignment is consistent for remaining basis
        # functions in this batch.
        for ifn in range(ncart):
            if not (
                type_assignments[ibasis + ncon * ifn : ibasis + ncon * (ifn + 1)]
                == type_assignments[ibasis + ncon * ifn]
            ).all():
                raise LoadError("Inconcsistent type assignments in current batch of shells.", lit)
        # Check if all basis functions in the current batch sit on
        # the same center. If not, IOData cannot read this file.
        icenter = icenters[ibasis]
        if not (icenters[ibasis : ibasis + ncon * ncart] == icenter).all():
            raise LoadError("Incomplete shells in WFN file not supported by IOData.", lit)
        # Check if the same exponent is used for corresponding basis functions.
        batch_exponents = exponents[ibasis : ibasis + ncon]
        for ifn in range(ncart):
            if not (
                exponents[ibasis + ncon * ifn : ibasis + ncon * (ifn + 1)] == batch_exponents
            ).all():
                raise LoadError(
                    "Exponents must be the same for corresponding basis functions.", lit
                )
        # A permutation is needed because we need to regroup basis functions
        # into shells.
        batch_primitive_names = [
            PRIMITIVE_NAMES[type_assignments[ibasis + ifn * ncon]] for ifn in range(ncart)
        ]
        for irep in range(ncon):
            for i, primitive_name in enumerate(batch_primitive_names):
                ifn = CONVENTIONS[(angmom, "c")].index(primitive_name)
                permutation[ibasis + irep * ncart + ifn] = ibasis + irep + i * ncon
        # WFN uses non-normalized primitives, which will be corrected for
        # when processing the MO coefficients. Normalized primitives will
        # be used here. No attempt is made here to reconstruct the contraction.
        shells.extend(
            Shell(icenter, [angmom], ["c"], np.array([exponent]), np.array([[1.0]]))
            for exponent in batch_exponents
        )
        # Move on to the next contraction
        ibasis += ncart * ncon
    obasis = MolecularBasis(shells, CONVENTIONS, "L2")
    assert obasis.nbasis == nbasis
    return obasis, permutation


def get_mocoeff_scales(obasis: MolecularBasis) -> NDArray[float]:
    """Get the L2-normalization of the un-normalized Cartesian basis functions.

    Parameters
    ----------
    obasis
        The molecular orbital basis.

    Returns
    -------
    Scaling factors to be multiplied into the molecular orbital coefficients.

    """
    scales = []
    for shell in obasis.shells:
        angmom = shell.angmoms[0]
        for name in obasis.conventions[(angmom, "c")]:
            if name == "1":
                nx, ny, nz = 0, 0, 0
            else:
                nx = name.count("x")
                ny = name.count("y")
                nz = name.count("z")
            scales.append(gob_cart_normalization(shell.exponents[0], np.array([nx, ny, nz])))
    return np.array(scales)


@document_load_one("WFN", ["atcoords", "atnums", "energy", "mo", "obasis", "title", "extra"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    (
        title,
        atnums,
        atcoords,
        icenters,
        type_assignments,
        exponents,
        mo_numbers,
        mo_occs,
        mo_energies,
        mo_coeffs,
        energy,
        virial,
        mo_spin,
    ) = load_wfn_low(lit)
    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(icenters, type_assignments, exponents, lit)
    # Extra dict to return.
    extra = {"virial_ratio": virial}
    # ----------------------------
    # Build the molecular orbitals
    # ----------------------------
    # Re-order the mo coefficients.
    mo_coeffs = mo_coeffs[permutation]
    # Fix normalization.
    mo_coeffs /= get_mocoeff_scales(obasis).reshape(-1, 1)
    norb = mo_coeffs.shape[1]
    # Determine norb_a,norb_b,norb_ab from mo_spin information.
    if mo_spin.size:
        norb_a = np.sum(mo_spin == 1)
        norb_b = np.sum(mo_spin == 2)
        norb_ab = np.sum(mo_spin == 3)
        if norb_a + norb_b + norb_ab != norb or (norb_b and norb_ab):
            raise LoadError("Invalid orbital spin types.", lit)
        extra["mo_spin"] = mo_spin
    # Determine norb_a,norb_b,norb_ab for restricted wave function by heuristic.
    elif mo_occs.max() > 1.0:
        norb_a = 0
        norb_b = 0
        norb_ab = norb
    # Determine norb_a,norb_b,norb_ab for unrestricted wave function by heuristic.
    # This may still fail in some corner cases.
    else:
        norb_a = 1
        while (
            norb_a < mo_coeffs.shape[1]
            and mo_energies[norb_a] >= mo_energies[norb_a - 1]
            and mo_occs[norb_a] <= mo_occs[norb_a - 1]
            and mo_numbers[norb_a] == mo_numbers[norb_a - 1] + 1
        ):
            norb_a += 1
        norb_b = norb - norb_a
        norb_ab = 0
    # Make wavefunction.
    if norb_ab:
        # Restricted wavefunction.
        mo = MolecularOrbitals(
            "restricted", norb_a + norb_ab, norb_a + norb_ab, mo_occs, mo_coeffs, mo_energies
        )
    else:
        # Unrestricted wavefunction.
        mo = MolecularOrbitals("unrestricted", norb_a, norb_b, mo_occs, mo_coeffs, mo_energies)
    return {
        "title": title,
        "atcoords": atcoords,
        "atnums": atnums,
        "obasis": obasis,
        "mo": mo,
        "energy": energy,
        "extra": extra,
    }


def _format_helper_section(header: str, skip: int, spec: str, nline: int) -> tuple[str, int]:
    """Return a format string for CENTRE_ASSIGMENTS, TYPE_ASSIGNMENTS, EXPONENTS lines."""
    return f"{header[:skip].ljust(skip)}{spec * nline}", len(spec)


def _dump_helper_section(f: TextIO, data: NDArray, fmt: str, skip: int, step: int, nline: int):
    """Write a CENTRE_ASSIGNMENTS, TYPE_ASSIGNMENTS, or EXPONENTS section to file ``f``."""
    while len(data) > 0:
        chunk = data[:nline]
        n_chunk = len(chunk)
        print(fmt[: skip + n_chunk * step].format(*chunk), file=f)
        data = data[n_chunk:]


# FORMAT (16X,I7,13X,I7,11X,I9)
FMT_NUM = "GAUSSIAN        {0:7d} MOL ORBITALS{1:7d} PRIMITIVES{2:9d} NUCLEI"
# FORMAT (2X,A3,I3,11X,I3,2X,3F12.8,10X,F5.1)
FMT_ATM = "  {0:3s}{1:3d}    (CENTRE{2:3d}) {3:12.8f}{4:12.8f}{5:12.8f}  CHARGE ={6:5.1f}"
# FORMAT (2X,5I,8X,F3.1,16X,F13.7,15X,F12.6)
FMT_MOS = "MO{0:5d}     MO {1:3.1f}        OCC NO ={2:13.7f}  ORB. ENERGY ={3:12.6f}"
# FORMAT (17X,F20.12,18X,F13.8)
FMT_ENERGY = " TOTAL ENERGY =  {0:20.12f} THE VIRIAL(-V/T)={1:13.8f}"
# FORMAT (20X,20I3)
FMT_CNTR, STEP_CNTR = _format_helper_section("CENTRE ASSIGNMENTS", 20, "{:3d}", 20)
# FORMAT (20X,20I3)
FMT_TYPE, STEP_TYPE = _format_helper_section("TYPE ASSIGNMENTS", 20, "{:3d}", 20)
# FORMAT (10X,5E14.7)
FMT_EXPN, STEP_EXPN = _format_helper_section("EXPONENTS", 10, "{:14.7E}", 5)
# FORMAT (5E16.8)
FMT_COEF, STEP_COEF = _format_helper_section("", 0, "{:16.8E}", 5)
# FORMAT (40I2)
FMT_SPIN, STEP_SPIN = _format_helper_section("", 0, "{:2d}", 40)
# Default .WFN title
DEFAULT_WFN_TTL = "WFN auto-generated by IOData"


def prepare_dump(filename: str, data: IOData):
    """Check the compatibility of the IOData object with the WFN format.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    data
        The IOData instance to be checked.
    """
    if data.mo is None:
        raise PrepareDumpError("The WFN format requires molecular orbitals", filename)
    if data.obasis is None:
        raise PrepareDumpError("The WFN format requires an orbital basis set", filename)
    if data.mo.kind == "generalized":
        raise PrepareDumpError("Cannot write WFN file with generalized orbitals.", filename)
    if data.mo.occs_aminusb is not None:
        raise PrepareDumpError("Cannot write WFN file when mo.occs_aminusb is set.", filename)
    for shell in data.obasis.shells:
        if any(kind != "c" for kind in shell.kinds):
            raise PrepareDumpError(
                "The WFN format only supports Cartesian MolecularBasis.", filename
            )


@document_dump_one(
    "WFN",
    ["atcoords", "atnums", "mo", "obasis"],
    ["energy", "title", "extra"],
)
def dump_one(f: TextIO, data: IOData) -> None:
    """Do not edit this docstring. It will be overwritten."""
    # get shells for the de-contracted basis
    shells = []
    for shell in data.obasis.shells:
        for i, (angmom, kind) in enumerate(zip(shell.angmoms, shell.kinds)):
            for exponent, coeff in zip(shell.exponents, shell.coeffs.T[i]):
                shells.append(
                    Shell(
                        shell.icenter, [angmom], [kind], np.array([exponent]), coeff.reshape(-1, 1)
                    )
                )
    # make a new instance of MolecularBasis with de-contracted basis shells; ideally for WFN we
    # want the primitive basis set, but IOData only supports shells.
    obasis = MolecularBasis(shells, data.obasis.conventions, data.obasis.primitive_normalization)
    # expand mo.coeffs in the new basis by repeating de-contracted basis coefficients
    permutation, signs = convert_conventions(data.obasis, CONVENTIONS)
    raw_coeffs = data.mo.coeffs[permutation] * signs.reshape(-1, 1)
    mo_coeffs = np.zeros((obasis.nbasis, data.mo.norb))
    index_mo_old, index_mo_new = 0, 0
    # loop over the shells of the old basis
    for shell in data.obasis.shells:
        for angmom, kind in zip(shell.angmoms, shell.kinds):
            n = len(data.obasis.conventions[angmom, kind])
            c = raw_coeffs[index_mo_old : index_mo_old + n]
            for _ in range(shell.nprim):
                mo_coeffs[index_mo_new : index_mo_new + n] = c
                index_mo_new += n
            index_mo_old += n

    # fix MO coefficients
    # 1) expansion coefficients in WFN correspond to un-normalized primitives, so the primitive
    # normalization constants should be included in the MO coefficients. However, IOData stores
    # normalized primitives (either L2 or L1 as recorded in MolecularBasis primitive types), so
    # we need to multiply the MO coefficients by the primitive normalization constants
    scales = get_mocoeff_scales(obasis)
    # 2) expansion coefficients in WFN represent the primitive basis coefficients, so contraction
    # coefficients needs to be multiplied by the MO expansion coefficients.
    contractions = []
    for shell in obasis.shells:
        contractions.extend(np.repeat(shell.coeffs.ravel(), [shell.nbasis], axis=0))
    contractions = np.array(contractions)

    # update MO coefficients to include primitives contraction coefficients & normalization
    for index in range(mo_coeffs.shape[1]):
        mo_coeffs[:, index] *= contractions * scales

    # Extract centre, type assignments and exponents from new obasis
    cntrs = [shell.icenter + 1 for shell in obasis.shells for _ in range(shell.nbasis)]
    expns = [shell.exponents[0] for shell in obasis.shells for _ in range(shell.nbasis)]
    angmom_prim = {}
    count = 1
    for angmom in range(max([shell.angmoms[0] for shell in obasis.shells]) + 1):
        angmom_prim[angmom] = [count + i for i in range(len(obasis.conventions[angmom, "c"]))]
        count += len(obasis.conventions[angmom, "c"])
    types = [item for shell in obasis.shells for item in angmom_prim[shell.angmoms[0]]]

    # Write header (title, # MOs, # primitives, # atoms)
    print(f" {data.title if data.title else DEFAULT_WFN_TTL}", file=f)
    print(FMT_NUM.format(data.mo.norb, obasis.nbasis, data.natom), file=f)

    # Write atoms (symbol, atom #, centre #, x pos., y pos., z pos., charge)
    for iatom, (n, (x, y, z)) in enumerate(zip(data.atnums, data.atcoords)):
        print(FMT_ATM.format(num2sym[n], iatom + 1, iatom + 1, x, y, z, n), file=f)

    # Write centre assignments, type assignments, exponents sections
    _dump_helper_section(f, cntrs, FMT_CNTR, 20, STEP_CNTR, 20)
    _dump_helper_section(f, types, FMT_TYPE, 20, STEP_TYPE, 20)
    _dump_helper_section(f, expns, FMT_EXPN, 10, STEP_EXPN, 5)

    # Write MOs (mo #, occ #, energy)
    mo_iter = enumerate(zip(data.mo.occs, data.mo.energies, mo_coeffs.transpose()))
    for iorb, (occ, energy, coeffs) in mo_iter:
        print(FMT_MOS.format(iorb + 1, 0, occ, energy), file=f)
        # Write ``iorb``th coefficients section
        _dump_helper_section(f, coeffs, FMT_COEF, 0, STEP_COEF, 5)

    # Write energy and virial coefficient
    print("END DATA", file=f)
    print(FMT_ENERGY.format(data.energy or np.nan, data.extra.get("virial_ratio", np.nan)), file=f)

    # Write MOSPIN extension section (optional)
    if data.extra.get("mo_spin") is not None:
        print(" $MOSPIN $END\n\n", file=f)
        _dump_helper_section(f, data.extra["mo_spin"], FMT_SPIN, 0, STEP_SPIN, 40)
