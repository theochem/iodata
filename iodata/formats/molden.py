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
"""Molden file format.

Many QC codes can write out Molden files, e.g. `Molpro <https://www.molpro.net/>`_,
`Orca <https://sites.google.com/site/orcainputlibrary/>`_, `PSI4 <http://www.psicode.org/>`_,
`Molden <http://www.cmbi.ru.nl/molden/>`_, `Turbomole <http://www.turbomole.com/>`_. Keep
in mind that several of these write incorrect versions of the file format, but these
errors are corrected when loading them with IOData.
"""

import copy
from typing import TextIO, Union
from warnings import warn

import attrs
import numpy as np
from numpy.typing import NDArray

from ..basis import (
    HORTON2_CONVENTIONS,
    MolecularBasis,
    Shell,
    angmom_its,
    angmom_sti,
    convert_conventions,
)
from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..overlap import compute_overlap, gob_cart_normalization
from ..periodic import num2sym, sym2num
from ..utils import DumpError, LineIterator, LoadError, LoadWarning, PrepareDumpError, angstrom

__all__ = []


PATTERNS = ["*.molden.input", "*.molden"]

# From the Molden format documentation:
#    5D: D 0, D+1, D-1, D+2, D-2
#    6D: xx, yy, zz, xy, xz, yz
#
#    7F: F 0, F+1, F-1, F+2, F-2, F+3, F-3
#   10F: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
#
#    9G: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
#   15G: xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy,
#        xxyy xxzz yyzz xxyz yyxz zzxy

CONVENTIONS = {
    (0, "c"): ["1"],
    (1, "c"): ["x", "y", "z"],
    (2, "p"): HORTON2_CONVENTIONS[(2, "p")],
    (2, "c"): ["xx", "yy", "zz", "xy", "xz", "yz"],
    (3, "p"): HORTON2_CONVENTIONS[(3, "p")],
    (3, "c"): ["xxx", "yyy", "zzz", "xyy", "xxy", "xxz", "xzz", "yzz", "yyz", "xyz"],
    (4, "p"): HORTON2_CONVENTIONS[(4, "p")],
    (4, "c"): [
        "xxxx",
        "yyyy",
        "zzzz",
        "xxxy",
        "xxxz",
        "xyyy",
        "yyyz",
        "xzzz",
        "yzzz",
        "xxyy",
        "xxzz",
        "yyzz",
        "xxyz",
        "xyyz",
        "xyzz",
    ],
    # H fubnctions are not officially supported by the Molden format but PSI4
    # and ORCA write out such files anyway.
    (5, "p"): HORTON2_CONVENTIONS[(5, "p")],
}


@document_load_one(
    "Molden",
    ["atcoords", "atnums", "atcorenums", "mo", "obasis"],
    ["title"],
    {
        "norm_threshold": "When the normalization of one of the orbitals exceeds "
        "norm_threshold, a correction is attempted or an error "
        "is raised when no suitable correction can be found."
    },
)
def load_one(lit: LineIterator, norm_threshold: float = 1e-4) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    result = _load_low(lit)
    _fix_molden_from_buggy_codes(result, lit, norm_threshold)
    return result


def _load_low(lit: LineIterator) -> dict:
    """Load data from a MOLDEN input file format, without trying to fix errors.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        output dictionary containing ``atcoords``, ``atnums``, ``atcorenums``,
        ``obasis``, ``mo`` & ``signs`` keys and corresponding values. It may contain
        ``title`` key and its corresponding value as well.

    """
    pure_angmoms = set()
    atnums = None
    atcoords = None
    obasis = None
    coeffsa = None
    energiesa = None
    occsa = None
    coeffsb = None
    energiesb = None
    occsb = None
    title = None

    line = next(lit)
    if line.strip() != "[Molden Format]":
        raise LoadError("Molden header not found.", lit)
    # The order of sections, denoted by "[...]", is not fixed in the Molden
    # format, so we need a loop that checks for all possible sections at
    # each iteration. If needed, the contents of the section is read.
    while True:
        try:
            line = next(lit).lower().strip()
        except StopIteration:
            # This means we continue reading till the end of the file.
            # There is no real way to know when a Molden file has ended, other
            # than reaching the end of the file.
            break
        # settings for pure or Cartesian shells.
        if line.startswith(("[5d]", "[5d7f]")):
            pure_angmoms.add(2)
            pure_angmoms.add(3)
        elif line.lower().startswith("[7f]"):
            pure_angmoms.add(3)
        elif line.lower().startswith("[5d10f]"):
            pure_angmoms.add(2)
        elif line.lower().startswith("[9g]"):
            pure_angmoms.add(4)
            # H functions are not part of the Molden standard but the
            # following line is compatible with files containing H functions
            # writen by PSI4 and ORCA.
            pure_angmoms.add(5)
        # title
        elif line == "[title]":
            title = next(lit).strip()
        # atoms
        elif line.startswith("[atoms]"):
            if "au" in line:
                cunit = 1.0
            elif "angs" in line:
                cunit = angstrom
            atnums, atcorenums, atcoords = _load_helper_atoms(lit, cunit)
        # we only support Gaussian-type orbitals (gto's)
        elif line == "[gto]":
            obasis = _load_helper_obasis(lit)
        elif line == "[sto]":
            raise LoadError("Slater-type orbitals are not supported by IODATA.", lit)
        # molecular-orbital coefficients.
        elif line == "[mo]":
            data_alpha, data_beta = _load_helper_coeffs(lit)
            occsa, coeffsa, energiesa, irrepsa = data_alpha
            occsb, coeffsb, energiesb, irrepsb = data_beta

    # Assign pure and Cartesian correctly. This needs to be done after reading
    # because the tags for pure functions may come after the basis set.
    for shell in obasis.shells:
        # Code only has to work for segmented contractions
        if shell.angmoms[0] in pure_angmoms:
            shell.kinds[0] = "p"

    if coeffsb is None:
        if coeffsa.shape[0] != obasis.nbasis:
            raise LoadError(
                "Number of alpha orbital coefficients does not match the size of the basis.", lit
            )
        mo = MolecularOrbitals(
            "restricted", coeffsa.shape[1], coeffsa.shape[1], occsa, coeffsa, energiesa, irrepsa
        )
    else:
        if coeffsb.shape[0] != obasis.nbasis:
            raise LoadError(
                "Number of beta orbital coefficients does not match the size of the basis.", lit
            )
        mo = MolecularOrbitals(
            "unrestricted",
            coeffsa.shape[1],
            coeffsb.shape[1],
            np.concatenate((occsa, occsb), axis=0),
            np.concatenate((coeffsa, coeffsb), axis=1),
            np.concatenate((energiesa, energiesb), axis=0),
            irrepsa + irrepsb,
        )

    result = {
        "atcoords": atcoords,
        "atnums": atnums,
        "obasis": obasis,
        "mo": mo,
        "atcorenums": atcorenums,
    }
    if title is not None:
        result["title"] = title
    return result


def _load_helper_atoms(
    lit: LineIterator, cunit: float
) -> tuple[NDArray[int], NDArray[float], NDArray[float]]:
    """Load element numbers and coordinates."""
    atnums = []
    atcorenums = []
    atcoords = []
    for line in lit:
        if line.strip() == "":
            break
        words = line.split()
        if len(words) != 6:
            # Go back to previous line and stop
            lit.back(line)
            break
        atnums.append(sym2num[words[0].title()])
        atcorenums.append(float(words[2]))
        atcoords.append([float(words[3]), float(words[4]), float(words[5])])
    atnums = np.array(atnums, int)
    atcorenums = np.array(atcorenums)
    atcoords = np.array(atcoords) * cunit
    return atnums, atcorenums, atcoords


def _load_helper_obasis(lit: LineIterator) -> MolecularBasis:
    """Load the orbital basis."""
    shells = []
    while True:
        line = next(lit)
        words = line.split()
        # Normally a new atom section begins with one or two integers,
        # of which the second is zero if present. If not, we are done
        # and have to push one line back.
        if not (words and words[0].isdigit()):
            lit.back(line)
            break
        icenter = int(words[0]) - 1
        # Loop over all shells until reaching an empty line
        while True:
            words = next(lit).split()
            if not words:
                break
            # Read a new shell
            angmom = angmom_sti(words[0])
            nprim = int(words[1])
            exponents = np.zeros(nprim)
            coeffs = np.zeros((nprim, 1))
            for iprim in range(nprim):
                words = next(lit).split()
                exponents[iprim] = float(words[0].replace("D", "E"))
                coeffs[iprim, 0] = float(words[1].replace("D", "E"))
            # Unless changed later, all shells are assumed to be Cartesian.
            shells.append(Shell(icenter, [angmom], ["c"], exponents, coeffs))
    return MolecularBasis(shells, CONVENTIONS, "L2")


def _load_helper_coeffs(lit: LineIterator) -> tuple:
    """Load the orbital coefficients."""
    occsa = []
    coeffsa = []
    energiesa = []
    irrepsa = []
    occsb = []
    coeffsb = []
    energiesb = []
    irrepsb = []

    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            # We have no proper way to check when a Molden file has ended, so
            # we must anticipate for it here.
            break
        # An empty line means we are done
        if line.strip() == "":
            break
        # An bracket also means we are done and a new section has started.
        # Other parts of the parser may need this section line, so we push it
        # back.
        if "[" in line:
            lit.back(line)
            break
        # prepare array with orbital coefficients
        info = {}
        lit.back(line)
        for line in lit:
            if line.count("=") != 1:
                lit.back(line)
                break
            key, value = line.split("=")
            info[key.strip().lower()] = value
        occ = float(info["occup"])
        col = []
        energy = float(info["ene"])
        irrep = info.get("sym", "??").strip()
        # store column of coefficients, i.e. one orbital, energy and occ
        if info["spin"].strip().lower() == "alpha":
            occsa.append(occ)
            coeffsa.append(col)
            energiesa.append(energy)
            irrepsa.append(irrep)
        else:
            occsb.append(occ)
            coeffsb.append(col)
            energiesb.append(energy)
            irrepsb.append(irrep)
        for line in lit:
            words = line.split()
            if len(words) != 2 or not words[0].isdigit():
                # The line does not look like an index with an orbital coefficient.
                # Time to stop and put the line back
                lit.back(line)
                break
            col.append(float(words[1]))

    coeffsa = np.array(coeffsa).T
    energiesa = np.array(energiesa)
    occsa = np.array(occsa)
    if not coeffsb:
        coeffsb = None
        energiesb = None
        occsb = None
    else:
        coeffsb = np.array(coeffsb).T
        energiesb = np.array(energiesb)
        occsb = np.array(occsb)
    return (occsa, coeffsa, energiesa, irrepsa), (occsb, coeffsb, energiesb, irrepsb)


def _is_normalized_properly(
    obasis: MolecularBasis,
    atcoords: NDArray[float],
    orb_alpha: NDArray[float],
    orb_beta: NDArray[float],
    norm_threshold: float = 1e-4,
) -> bool:
    """Test the normalization of the occupied and virtual orbitals.

    Parameters
    ----------
    obasis
        A dictionary containing the parameters of the GOBasis class.
    atcoords
        The atomic Cartesian coordinates, shape = (natom, 3).
    orb_alpha
        The alpha orbitals coefficients
    orb_beta
        The beta orbitals (may be None).
    norm_threshold
        When the error on one of the orbitals norm exceeds norm_threshold,
        the function returns False. True is returned otherwise.

    """
    # Compute the overlap matrix. Unfortunately, we have to recalculate it at
    # every attempt because also the primitive normalization may differ in
    # different cases.
    olp = compute_overlap(obasis, atcoords)
    # Convenient code for debugging files coming from crappy QC codes.
    # np.set_printoptions(linewidth=5000, precision=2, suppress=True, threshold=100000)
    # coeffs = orb_alpha._coeffs
    # if permutation is not None:
    #     coeffs = coeffs[permutation].copy()
    # if signs is not None:
    #     coeffs = coeffs*signs.reshape(-1, 1)
    # print np.dot(coeffs.T, np.dot(olp._array, coeffs))
    # print

    # Convert the orbitals to the conventions of the overlap matrix.
    # permutation, signs = convert_conventions(obasis, HORTON2_CONVENTIONS)
    orbs = [orb_alpha]
    if orb_beta is not None:
        orbs.append(orb_beta)
    # Compute the norm of each occupied and virtual orbital. Keep track of
    # the largest deviation from unity
    error_max = 0.0
    for orb in orbs:
        for iorb in range(orb.shape[1]):
            vec = orb[:, iorb].copy()
            norm = np.dot(vec, np.dot(olp, vec))
            # print(iorb, norm)
            error_max = max(error_max, abs(norm - 1))

    # final judgement
    return error_max <= norm_threshold


def _fix_obasis_orca(obasis: MolecularBasis) -> MolecularBasis:
    """Return a new MolecularBasis correcting for errors from ORCA.

    Orca has different normalization conventions for the primitives and also
    different sign conventions for some of the pure functions.
    """
    orca_conventions = {
        (0, "c"): ["1"],
        (1, "c"): ["x", "y", "z"],
        (2, "p"): ["c0", "c1", "s1", "c2", "s2"],
        (2, "c"): ["xx", "yy", "zz", "xy", "xz", "yz"],
        (3, "p"): ["c0", "c1", "s1", "c2", "s2", "-c3", "-s3"],
        (3, "c"): ["xxx", "yyy", "zzz", "xyy", "xxy", "xxz", "xzz", "yzz", "yyz", "xyz"],
        (4, "p"): ["c0", "c1", "s1", "c2", "s2", "-c3", "-s3", "-c4", "-s4"],
        (4, "c"): [
            "xxxx",
            "yyyy",
            "zzzz",
            "xxxy",
            "xxxz",
            "xyyy",
            "yyyz",
            "xzzz",
            "yzzz",
            "xxyy",
            "xxzz",
            "yyzz",
            "xxyz",
            "xyyz",
            "xyzz",
        ],
        # H functions are not officialy supported by Molden, but this is how
        # ORCA writes Molden files anyway:
        (5, "p"): ["c0", "c1", "s1", "c2", "s2", "-c3", "-s3", "-c4", "-s4", "c5", "s5"],
    }
    fixed_shells = []
    for shell in obasis.shells:
        fixed_shell = copy.deepcopy(shell)
        fixed_shells.append(fixed_shell)
        # We can safely assume segmented shells.
        angmom = shell.angmoms[0]
        kind = shell.kinds[0]
        for iprim, exponent in enumerate(shell.exponents):
            # Default 1.0: do not to correct anything, unless we know how to correct.
            correction = 1.0
            if angmom == 0:
                correction = gob_cart_normalization(exponent, np.array([0, 0, 0]))
            elif angmom == 1:
                correction = gob_cart_normalization(exponent, np.array([1, 0, 0]))
            elif angmom == 2 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([1, 1, 0]))
            elif angmom == 3 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([1, 1, 1]))
            elif angmom == 4 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([2, 1, 1]))
            elif angmom == 5 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([5, 0, 0]))
            if correction != 1.0:
                fixed_shell.coeffs[iprim, 0] /= correction
    return MolecularBasis(fixed_shells, orca_conventions, obasis.primitive_normalization)


def _fix_obasis_psi4(obasis: MolecularBasis) -> Union[MolecularBasis, None]:
    """Return a new MolecularBasis correcting for errors from PSI4 <= 1.0.

    Old PSI4 version used a different normalization of the primitives.
    """
    fixed_shells = []
    corrected = False
    for shell in obasis.shells:
        # We can safely assume segmented shells.
        fixed_shell = copy.deepcopy(shell)
        fixed_shells.append(fixed_shell)
        angmom = shell.angmoms[0]
        kind = shell.kinds[0]
        for iprim, exponent in enumerate(shell.exponents):
            # Default 1.0: do not to correct anything, unless we know how to correct.
            correction = 1.0
            if angmom == 0:
                correction = gob_cart_normalization(exponent, np.array([0, 0, 0]))
            elif angmom == 1:
                correction = gob_cart_normalization(exponent, np.array([1, 0, 0]))
            elif angmom == 2 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([1, 1, 0])) / np.sqrt(3.0)
            elif angmom == 3 and kind == "p":
                correction = gob_cart_normalization(exponent, np.array([1, 1, 1])) / np.sqrt(15.0)
            # elif angmom == 4 and kind == 'p': ##  ! Not tested
            #     correction = gob_cart_normalization(exponent, np.array([2, 1, 1]))/np.sqrt(105.0)
            if correction != 1.0:
                corrected = True
                fixed_shell.coeffs[iprim, 0] /= correction
    if corrected:
        return MolecularBasis(fixed_shells, obasis.conventions, obasis.primitive_normalization)
    return None


def _fix_obasis_turbomole(obasis: MolecularBasis) -> Union[MolecularBasis, None]:
    """Return a new MolecularBasis correcting for errors from turbomole.

    Turbomole uses a different normalization of the primitives.
    """
    fixed_shells = []
    corrected = False
    for shell in obasis.shells:
        # We can safely assume segmented shells.
        fixed_shell = copy.deepcopy(shell)
        fixed_shells.append(fixed_shell)
        angmom = shell.angmoms[0]
        kind = shell.kinds[0]
        for iprim in range(shell.nprim):
            # Default 1.0: do not to correct anything, unless we know how to correct.
            correction = 1.0
            if angmom == 2 and kind == "c":
                correction = 1.0 / np.sqrt(3.0)
            elif angmom == 3 and kind == "c":
                correction = 1.0 / np.sqrt(15.0)
            elif angmom == 4 and kind == "c":
                correction = 1.0 / np.sqrt(105.0)
            if correction != 1.0:
                corrected = True
                fixed_shell.coeffs[iprim, 0] /= correction
    if corrected:
        return MolecularBasis(fixed_shells, obasis.conventions, obasis.primitive_normalization)
    return None


def _fix_obasis_normalize_contractions(obasis: MolecularBasis) -> MolecularBasis:
    """Return a basis with normalized contractions.

    Files written by Molden don't need this fix and have properly normalized
    contractions. When Molden reads files in the Molden format, it does
    renormalize the contractions and other programs than Molden may generate
    Molden files with unnormalized contractions. This renormalization is only a
    last resort in IOData. If we would do it up-front, like Molden, we would not
    be able to fix errors in files from ORCA and older PSI4 versions.
    """
    fixed_shells = []
    for shell in obasis.shells:
        shell_obasis = MolecularBasis(
            [attrs.evolve(shell, icenter=0)], obasis.conventions, obasis.primitive_normalization
        )
        # 2) Get the first diagonal element of the overlap matrix
        olpdiag = compute_overlap(shell_obasis, np.zeros((1, 3), float))[0, 0]
        # 3) Normalize the contraction
        fixed_shell = copy.deepcopy(shell)
        fixed_shell.coeffs[:] /= np.sqrt(olpdiag)
        fixed_shells.append(fixed_shell)
    return MolecularBasis(fixed_shells, obasis.conventions, obasis.primitive_normalization)


def _fix_mo_coeffs_psi4(obasis: MolecularBasis) -> Union[MolecularBasis, None]:
    """Return correction values for the MO coefficients.

    PSI4 <= 1.3.2 uses a different normalizationion conventions for Cartesian
    AO basis functions. The coefficients need to be divided by the returned
    correction factor.
    """
    correction = []
    corrected = False
    for shell in obasis.shells:
        # We can safely assume segmented shells.
        assert shell.ncon == 1
        angmom = shell.angmoms[0]
        kind = shell.kinds[0]
        factors = None
        if kind == "c":
            if angmom == 2:
                factors = np.sqrt([1] * 3 + [3] * 3)
            elif angmom == 3:
                factors = np.sqrt([1] * 3 + [5] * 6 + [15])
            elif angmom == 4:
                factors = np.sqrt([1] * 3 + [7] * 6 + [35 / 3] * 3 + [35] * 3)
        if factors is None:
            factors = np.ones(shell.nbasis)
        else:
            assert len(factors) == shell.nbasis
            corrected = True
        correction.append(factors)
    if corrected:
        return np.concatenate(correction)
    return None


def _fix_mo_coeffs_cfour(obasis: MolecularBasis) -> Union[MolecularBasis, None]:
    """Return correction values for the MO coefficients.

    CFOUR (up to current 2.1) uses different normalization conventions for Cartesian
    AO basis functions. The coefficients need to be divided by the returned
    correction factor.
    """
    correction = []
    corrected = False
    for shell in obasis.shells:
        # We can safely assume segmented shells.
        assert shell.ncon == 1
        angmom = shell.angmoms[0]
        kind = shell.kinds[0]
        factors = None
        if kind == "c":
            if angmom == 2:
                factors = np.array([1.0 / np.sqrt(3.0)] * 3 + [1.0] * 3)
            elif angmom == 3:
                factors = np.array([1.0 / np.sqrt(15.0)] * 3 + [1.0 / (np.sqrt(3.0))] * 6 + [1.0])
            elif angmom == 4:
                factors = np.array(
                    [1.0 / np.sqrt(105.0)] * 3
                    + [1.0 / (np.sqrt(15.0))] * 6
                    + [1.0 / 3.0] * 3
                    + [1.0 / (np.sqrt(3.0))] * 3
                )
        if factors is None:
            factors = np.ones(shell.nbasis)
        else:
            assert len(factors) == shell.nbasis
            corrected = True
        correction.append(factors)
    if corrected:
        return np.concatenate(correction)
    return None


def _fix_molden_from_buggy_codes(result: dict, lit: LineIterator, norm_threshold: float = 1e-4):
    """Detect errors in the data loaded from a molden or mkl file and correct.

    This function can recognize erroneous files created by PSI4, ORCA and
    Turbomole. The value `results['obasis']` will be updated accordingly.

    Parameters
    ----------
    result
        A dictionary with the data loaded in the ``load_molden`` function.
    lit
        The line iterator to read the data from, used for warnings.
    norm_threshold
        When the error on one of the orbitals norm exceeds norm_threshold,
        the (corrected) data loaded from the Molden file is considered to be
        incorrect, in which case other corrections are tested or an exception
        is raised when no more corrections can be applied.

    """
    obasis = result["obasis"]
    atcoords = result["atcoords"]
    if result["mo"].kind == "restricted":
        coeffsa = result["mo"].coeffs
        # Skip testing coeffsb if it is the same array as coeffsa.
        coeffsb = None
    elif result["mo"].kind == "unrestricted":
        coeffsa = result["mo"].coeffsa
        coeffsb = result["mo"].coeffsb
    else:
        raise LoadError(f"Molecular orbital kind={result['mo'].kind} not recognized.", lit)

    if _is_normalized_properly(obasis, atcoords, coeffsa, coeffsb, norm_threshold):
        # The file is good. No need to change obasis.
        return

    # --- ORCA
    orca_obasis = _fix_obasis_orca(obasis)
    if _is_normalized_properly(orca_obasis, atcoords, coeffsa, coeffsb, norm_threshold):
        warn(
            LoadWarning("Corrected for typical ORCA errors in Molden/MKL file.", lit.filename),
            stacklevel=2,
        )
        result["obasis"] = orca_obasis
        return

    # --- PSI4 < 1.0
    psi4_obasis = _fix_obasis_psi4(obasis)
    if psi4_obasis is not None and _is_normalized_properly(
        psi4_obasis, atcoords, coeffsa, coeffsb, norm_threshold
    ):
        warn(
            LoadWarning("Corrected for PSI4 < 1.0 errors in Molden/MKL file.", lit.filename),
            stacklevel=2,
        )
        result["obasis"] = psi4_obasis
        return

    # -- Turbomole
    turbom_obasis = _fix_obasis_turbomole(obasis)
    if turbom_obasis is not None and _is_normalized_properly(
        turbom_obasis, atcoords, coeffsa, coeffsb, norm_threshold
    ):
        warn(
            LoadWarning("Corrected for Turbomole errors in Molden/MKL file.", lit.filename),
            stacklevel=2,
        )
        result["obasis"] = turbom_obasis
        return

    # --- CFOUR 2.1
    cfour_coeff_correction = _fix_mo_coeffs_cfour(obasis)
    if cfour_coeff_correction is not None:
        coeffsa_cfour = coeffsa / cfour_coeff_correction[:, np.newaxis]
        coeffsb_cfour = None if coeffsb is None else coeffsb / cfour_coeff_correction[:, np.newaxis]
        if _is_normalized_properly(obasis, atcoords, coeffsa_cfour, coeffsb_cfour, norm_threshold):
            warn(
                LoadWarning("Corrected for CFOUR 2.1 errors in Molden/MKL file.", lit.filename),
                stacklevel=2,
            )
            result["obasis"] = obasis
            if result["mo"].kind == "restricted":
                result["mo"].coeffs[:] = coeffsa_cfour
            else:
                result["mo"].coeffsa[:] = coeffsa_cfour
                result["mo"].coeffsb[:] = coeffsb_cfour
            return

    # --- Renormalized contractions
    normed_obasis = _fix_obasis_normalize_contractions(obasis)
    if _is_normalized_properly(normed_obasis, atcoords, coeffsa, coeffsb, norm_threshold):
        warn(
            LoadWarning(
                "Corrected for unnormalized contractions in Molden/MKL file.", lit.filename
            ),
            stacklevel=2,
        )
        result["obasis"] = normed_obasis
        return

    # --- PSI4 <= 1.3.2
    psi4_coeff_correction = _fix_mo_coeffs_psi4(obasis)
    if psi4_coeff_correction is not None:
        coeffsa_psi4 = coeffsa / psi4_coeff_correction[:, np.newaxis]
        coeffsb_psi4 = None if coeffsb is None else coeffsb / psi4_coeff_correction[:, np.newaxis]
        if _is_normalized_properly(
            normed_obasis, atcoords, coeffsa_psi4, coeffsb_psi4, norm_threshold
        ):
            warn(
                LoadWarning("Corrected for PSI4 <= 1.3.2 errors in Molden/MKL file.", lit.filename),
                stacklevel=2,
            )
            result["obasis"] = normed_obasis
            if result["mo"].kind == "restricted":
                result["mo"].coeffs[:] = coeffsa_psi4
            else:
                result["mo"].coeffsa[:] = coeffsa_psi4
                result["mo"].coeffsb[:] = coeffsb_psi4
            return

    raise LoadError(
        "The molden or mkl file you are trying to load contains errors. "
        "Please make an issue here: https://github.com/theochem/iodata/issues, "
        "and attach this file and explain which program you used to create it. "
        "Please provide one or more small files causing this error. "
        "Thanks!",
        lit,
    )


def prepare_dump(filename: str, data: IOData):
    """Check the compatibility of the IOData object with the Molden format.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    data
        The IOData instance to be checked.
    """
    if data.mo is None:
        raise PrepareDumpError("The Molden format requires molecular orbitals.", filename)
    if data.obasis is None:
        raise PrepareDumpError("The Molden format requires an orbital basis set.", filename)
    if data.mo.occs_aminusb is not None:
        raise PrepareDumpError("Cannot write Molden file when mo.occs_aminusb is set.", filename)
    if data.mo.kind == "generalized":
        raise PrepareDumpError("Cannot write Molden file with generalized orbitals.", filename)


@document_dump_one("Molden", ["atcoords", "atnums", "mo", "obasis"], ["atcorenums", "title"])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # Print the header
    f.write("[Molden Format]\n")
    if data.title is not None:
        f.write("[Title]\n")
        f.write(f" {data.title}\n")

    # Print the elements numbers and the coordinates
    f.write("[Atoms] AU\n")
    for iatom in range(data.natom):
        atnum = data.atnums[iatom]
        atcorenum = data.atcorenums[iatom]
        x, y, z = data.atcoords[iatom]
        f.write(
            f"{num2sym[atnum].ljust(2):2s} {iatom + 1:3d} {atcorenum:3.0f}  "
            f"{x:25.18f} {y:25.18f} {z:25.18f}\n"
        )
    f.write("\n")

    # Print the basis set
    obasis = data.obasis

    # Figure out the pure/Cartesian situation. Note that the Molden
    # format does not support mixed Cartesian and pure functions for the,
    # same angular momentum. In practice, such combinations are too unlikely
    # to be relevant. If it happens, an error is raised.
    angmom_kinds = {}
    for shell in obasis.shells:
        for angmom, kind in zip(shell.angmoms, shell.kinds):
            if angmom in angmom_kinds:
                if kind != angmom_kinds[angmom]:
                    raise DumpError(
                        "Molden format does not support mixed pure+Cartesian functions for one "
                        "angular momentum.",
                        f,
                    )
            else:
                angmom_kinds[angmom] = kind

    # Fill in some defaults (Cartesian) for angmom kinds if needed.
    angmom_kinds.setdefault(2, "c")
    angmom_kinds.setdefault(3, "c")
    angmom_kinds.setdefault(4, "c")
    angmom_kinds.setdefault(5, "c")

    # Write out the Cartesian/Pure conventions. What a messy format...
    if angmom_kinds[2] == "p":
        if angmom_kinds[3] == "p":
            f.write("[5D]\n")
        else:
            f.write("[5D10F]\n")
    elif angmom_kinds[3] == "p":
        f.write("[7F]\n")
    if angmom_kinds[4] == "p":
        f.write("[9G]\n")

    f.write("[GTO]\n")
    last_icenter = -1
    # The shells must be sorted by center.
    for shell in sorted(obasis.shells, key=(lambda s: s.icenter)):
        if shell.icenter != last_icenter:
            if last_icenter != -1:
                f.write("\n")
            last_icenter = shell.icenter
            f.write("%3i 0\n" % (shell.icenter + 1))
        # Write out as a segmented basis. Molden format does not support
        # generalized contractions.
        for iangmom, angmom in enumerate(shell.angmoms):
            f.write(f" {angmom_its(angmom):1s}  {shell.nprim:3d} 1.00\n")
            for exponent, coeff in zip(shell.exponents, shell.coeffs[:, iangmom]):
                f.write(f"{exponent:20.10f} {coeff:20.10f}\n")
    f.write("\n")

    # Get the permutation to convert the orbital coefficients to Molden conventions.
    permutation, signs = convert_conventions(obasis, CONVENTIONS)

    # Print the mean-field orbitals
    if data.mo.kind == "unrestricted":
        f.write("[MO]\n")
        irrepsa = data.mo.irrepsa
        if irrepsa is None:
            irrepsa = ["1a"] * data.mo.norba
        _dump_helper_orb(
            f,
            "Alpha",
            data.mo.occsa,
            data.mo.coeffsa[permutation] * signs.reshape(-1, 1),
            data.mo.energiesa,
            irrepsa,
        )
        irrepsb = data.mo.irrepsb
        if irrepsb is None:
            irrepsb = ["1a"] * data.mo.norbb
        _dump_helper_orb(
            f,
            "Beta",
            data.mo.occsb,
            data.mo.coeffsb[permutation] * signs.reshape(-1, 1),
            data.mo.energiesb,
            irrepsb,
        )
    elif data.mo.kind == "restricted":
        f.write("[MO]\n")
        irreps = data.mo.irreps
        if irreps is None:
            irreps = ["1a"] * data.mo.norb
        _dump_helper_orb(
            f,
            "Alpha",
            data.mo.occs,
            data.mo.coeffs[permutation] * signs.reshape(-1, 1),
            data.mo.energies,
            irreps,
        )
    else:
        raise RuntimeError("This should not happen because of prepare_dump")


def _dump_helper_orb(f, spin, occs, coeffs, energies, irreps):
    for ifn in range(coeffs.shape[1]):
        f.write(f" Ene= {energies[ifn]:.17e}\n")
        f.write(f" Sym= {irreps[ifn]}\n")
        f.write(f" Spin= {spin}\n")
        f.write(f" Occup= {occs[ifn]:.17e}\n")
        for ibasis in range(coeffs.shape[0]):
            # The original molden floating-point formatting is too low
            # precision. Molden also reads high-precision, so we use this
            # instead.
            # f.write('{:4d} {:10.6f}\n'.format(ibasis + 1, orb_coeffs[ibasis, ifn]))
            f.write(f"{ibasis + 1:4d} {coeffs[ibasis, ifn]:.17e}\n")
