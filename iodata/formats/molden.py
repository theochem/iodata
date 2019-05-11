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
"""Module for handling MOLDEN file format."""


from typing import Tuple, Union, TextIO
import copy

import numpy as np

from ..basis import (angmom_its, angmom_sti, MolecularBasis, Shell,
                     convert_conventions, HORTON2_CONVENTIONS)
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..overlap import compute_overlap, gob_cart_normalization
from ..utils import angstrom, LineIterator, MolecularOrbitals


__all__ = []


patterns = ['*.molden.input', '*.molden']

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
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'p'): HORTON2_CONVENTIONS[(4, 'p')],
    (4, 'c'): ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz',
               'yzzz', 'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz'],
}


def load_one(lit: LineIterator) -> dict:
    """Load data from a MOLDEN input file format.

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
    result = _load_low(lit)
    _fix_molden_from_buggy_codes(result, lit.filename)
    return result


# pylint: disable=too-many-branches,too-many-statements
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
    pure_angmoms = set([])
    atnums = None
    atcoords = None
    obasis = None
    coeff_alpha = None
    ener_alpha = None
    occ_alpha = None
    coeff_beta = None
    ener_beta = None
    occ_beta = None
    title = None

    line = next(lit)
    if line != '[Molden Format]\n':
        lit.error('Molden header not found')
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
        if line.startswith('[5d]') or line.startswith('[5d7f]'):
            pure_angmoms.add(2)
            pure_angmoms.add(3)
        elif line.lower().startswith('[7f]'):
            pure_angmoms.add(3)
        elif line.lower().startswith('[5d10f]'):
            pure_angmoms.add(2)
        elif line.lower().startswith('[9g]'):
            pure_angmoms.add(4)
        # title
        elif line == '[title]':
            title = next(lit).strip()
        # atoms
        elif line.startswith('[atoms]'):
            if 'au' in line:
                cunit = 1.0
            elif 'angs' in line:
                cunit = angstrom
            atnums, atcorenums, atcoords = _load_helper_atoms(lit, cunit)
        # we only support Gaussian-type orbitals (gto's)
        elif line == '[gto]':
            obasis = _load_helper_obasis(lit)
        elif line == '[sto]':
            lit.error('Slater-type orbitals are not supported by IODATA.')
        # molecular-orbital coefficients.
        elif line == '[mo]':
            data_alpha, data_beta = _load_helper_coeffs(lit)
            coeff_alpha, ener_alpha, occ_alpha = data_alpha
            coeff_beta, ener_beta, occ_beta = data_beta

    # Assign pure and Cartesian correctly. This needs to be done after reading
    # because the tags for pure functions may come after the basis set.
    for shell in obasis.shells:
        # Code only has to work for segmented contractions
        if shell.angmoms[0] in pure_angmoms:
            shell.kinds[0] = 'p'

    if coeff_beta is None:
        mo_type = 'restricted'
        if coeff_alpha.shape[0] != obasis.nbasis:
            lit.error("Number of alpha orbital coefficients does not match the size of the basis.")
        norba = norbb = coeff_alpha.shape[1]
        mo_occs = occ_alpha
        mo_coeffs = coeff_alpha
        mo_energy = ener_alpha
    else:
        mo_type = 'unrestricted'
        if coeff_beta.shape[0] != obasis.nbasis:
            lit.error("Number of beta orbital coefficients does not match the size of the basis.")
        norba = coeff_alpha.shape[1]
        norbb = coeff_beta.shape[1]
        mo_occs = np.concatenate((occ_alpha, occ_beta), axis=0)
        mo_energy = np.concatenate((ener_alpha, ener_beta), axis=0)
        mo_coeffs = np.concatenate((coeff_alpha, coeff_beta), axis=1)
    # create a MO namedtuple
    mo = MolecularOrbitals(mo_type, norba, norbb, mo_occs, mo_coeffs, None, mo_energy)

    result = {
        'atcoords': atcoords,
        'atnums': atnums,
        'obasis': obasis,
        'mo': mo,
        'atcorenums': atcorenums,
    }
    if title is not None:
        result['title'] = title
    return result


def _load_helper_atoms(lit: LineIterator, cunit: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                exponents[iprim] = float(words[0].replace('D', 'E'))
                coeffs[iprim, 0] = float(words[1].replace('D', 'E'))
            # Unless changed later, all shells are assumed to be Cartesian.
            shells.append(Shell(icenter, [angmom], ['c'], exponents, coeffs))
    return MolecularBasis(shells, CONVENTIONS, 'L2')


def _load_helper_coeffs(lit: LineIterator) -> Tuple:
    """Load the orbital coefficients."""
    coeff_alpha = []
    ener_alpha = []
    occ_alpha = []
    coeff_beta = []
    ener_beta = []
    occ_beta = []

    while True:
        try:
            line = next(lit).lower().strip()
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
        if '[' in line:
            lit.back(line)
            break
        # prepare array with orbital coefficients
        info = {}
        lit.back(line)
        for line in lit:
            if line.count('=') != 1:
                lit.back(line)
                break
            key, value = line.split('=')
            info[key.strip().lower()] = value
        energy = float(info['ene'])
        occ = float(info['occup'])
        col = []
        # store column of coefficients, i.e. one orbital, energy and occ
        if info['spin'].strip().lower() == 'alpha':
            coeff_alpha.append(col)
            ener_alpha.append(energy)
            occ_alpha.append(occ)
        else:
            coeff_beta.append(col)
            ener_beta.append(energy)
            occ_beta.append(occ)
        for line in lit:
            words = line.split()
            if len(words) != 2 or not words[0].isdigit():
                # The line does not look like an index with an orbital coefficient.
                # Time to stop and put the line back
                lit.back(line)
                break
            col.append(float(words[1]))

    coeff_alpha = np.array(coeff_alpha).T
    ener_alpha = np.array(ener_alpha)
    occ_alpha = np.array(occ_alpha)
    if not coeff_beta:
        coeff_beta = None
        ener_beta = None
        occ_beta = None
    else:
        coeff_beta = np.array(coeff_beta).T
        ener_beta = np.array(ener_beta)
        occ_beta = np.array(occ_beta)
    return (coeff_alpha, ener_alpha, occ_alpha), (coeff_beta, ener_beta, occ_beta)


def _is_normalized_properly(obasis: MolecularBasis, atcoords: np.ndarray,
                            orb_alpha: np.ndarray, orb_beta: np.ndarray,
                            threshold: float = 1e-4) -> bool:
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
    threshold
        When the maximal error on the norm is large than the threshold,
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
    return error_max <= threshold


def _fix_obasis_orca(obasis: MolecularBasis) -> MolecularBasis:
    """Return a new MolecularBasis correcting for errors from ORCA.

    Orca has different normalization conventions for the primitives and also
    different sign conventions for some of the pure functions.
    """
    orca_conventions = {
        (0, 'c'): ['1'],
        (1, 'c'): ['x', 'y', 'z'],
        (2, 'p'): ['dc0', 'dc1', 'ds1', 'dc2', 'ds2'],
        (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
        (3, 'p'): ['fc0', 'fc1', 'fs1', 'fc2', 'fs2', '-fc3', '-fs3'],
        (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
        (4, 'p'): ['gc0', 'gc1', 'gs1', 'gc2', 'gs2', '-gc3', '-gs3', '-gc4', '-gs4'],
        (4, 'c'): ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'xyyy', 'yyyz', 'xzzz',
                   'yzzz', 'xxyy', 'xxzz', 'yyzz', 'xxyz', 'xyyz', 'xyzz'],
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
            elif angmom == 2 and kind == 'p':
                correction = gob_cart_normalization(exponent, np.array([1, 1, 0]))
            elif angmom == 3 and kind == 'p':
                correction = gob_cart_normalization(exponent, np.array([1, 1, 1]))
            elif angmom == 4 and kind == 'p':
                correction = gob_cart_normalization(exponent, np.array([2, 1, 1]))
            if correction != 1.0:
                fixed_shell.coeffs[iprim, 0] /= correction
            iprim += 1
    return MolecularBasis(fixed_shells, orca_conventions, obasis.primitive_normalization)


def _fix_obasis_psi4(obasis: MolecularBasis) -> Union[MolecularBasis, None]:
    """Return a new MolecularBasis correcting for errors from old PSI4 versions.

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
            elif angmom == 2 and kind == 'p':
                correction = gob_cart_normalization(exponent, np.array([1, 1, 0])) / np.sqrt(3.0)
            elif angmom == 3 and kind == 'p':
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
            if angmom == 2 and kind == 'c':
                correction = 1.0 / np.sqrt(3.0)
            elif angmom == 3 and kind == 'c':
                correction = 1.0 / np.sqrt(15.0)
            elif angmom == 4 and kind == 'c':
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
            [shell._replace(icenter=0)],
            obasis.conventions,
            obasis.primitive_normalization
        )
        # 2) Get the first diagonal element of the overlap matrix
        olpdiag = compute_overlap(shell_obasis, np.zeros((3, 1), float))[0, 0]
        # 3) Normalize the contraction
        fixed_shell = copy.deepcopy(shell)
        fixed_shell.coeffs[:] /= np.sqrt(olpdiag)
        fixed_shells.append(fixed_shell)
    return MolecularBasis(fixed_shells, obasis.conventions, obasis.primitive_normalization)


def _fix_molden_from_buggy_codes(result: dict, filename: str):
    """Detect errors in the data loaded from a molden or mkl file and correct.

    This function can recognize erroneous files created by PSI4, ORCA and
    Turbomole. The value `results['obasis']` will be updated accordingly.

    Parameters
    ----------
    result
        A dictionary with the data loaded in the ``load_molden`` function.
    filename
        The name of the molden/mkl/... file.

    """
    obasis = result['obasis']
    atcoords = result['atcoords']
    if result['mo'].type == 'restricted':
        coeffs_a = result['mo'].coeffs
        coeffs_b = None
    elif result['mo'].type == 'unrestricted':
        coeffs_a = result['mo'].coeffs[:, :result['mo'].norba]
        coeffs_b = result['mo'].coeffs[:, result['mo'].norba:]
    else:
        raise ValueError('Molecular orbital type={0} not recognized'.format(result['mo'].type))
    if _is_normalized_properly(obasis, atcoords, coeffs_a, coeffs_b):
        # The file is good. No need to change obasis.
        return
    print('5:Detected incorrect normalization of orbitals loaded from a Molden or MKL file.')

    # --- ORCA
    print('5:Trying to fix it as if it was a file generated by ORCA.')
    orca_obasis = _fix_obasis_orca(obasis)
    if _is_normalized_properly(orca_obasis, atcoords, coeffs_a, coeffs_b):
        print('5:Detected typical ORCA errors in file. Fixing them...')
        result['obasis'] = orca_obasis
        return

    # --- PSI4
    print('5:Trying to fix it as if it was a file generated by PSI4 (pre 1.0).')
    psi4_obasis = _fix_obasis_psi4(obasis)
    if psi4_obasis is not None and \
       _is_normalized_properly(psi4_obasis, atcoords, coeffs_a, coeffs_b):
        print('5:Detected typical PSI4 errors in file. Fixing them...')
        result['obasis'] = psi4_obasis
        return

    # -- Turbomole
    print('5:Trying to fix it as if it was a file generated by Turbomole.')
    turbom_obasis = _fix_obasis_turbomole(obasis)
    if turbom_obasis is not None and \
       _is_normalized_properly(turbom_obasis, atcoords, coeffs_a, coeffs_b):
        print('5:Detected typical Turbomole errors in file. Fixing them...')
        result['obasis'] = turbom_obasis
        return

    # --- Renormalized contractions
    print('5:Last resort: trying by renormalizing all contractions')
    normed_obasis = _fix_obasis_normalize_contractions(obasis)
    if _is_normalized_properly(normed_obasis, atcoords, coeffs_a, coeffs_b):
        print('5:Detected unnormalized contractions in file. Fixing them...')
        result['obasis'] = normed_obasis
        return

    raise IOError(('Could not correct the data read from {}. The molden or '
                   'mkl file you are trying to load contains errors. Please '
                   'make an issue here: https://github.com/theochem/iodata/issues, '
                   'and attach a this file. Please provide one or more small '
                   'files causing this error'.format(filename)))


def dump_one(f: TextIO, data: IOData):
    """Write data into a MOLDEN input file format.

    Parameters
    ----------
    f
        A file to write to.
    data : IOData
        An IOData instance which must contain ```atcoords``, ``atnums``,
        ``obasis`` & ``orb_alpha`` attributes. It may contain ```title``,
        ``atcorenums``, ``orb_beta`` attributes.

    """
    # Print the header
    f.write('[Molden Format]\n')
    if hasattr(data, 'title'):
        f.write('[Title]\n')
        f.write(' {}\n'.format(data.title))

    # Print the elements numbers and the coordinates
    f.write('[Atoms] AU\n')
    for iatom in range(data.natom):
        atnum = data.atnums[iatom]
        atcorenum = data.atcorenums[iatom]
        x, y, z = data.atcoords[iatom]
        f.write('{:2s} {:3d} {:3.0f}  {:25.18f} {:25.18f} {:25.18f}\n'.format(
            num2sym[atnum].ljust(2), iatom + 1, atcorenum, x, y, z
        ))
    f.write('\n')

    # Print the basis set
    if not hasattr(data, 'obasis'):
        raise IOError('A Gaussian orbital basis is required to write a molden input file.')
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
                    raise IOError('Molden format does not support mixed '
                                  'pure+Cartesian functions for one '
                                  'angular momentum.')
            else:
                angmom_kinds[angmom] = kind

    # Fill in some defaults (Cartesian) for angmom kinds if needed.
    angmom_kinds.setdefault(2, 'c')
    angmom_kinds.setdefault(3, 'c')
    angmom_kinds.setdefault(4, 'c')

    # Write out the Cartesian/Pure conventions. What a messy format...
    if angmom_kinds[2] == 'p':
        if angmom_kinds[3] == 'p':
            f.write('[5D]\n')
        else:
            f.write('[5D10F]\n')
    else:
        if angmom_kinds[3] == 'p':
            f.write('[7F]\n')
    if angmom_kinds[4] == 'p':
        f.write('[9G]\n')

    f.write('[GTO]\n')
    last_icenter = -1
    # The shells must be sorted by center.
    for shell in sorted(obasis.shells, key=(lambda s: s.icenter)):
        if shell.icenter != last_icenter:
            if last_icenter != -1:
                f.write("\n")
            last_icenter = shell.icenter
            f.write('%3i 0\n' % (shell.icenter + 1))
        # Decontract the basis
        for iangmom, angmom in enumerate(shell.angmoms):
            f.write(' {:1s}  {:3d} 1.00\n'.format(angmom_its(angmom), shell.nprim))
            for exponent, coeff in zip(shell.exponents, shell.coeffs[:, iangmom]):
                f.write('{:20.10f} {:20.10f}\n'.format(exponent, coeff))
    f.write("\n")

    # Get the permutation to convert the orbital coefficients to Molden conventions.
    permutation, signs = convert_conventions(obasis, CONVENTIONS)

    # Print the mean-field orbitals
    if data.mo.type == 'unrestricted':
        f.write('[MO]\n')
        norba = data.mo.norba
        _dump_helper_orb(f, 'Alpha', data.mo.energies[:norba], data.mo.occs[:norba],
                         data.mo.coeffs[:, :norba][permutation] * signs.reshape(-1, 1))
        _dump_helper_orb(f, 'Beta', data.mo.energies[norba:], data.mo.occs[norba:],
                         data.mo.coeffs[:, norba:][permutation] * signs.reshape(-1, 1))
    else:
        f.write('[MO]\n')
        _dump_helper_orb(f, 'Alpha', data.mo.energies, data.mo.occs,
                         data.mo.coeffs[permutation] * signs.reshape(-1, 1))


def _dump_helper_orb(f, spin, orb_energies, orb_occs, orb_coeffs):
    for ifn in range(orb_coeffs.shape[1]):
        f.write(f' Ene= {orb_energies[ifn]:.17e}\n')
        f.write(' Sym=     1a\n')
        f.write(f' Spin= {spin}\n')
        f.write(f' Occup= {orb_occs[ifn]:.17e}\n')
        for ibasis in range(orb_coeffs.shape[0]):
            # The original molden floating-point formatting is too low
            # precision. Molden also reads high-precision, so we use this
            # instead.
            # f.write('{:4d} {:10.6f}\n'.format(ibasis + 1, orb_coeffs[ibasis, ifn]))
            f.write('{:4d} {:.17e}\n'.format(ibasis + 1, orb_coeffs[ibasis, ifn]))
