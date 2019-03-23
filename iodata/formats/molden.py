# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --
"""Module for handling MOLDEN file format."""


from typing import Tuple, Dict, Union, TextIO

import numpy as np

from ..periodic import sym2num, num2sym
from ..overlap import compute_overlap, get_shell_nbasis, gob_cart_normalization
from ..utils import (angstrom, str_to_shell_types, shell_type_to_str, shells_to_nbasis,
                     LineIterator)


__all__ = ['load', 'dump']


patterns = ['*.molden.input', '*.molden']


def _get_molden_permutation(shell_types: np.ndarray, reverse=False) -> np.ndarray:
    # Reorder the Cartesian basis functions to obtain the HORTON standard ordering.
    permutation_rules = {
        2: np.array([0, 3, 4, 1, 5, 2]),
        3: np.array([0, 4, 5, 3, 9, 6, 1, 8, 7, 2]),
        4: np.array([0, 3, 4, 9, 12, 10, 5, 13, 14, 7, 1, 6, 11, 8, 2]),
    }
    permutation = []
    for shell_type in shell_types:
        rule = permutation_rules.get(shell_type)
        if reverse and rule is not None:
            reverse_rule = np.zeros(len(rule), int)
            for i, j in enumerate(rule):
                reverse_rule[j] = i
            rule = reverse_rule
        if rule is None:
            rule = np.arange(get_shell_nbasis(shell_type))
        permutation.extend(rule + len(permutation))
    return np.array(permutation, dtype=int)


def _load_helper_coordinates(lit: LineIterator, cunit: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load element numbers and coordinates."""
    numbers = []
    pseudo_numbers = []
    coordinates = []
    for line in lit:
        if line.strip() == "":
            break
        words = line.split()
        if len(words) != 6:
            # Go back to previous line and stop
            lit.back(line)
            break
        numbers.append(sym2num[words[0].title()])
        pseudo_numbers.append(float(words[2]))
        coordinates.append([float(words[3]), float(words[4]), float(words[5])])
    numbers = np.array(numbers, int)
    pseudo_numbers = np.array(pseudo_numbers)
    coordinates = np.array(coordinates) * cunit
    return numbers, pseudo_numbers, coordinates


def _load_helper_obasis(lit: LineIterator, coordinates: np.ndarray) -> Tuple[Dict, int]:
    """Load the orbital basis."""
    shell_labels = []
    shell_map = []
    nprims = []
    alphas = []
    con_coeffs = []

    icenter = 0
    in_atom = False
    in_shell = False
    # Don't take this code as a good example. The structure with in_shell and
    # in_atom flags is not very transparent.
    while True:
        line = next(lit)
        words = line.split()
        if not words:
            in_atom = False
            in_shell = False
        elif len(words) == 2 and not in_atom:
            icenter = int(words[0]) - 1
            in_atom = True
            in_shell = False
        elif len(words) == 3:
            in_shell = True
            shell_map.append(icenter)
            shell_label = words[0].lower()
            shell_labels.append(shell_label)
            nprims.append(int(words[1]))
        elif len(words) == 2 and in_atom:
            assert in_shell
            alpha = float(words[0].replace('D', 'E'))
            alphas.append(alpha)
            con_coeff = float(words[1].replace('D', 'E'))
            con_coeffs.append(con_coeff)
        else:
            # done, go back one line
            lit.back(line)
            break

    shell_map = np.array(shell_map)
    nprims = np.array(nprims)
    alphas = np.array(alphas)
    con_coeffs = np.array(con_coeffs)

    obasis = {"centers": coordinates, "shell_map": shell_map, "nprims": nprims,
              "shell_labels": shell_labels, "alphas": alphas, "con_coeffs": con_coeffs}

    return obasis


# pylint: disable=too-many-branches,too-many-statements
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

    print(coeff_alpha)
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


# pylint: disable=too-many-branches,too-many-statements
def load(lit: LineIterator) -> Dict:
    """Load data from a MOLDEN input file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        output dictionary containing ``coordinates``, ``numbers``, ``pseudo_numbers``,
        ``obasis``, ``orb_alpha`` & ``signs`` keys and corresponding values. It may contain
        ``title`` and ``orb_beta`` keys and their values as well.

    """
    pure = {'d': False, 'f': False, 'g': False}
    numbers = None
    coordinates = None
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
    while True:
        try:
            line = next(lit).lower().strip()
        except StopIteration:
            # This means we continue reading till the end of the file.
            # There is no real way to know when a Molden file has ended, other
            # than reaching the end of the file.
            break
        if line.startswith('[5d]') or line.startswith('[5d7f]'):
            pure['d'] = True
            pure['f'] = True
        elif line.lower().startswith('[7f]'):
            pure['f'] = True
        elif line.lower().startswith('[5d10f]'):
            pure['d'] = True
            pure['f'] = False
        elif line.lower().startswith('[9g]'):
            pure['g'] = True
        elif line == '[title]':
            title = next(lit).strip()
        elif line.startswith('[atoms]'):
            if 'au' in line:
                cunit = 1.0
            elif 'angs' in line:
                cunit = angstrom
            numbers, pseudo_numbers, coordinates = _load_helper_coordinates(lit, cunit)
        elif line == '[gto]':
            obasis = _load_helper_obasis(lit, coordinates)
        elif line == '[sto]':
            lit.error('Slater-type orbitals are not supported by IODATA.')
        elif line == '[mo]':
            data_alpha, data_beta = _load_helper_coeffs(lit)
            coeff_alpha, ener_alpha, occ_alpha = data_alpha
            coeff_beta, ener_beta, occ_beta = data_beta

    # Convert shell_labels to shell_types.
    # This needs to be done after reading because the tags for pure functions
    # may come at the end.
    obasis['shell_types'] = np.array([
        str_to_shell_types(shell_label, pure.get(shell_label, False))[0]
        for shell_label in obasis['shell_labels']])
    del obasis['shell_labels']
    nbasis = shells_to_nbasis(obasis['shell_types'])

    if coeff_beta is None:
        if coeff_alpha.shape[0] != nbasis:
            lit.error("Number of alpha orbital coefficients does not match the size of the basis.")
        orb_alpha = (nbasis, coeff_alpha.shape[1])
        orb_alpha_coeffs = coeff_alpha
        orb_alpha_energies = ener_alpha
        orb_alpha_occs = occ_alpha / 2
        orb_beta = None
    else:
        if coeff_beta.shape[0] != nbasis:
            lit.error("Number of beta orbital coefficients does not match the size of the basis.")
        orb_alpha = (nbasis, coeff_alpha.shape[1])
        orb_alpha_coeffs = coeff_alpha
        orb_alpha_energies = ener_alpha
        orb_alpha_occs = occ_alpha
        orb_beta = (nbasis, coeff_beta.shape[1])
        orb_beta_coeffs = coeff_beta
        orb_beta_energies = ener_beta
        orb_beta_occs = occ_beta

    permutation = _get_molden_permutation(obasis["shell_types"])

    # filter out ghost atoms
    mask = pseudo_numbers != 0
    coordinates = coordinates[mask]
    numbers = numbers[mask]
    pseudo_numbers = pseudo_numbers[mask]

    result = {
        'coordinates': coordinates,
        'orb_alpha': orb_alpha,
        'orb_alpha_coeffs': orb_alpha_coeffs,
        'orb_alpha_energies': orb_alpha_energies,
        'orb_alpha_occs': orb_alpha_occs,
        'numbers': numbers,
        'obasis': obasis,
        'permutation': permutation,
        'pseudo_numbers': pseudo_numbers,
    }
    if title is not None:
        result['title'] = title
    if orb_beta is not None:
        result['orb_beta'] = orb_beta
        result['orb_beta_coeffs'] = orb_beta_coeffs
        result['orb_beta_energies'] = orb_beta_energies
        result['orb_beta_occs'] = orb_beta_occs

    _fix_molden_from_buggy_codes(result, lit.filename)
    return result


def _is_normalized_properly(obasis: Dict, permutation: np.ndarray, orb_alpha: np.ndarray,
                            orb_beta: np.ndarray, signs: np.ndarray = None,
                            threshold: float = 1e-4):
    """Test the normalization of the occupied and virtual orbitals.

    Parameters
    ----------
    obasis
        A dictionary containing the parameters of the GOBasis class.
    permutation
        The permutation of the basis functions to bring them in HORTON's
        standard ordering.
    orb_alpha
        The alpha orbitals coefficients
    orb_beta
        The beta orbitals (may be None).
    signs
        Changes in sign conventions.
    threshold
        When the maximal error on the norm is large than the threshold,
        the function returns False. True is returned otherwise.

    """
    # Set default value for signs
    if signs is None:
        signs = np.ones(orb_alpha.shape[0], int)
    # Compute the overlap matrix.
    olp = compute_overlap(**obasis)

    # Convenient code for debugging files coming from crappy QC codes.
    # np.set_printoptions(linewidth=5000, precision=2, suppress=True, threshold=100000)
    # coeffs = orb_alpha._coeffs
    # if permutation is not None:
    #     coeffs = coeffs[permutation].copy()
    # if signs is not None:
    #     coeffs = coeffs*signs.reshape(-1, 1)
    # print np.dot(coeffs.T, np.dot(olp._array, coeffs))
    # print

    # Compute the norm of each occupied and virtual orbital. Keep track of
    # the largest deviation from unity
    orbs = [orb_alpha]
    if orb_beta is not None:
        orbs.append(orb_beta)
    error_max = 0.0
    for orb in orbs:
        for iorb in range(orb.shape[1]):
            vec = orb[:, iorb].copy()
            if signs is not None:
                vec *= signs
            if permutation is not None:
                vec = vec[permutation]
            norm = np.dot(vec, np.dot(olp, vec))
            # print iorb, norm
            error_max = max(error_max, abs(norm - 1))

    # final judgement
    return error_max <= threshold


def _get_orca_signs(shell_types: np.ndarray) -> np.ndarray:
    """Return an array with sign corrections for orbitals read from ORCA.

    Parameters
    ----------
    shell_types
        An array with integer shell types.

    Returns
    -------
    signs
        An array with sign flips.

    """
    sign_rules = {
        -4: [1, 1, 1, 1, 1, -1, -1, -1, -1],
        -3: [1, 1, 1, 1, 1, -1, -1],
        -2: [1, 1, 1, 1, 1],
        0: [1],
        1: [1, 1, 1],
    }
    signs = []
    for shell_type in shell_types:
        if shell_type in sign_rules:
            signs.extend(sign_rules[shell_type])
        else:
            signs.extend([1] * get_shell_nbasis(shell_type))
    return np.array(signs, dtype=int)


# pylint: disable=too-many-branches
def _get_fixed_con_coeffs(nprims: np.ndarray, shell_types: np.ndarray, alphas: np.ndarray,
                          con_coeffs: np.ndarray, code: str) -> Union[np.ndarray, None]:
    """Return corrected contraction coefficients, assuming they came from a broken QC code.

    The arguments here are the same as the ones that are given from ``helper_obasis``.

    Parameters
    ----------
    nprims
        Number of primitives in each shell
    shell_types
        Shell angular momenta
    alphas
        Gaussian basis exponents
    con_coeffs
        Contraction coefficients
    code
        Name of code: 'orca', 'psi4' or 'turbomole'.

    Returns
    -------
    fixed_con_coeffs
        Corrected contraction coefficients, or None if corrections were not applicable.

    """
    assert code in ['orca', 'psi4', 'turbomole']
    fixed_con_coeffs = con_coeffs.copy()
    iprim = 0
    corrected = False
    for ishell in range(shell_types.size):
        shell_type = shell_types[ishell]
        for _ialpha in range(nprims[ishell]):
            alpha = alphas[iprim]
            # Default 1.0: do not to correct anything, unless we know how to correct.
            correction = 1.0
            if code == 'turbomole':
                if shell_type == 2:
                    correction = 1.0 / np.sqrt(3.0)
                elif shell_type == 3:
                    correction = 1.0 / np.sqrt(15.0)
                elif shell_type == 4:
                    correction = 1.0 / np.sqrt(105.0)
            elif code == 'orca':
                if shell_type == 0:
                    correction = gob_cart_normalization(alpha, np.array([0, 0, 0]))
                elif shell_type == 1:
                    correction = gob_cart_normalization(alpha, np.array([1, 0, 0]))
                elif shell_type == -2:
                    correction = gob_cart_normalization(alpha, np.array([1, 1, 0]))
                elif shell_type == -3:
                    correction = gob_cart_normalization(alpha, np.array([1, 1, 1]))
                elif shell_type == -4:
                    correction = gob_cart_normalization(alpha, np.array([2, 1, 1]))
            elif code == 'psi4':
                if shell_type == 0:
                    correction = gob_cart_normalization(alpha, np.array([0, 0, 0]))
                elif shell_type == 1:
                    correction = gob_cart_normalization(alpha, np.array([1, 0, 0]))
                elif shell_type == -2:
                    correction = gob_cart_normalization(alpha, np.array([1, 1, 0])) / np.sqrt(3.0)
                elif shell_type == -3:
                    correction = gob_cart_normalization(alpha, np.array([1, 1, 1])) / np.sqrt(15.0)
                # elif shell_type == -4: ##  ! Not tested
                #     correction = gob_cart_normalization(alpha, np.array([2, 1, 1]))/np.sqrt(105.0)
            fixed_con_coeffs[iprim] /= correction
            if correction != 1.0:
                corrected = True
            iprim += 1
    if corrected:
        return fixed_con_coeffs
    return None


def _normalized_contractions(obasis_dict: Dict):
    """Return contraction coefficients of normalized contractions."""
    # Files written by Molden don't need this and have properly normalized contractions.
    # When Molden reads files in the Molden format, it does renormalize the contractions
    # and other programs than Molden may generate Molden files with unnormalized
    # contractions. Note: this renormalization is only a last resort in HORTON. If we
    # would do it up-front, like Molden, we would not be able to fix errors in files from
    # ORCA and older PSI-4 versions.
    iprim = 0
    new_con_coeffs = obasis_dict['con_coeffs'].copy()
    for nprim, shell_type in zip(obasis_dict['nprims'], obasis_dict['shell_types']):
        centers = np.array([[0.0, 0.0, 0.0]])
        shell_map = np.array([0])
        shell_types = np.array([shell_type])
        alphas = obasis_dict['alphas'][iprim:iprim + nprim]
        nprims = np.array([len(alphas)])
        con_coeffs = new_con_coeffs[iprim:iprim + nprim]

        # 2) Get the first diagonal element of the overlap matrix
        olpdiag = compute_overlap(centers, shell_map, nprims, shell_types, alphas, con_coeffs)[0, 0]
        # 3) Normalize the contraction
        con_coeffs /= np.sqrt(olpdiag)
        iprim += nprim
    return new_con_coeffs


def _fix_molden_from_buggy_codes(result: Dict, filename: str):
    """Detect errors in the data loaded from a molden/mkl/... file and correct.

    This function can recognize erroneous files created by PSI4, ORCA and Turbomole. The
    values in `results` for the `obasis` and `signs` keys will be updated accordingly.

    Parameters
    ----------
    result
        A dictionary with the data loaded in the ``load_molden`` function.
    filename
        The name of the molden/mkl/... file.

    """
    obasis_dict = result['obasis']
    permutation = result.get('permutation', None)
    if _is_normalized_properly(obasis_dict, permutation, result['orb_alpha_coeffs'],
                               result.get('orb_beta_coeffs')):
        # The file is good. No need to change data.
        return
    print('5:Detected incorrect normalization of orbitals loaded from a file.')

    # --- ORCA
    print('5:Trying to fix it as if it was a file generated by ORCA.')
    orca_signs = _get_orca_signs(obasis_dict['shell_types'])
    orca_con_coeffs = _get_fixed_con_coeffs(obasis_dict["nprims"], obasis_dict["shell_types"],
                                            obasis_dict["alphas"], obasis_dict["con_coeffs"],
                                            code='orca')
    if orca_con_coeffs is not None:
        # Only try if some changes were made to the contraction coefficients.
        obasis_dict_orca = obasis_dict.copy()
        obasis_dict_orca["con_coeffs"] = orca_con_coeffs
        if _is_normalized_properly(obasis_dict_orca, permutation,
                                   result['orb_alpha_coeffs'], result.get('orb_beta_coeffs'),
                                   orca_signs):
            print('5:Detected typical ORCA errors in file. Fixing them...')
            result['obasis'] = obasis_dict_orca
            result['signs'] = orca_signs
            return

    # --- PSI4
    print('5:Trying to fix it as if it was a file generated by PSI4 (pre 1.0).')
    psi4_con_coeffs = _get_fixed_con_coeffs(obasis_dict["nprims"], obasis_dict["shell_types"],
                                            obasis_dict["alphas"], obasis_dict["con_coeffs"],
                                            code='psi4')
    if psi4_con_coeffs is not None:
        # Only try if some changes were made to the contraction coefficients.
        obasis_dict_psi4 = obasis_dict.copy()
        obasis_dict_psi4["con_coeffs"] = psi4_con_coeffs
        if _is_normalized_properly(obasis_dict_psi4, permutation,
                                   result['orb_alpha_coeffs'], result.get('orb_beta_coeffs')):
            print('5:Detected typical PSI4 errors in file. Fixing them...')
            result['obasis'] = obasis_dict_psi4
            return

    # -- Turbomole
    print('5:Trying to fix it as if it was a file generated by Turbomole.')
    tb_con_coeffs = _get_fixed_con_coeffs(obasis_dict["nprims"], obasis_dict["shell_types"],
                                          obasis_dict["alphas"], obasis_dict["con_coeffs"],
                                          code='turbomole')
    if tb_con_coeffs is not None:
        # Only try if some changes were made to the contraction coefficients.
        obasis_dict_tb = obasis_dict.copy()
        obasis_dict_tb["con_coeffs"] = tb_con_coeffs
        if _is_normalized_properly(obasis_dict_tb, permutation,
                                   result['orb_alpha_coeffs'], result.get('orb_beta_coeffs')):
            print('5:Detected typical Turbomole errors in file. Fixing them...')
            result['obasis'] = obasis_dict_tb
            return

    # --- Renormalized contractions
    print('5:Last resort: trying by renormalizing all contractions')
    normed_con_coeffs = _normalized_contractions(obasis_dict)
    obasis_dict_norm = obasis_dict.copy()
    obasis_dict_norm["con_coeffs"] = normed_con_coeffs
    if _is_normalized_properly(obasis_dict_norm, permutation,
                               result['orb_alpha_coeffs'], result.get('orb_beta_coeffs')):
        print('5:Detected unnormalized contractions in file. Fixing them...')
        result['obasis'] = obasis_dict_norm
        return

    raise IOError(('Could not correct the data read from %s. The molden or '
                   'mkl file you are trying to load contains errors. Please '
                   'report this problem to Toon.Verstraelen@UGent.be, so he '
                   'can fix it.') % filename)


# pylint: disable=too-many-branches,too-many-statements
def dump(f: TextIO, data: 'IOData'):
    """Write data into a MOLDEN input file format.

    Parameters
    ----------
    f
        A file to write to.
    data : IOData
        An IOData instance which must contain ```coordinates``, ``numbers``,
        ``obasis`` & ``orb_alpha`` attributes. It may contain ```title``,
        ``pseudo_numbers``, ``orb_beta`` attributes.

    """
    # Print the header
    print('[Molden Format]', file=f)
    print('[Title]', file=f)
    print(' %s' % getattr(data, 'title', 'Created with HORTON'), file=f)
    print(file=f)

    # Print the elements numbers and the coordinates
    print('[Atoms] AU', file=f)
    for iatom in range(data.natom):
        number = data.numbers[iatom]
        pseudo_number = data.pseudo_numbers[iatom]
        x, y, z = data.coordinates[iatom]
        print('%2s %3i %3i  %25.18f %25.18f %25.18f' % (
            num2sym[number].ljust(2), iatom + 1, pseudo_number, x, y, z
        ), file=f)

    # Print the basis set
    if isinstance(data.obasis, dict):
        # Figure out the pure/Cartesian situation. Note that the Molden
        # format does not support mixed Cartesian and pure functions in the
        # way HORTON does. In practice, such combinations are too unlikely
        # to be relevant.
        pure = {'d': None, 'f': None, 'g': None}
        try:
            for shell_type in data.obasis["shell_types"]:
                if shell_type == 2:
                    assert pure['d'] is None or not pure['d']
                    pure['d'] = False
                elif shell_type == -2:
                    assert pure['d'] is None or pure['d']
                    pure['d'] = True
                elif shell_type == 3:
                    assert pure['f'] is None or not pure['f']
                    pure['f'] = False
                elif shell_type == -3:
                    assert pure['f'] is None or pure['f']
                    pure['f'] = True
                elif shell_type == 4:
                    assert pure['g'] is None or not pure['g']
                    pure['g'] = False
                elif shell_type == -4:
                    assert pure['g'] is None or pure['g']
                    pure['g'] = True
                else:
                    assert abs(shell_type) < 2
        except AssertionError:
            raise IOError('The basis set is not supported by the Molden format.')

        # Write out the Cartesian/Pure conventions. What a messy format...
        if pure['d']:
            if pure['f']:
                print('[5D]', file=f)
            else:
                print('[5D10F]', file=f)
        else:
            if pure['f']:
                print('[7F]', file=f)
        if pure['g']:
            print('[9G]', file=f)

        # First convert it to a format that is amenable for printing. The molden
        # format assumes that every basis function is centered on one of the atoms.
        # (This may not always be the case.)
        centers = [list() for _ in range(data.obasis["centers"].shape[0])]
        begin_prim = 0
        for ishell in range(data.obasis["shell_types"].size):
            icenter = data.obasis["shell_map"][ishell]
            shell_type = data.obasis["shell_types"][ishell]
            sts = shell_type_to_str(shell_type)
            end_prim = begin_prim + data.obasis["nprims"][ishell]
            prims = []
            for iprim in range(begin_prim, end_prim):
                alpha = data.obasis["alphas"][iprim]
                con_coeff = data.obasis["con_coeffs"][iprim]
                prims.append((alpha, con_coeff))
            centers[icenter].append((sts, prims))
            begin_prim = end_prim

        print('[GTO]', file=f)
        for icenter in range(data.obasis["centers"].shape[0]):
            print('%3i 0' % (icenter + 1), file=f)
            for sts, prims in centers[icenter]:
                print('%1s %3i 1.0' % (sts, len(prims)), file=f)
                for alpha, con_coeff in prims:
                    print('%20.10f %20.10f' % (alpha, con_coeff), file=f)
            print(file=f)
    else:
        raise NotImplementedError(
            'A Gaussian orbital basis is required to write a molden input file.')

    def helper_orb(spin, occ_scale=1.0):
        orb_coeffs = getattr(data, f'orb_{spin}_coeffs')
        orb_energies = getattr(data, f'orb_{spin}_energies')
        orb_occupations = getattr(data, f'orb_{spin}_occs')
        for ifn in range(orb_coeffs.shape[1]):
            print(' Sym=     1a', file=f)
            print(f' Ene= {orb_energies[ifn]:20.14E}', file=f)
            print(f' Spin= {spin.capitalize()}', file=f)
            print(f' Occup= {orb_occupations[ifn] * occ_scale:8.6f}', file=f)
            for ibasis in range(orb_coeffs.shape[0]):
                print('%3i %20.12f' % (ibasis + 1, orb_coeffs[permutation[ibasis], ifn]),
                      file=f)

    # Construct the permutation of the basis functions
    permutation = _get_molden_permutation(data.obasis["shell_types"], reverse=True)

    # Print the mean-field orbitals
    if hasattr(data, 'orb_beta_coeffs'):
        print('[MO]', file=f)
        helper_orb('alpha')
        helper_orb('beta')
    else:
        print('[MO]', file=f)
        helper_orb('alpha', 2.0)
