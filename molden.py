# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Molden wavefunction input file format"""

from __future__ import print_function

import numpy as np

from .overlap import compute_overlap, get_shell_nbasis, gob_cart_normalization
from .periodic import sym2num, num2sym
from .utils import angstrom, str_to_shell_types, shell_type_to_str, shells_to_nbasis

__all__ = ['load_molden', 'dump_molden']


def _get_molden_permutation(shell_types, reverse=False):
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


def load_molden(filename):
    """Load data from a molden input file.

    Parameters
    ----------
    filename : str
        The filename of the molden input file.

    Returns
    -------
    results : dict
        Data loaded from file, with with: ``coordinates``, ``numbers``,
        ``pseudo_numbers``, ``obasis``, ``orb_alpha``, ``signs``. It may also contain:
        ``title``, ``orb_beta``.
    """

    def helper_coordinates(f, cunit):
        """Load element numbers and coordinates"""
        numbers = []
        pseudo_numbers = []
        coordinates = []
        while True:
            last_pos = f.tell()
            line = f.readline()
            if len(line) == 0:
                break
            words = line.split()
            if len(words) != 6:
                # Go back to previous line and stop
                f.seek(last_pos)
                break
            else:
                numbers.append(sym2num[words[0].title()])
                pseudo_numbers.append(float(words[2]))
                coordinates.append([float(words[3]), float(words[4]), float(words[5])])
        numbers = np.array(numbers, int)
        pseudo_numbers = np.array(pseudo_numbers)
        coordinates = np.array(coordinates) * cunit
        return numbers, pseudo_numbers, coordinates

    def helper_obasis(f, coordinates, pure):
        """Load the orbital basis"""
        shell_types = []
        shell_map = []
        nprims = []
        alphas = []
        con_coeffs = []

        icenter = 0
        in_atom = False
        in_shell = False
        while True:
            last_pos = f.tell()
            line = f.readline()
            if len(line) == 0:
                break
            words = line.split()
            if len(words) == 0:
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
                shell_type = str_to_shell_types(shell_label, pure.get(shell_label, False))[0]
                shell_types.append(shell_type)
                nprims.append(int(words[1]))
            elif len(words) == 2 and in_atom:
                assert in_shell
                alpha = float(words[0].replace('D', 'E'))
                alphas.append(alpha)
                con_coeff = float(words[1].replace('D', 'E'))
                con_coeffs.append(con_coeff)
            else:
                # done, go back one line
                f.seek(last_pos)
                break

        shell_map = np.array(shell_map)
        nprims = np.array(nprims)
        shell_types = np.array(shell_types)
        alphas = np.array(alphas)
        con_coeffs = np.array(con_coeffs)

        obasis = {"centers": coordinates, "shell_map": shell_map, "nprims": nprims,
                  "shell_types": shell_types, "alphas": alphas, "con_coeffs": con_coeffs}

        nbasis = shells_to_nbasis(shell_types)
        return obasis, nbasis

    def helper_coeffs(f, nbasis):
        """Load the orbital coefficients"""
        coeff_alpha = []
        ener_alpha = []
        occ_alpha = []
        coeff_beta = []
        ener_beta = []
        occ_beta = []

        new_orb = True
        icoeff = nbasis
        while True:
            line = f.readline()
            if len(line) == 0 or '[' in line:
                break
            # prepare array with orbital coefficients
            if '=' in line:
                if line.startswith(' Ene='):
                    energy = float(line[5:])
                elif line.startswith(' Spin='):
                    spin = line[6:].strip()
                elif line.startswith(' Occup='):
                    occ = float(line[7:])
                new_orb = True
            else:
                if new_orb:
                    # store col, energy and occ
                    col = np.zeros((nbasis, 1))
                    if spin.lower() == 'alpha':
                        coeff_alpha.append(col)
                        ener_alpha.append(energy)
                        occ_alpha.append(occ)
                    else:
                        coeff_beta.append(col)
                        ener_beta.append(energy)
                        occ_beta.append(occ)
                    new_orb = False
                    if icoeff < nbasis:
                        raise IOError(
                            'Too little expansions coefficients in one orbital in molden file.')
                    icoeff = 0
                words = line.split()
                if icoeff >= nbasis:
                    raise IOError('Too many expansions coefficients in one orbital in molden file.')
                col[icoeff] = float(words[1])
                icoeff += 1
                assert int(words[0]) == icoeff

        coeff_alpha = np.hstack(coeff_alpha)
        ener_alpha = np.array(ener_alpha)
        occ_alpha = np.array(occ_alpha)
        if len(coeff_beta) == 0:
            coeff_beta = None
            ener_beta = None
            occ_beta = None
        else:
            coeff_beta = np.hstack(coeff_beta)
            ener_beta = np.array(ener_beta)
            occ_beta = np.array(occ_beta)
        return (coeff_alpha, ener_alpha, occ_alpha), (coeff_beta, ener_beta, occ_beta)

    # First pass: scan the file for pure/cartesian modifiers.
    # Unfortunately, some program put this information _AFTER_ the basis
    # set specification.
    pure = {'d': False, 'f': False, 'g': False}
    with open(filename) as f:
        for line in f:
            line = line.lower()
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

    # Second pass: read all the other info.
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
    with open(filename) as f:
        line = f.readline()
        if line != '[Molden Format]\n':
            raise IOError('Molden header not found')
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if line == '[Title]':
                title = f.readline().strip()
            elif line.startswith('[Atoms]'):
                if 'au' in line.lower():
                    cunit = 1.0
                elif 'angs' in line.lower():
                    cunit = angstrom
                numbers, pseudo_numbers, coordinates = helper_coordinates(f, cunit)
            elif line == '[GTO]':
                obasis, nbasis = helper_obasis(f, coordinates, pure)
            elif line == '[STO]':
                raise NotImplementedError('Slater-type orbitals are not supported in HORTON.')
            elif line == '[MO]':
                data_alpha, data_beta = helper_coeffs(f, nbasis)
                coeff_alpha, ener_alpha, occ_alpha = data_alpha
                coeff_beta, ener_beta, occ_beta = data_beta

    if coordinates is None:
        raise IOError('Coordinates not found in molden input file.')
    if obasis is None:
        raise IOError('Orbital basis not found in molden input file.')
    if coeff_alpha is None:
        raise IOError('Alpha orbitals not found in molden input file.')

    if coeff_beta is None:
        orb_alpha = (nbasis, coeff_alpha.shape[1])
        orb_alpha_coeffs = coeff_alpha
        orb_alpha_energies = ener_alpha
        orb_alpha_occs = occ_alpha / 2
        orb_beta = None
    else:
        assert coeff_alpha.shape == coeff_beta.shape
        assert ener_alpha.shape == ener_beta.shape
        assert occ_alpha.shape == occ_beta.shape
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

    _fix_molden_from_buggy_codes(result, filename)

    return result


def _is_normalized_properly(obasis, permutation, orb_alpha, orb_beta, signs=None, threshold=1e-4):
    """Test the normalization of the occupied and virtual orbitals

       **Arguments:**

       obasis
            An instance of the GOBasis class.

       permutation
            The permutation of the basis functions to bring them in HORTON's
            standard ordering.

       orb_alpha
            The alpha orbitals coefficients

       orb_beta
            The beta orbitals (may be None).

       **Optional arguments:**

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


def _get_orca_signs(shell_types):
    """Return an array with sign corrections for orbitals read from ORCA.

       **Arguments:**

       obasis
            An instance of GOBasis.
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


def _get_fixed_con_coeffs(nprims, shell_types, alphas, con_coeffs, code):
    """Return corrected contraction coefficients, assuming they came from a broken QC code.

    Parameters
    ----------
    obasis: GOBasos
        The orbital basis with contraction coefficients to be corrected.

    code: str
        Name of code: 'orca', 'psi4' or 'turbomole'.

    Returns
    -------
    fixed_con_coeffs : np.ndarray, dtype=float, shape=nbasis
        Corrected contraction coefficients, or None if corrections were not applicable.
    """
    assert code in ['orca', 'psi4', 'turbomole']
    fixed_con_coeffs = con_coeffs.copy()
    iprim = 0
    corrected = False
    for ishell in range(shell_types.size):
        shell_type = shell_types[ishell]
        for ialpha in range(nprims[ishell]):
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


def _normalized_contractions(obasis_dict):
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


def _fix_molden_from_buggy_codes(result, filename):
    """Detect errors in the data loaded from a molden/mkl/... file and correct.

    Parameters
    ----------

    result: dict
        A dictionary with the data loaded in the ``load_molden`` function.

   This function can recognize erroneous files created by PSI4, ORCA and Turbomole. The
   values in `results` for the `obasis` and `signs` keys will be updated accordingly.
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


def dump_molden(filename, data):
    """Write data to a file in the molden input format.

    Parameters
    ----------
    filename : str
        The filename of the molden input file, which is an output file for this routine.
    data : IOData
        Must contain ``coordinates``, ``numbers``, ``obasis``, ``orb_alpha``. May contain
        ``title``, ``pseudo_numbers``, ``orb_beta``.
    """
    with open(filename, 'w') as f:
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
            orb_coeffs = getattr(data, 'orb_%s_coeffs' % spin)
            orb_energies = getattr(data, 'orb_%s_energies' % spin)
            orb_occupations = getattr(data, 'orb_%s_occs' % spin)
            for ifn in range(orb_coeffs.shape[1]):
                print(' Sym=     1a', file=f)
                print(' Ene= %20.14E' % orb_energies[ifn], file=f)
                print(' Spin= %s' % spin.capitalize(), file=f)
                print(' Occup= %8.6f' % (orb_occupations[ifn] * occ_scale), file=f)
                for ibasis in range(orb_coeffs.shape[0]):
                    print('%3i %20.12f' % (ibasis + 1, orb_coeffs[permutation[ibasis], ifn]), file=f)

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
