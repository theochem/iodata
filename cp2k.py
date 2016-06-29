# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
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
"""CP2K atomic wavefunctions"""

import numpy as np

from horton.gbasis.iobas import str_to_shell_types
from horton.gbasis.cext import GOBasis, fac2


__all__ = ['load_atom_cp2k']


def _get_cp2k_norm_corrections(l, alphas):
    """Compute the corrections for the normalization of the basis functions.

    This correction is needed because the CP2K atom code works with non-normalized basis
    functions. HORTON assumes Gaussian primitives are always normalized.

    Parameters
    ----------
    l : int
        The angular momentum of the (pure) basis function. (s=0, p=1, ...)
    alphas : float or np.ndarray
             The exponent or exponents of the Gaussian primitives for which the correction
             is to be computed.

    Returns
    -------
    corrections : float or np.ndarray
                  The scale factor for the expansion coefficients of the wavefunction in
                  terms of primitive Gaussians. The inverse of this correction can be
                  applied to the contraction coefficients.
    """
    expzet = 0.25*(2*l + 3)
    prefac = np.sqrt(np.sqrt(np.pi)/2.0**(l + 2)*fac2(2*l + 1))
    zeta = 2.0*alphas
    return zeta**expzet/prefac


def _read_cp2k_contracted_obasis(f):
    """Read a contracted basis set from an open CP2K ATOM output file.

    Parameters
    ----------
    f : file
        An open readable file object.

    Returns
    -------
    obasis : GOBasis
             The orbital basis read from the file.
    """
    # Load the relevant data from the file
    basis_desc = []
    for line in f:
        if line.startswith(' *******************'):
            break
        elif line[3:12] == 'Functions':
            shell_type = str_to_shell_types(line[1:2], pure=True)[0]
            a = []  # exponents (alpha)
            c = []  # contraction coefficients
            basis_desc.append((shell_type, a, c))
        else:
            values = [float(w) for w in line.split()]
            a.append(values[0])   # one exponent per line
            c.append(values[1:])  # many contraction coefficients per line

    # Convert the basis into HORTON format
    shell_map = []
    shell_types = []
    nprims = []
    alphas = []
    con_coeffs = []

    for shell_type, a, c in basis_desc:
        # get correction to contraction coefficients. CP2K uses different normalization
        # conventions.
        corrections = _get_cp2k_norm_corrections(abs(shell_type), np.array(a))
        c = np.array(c)/corrections.reshape(-1, 1)
        # fill in arrays
        for col in c.T:
            shell_map.append(0)
            shell_types.append(shell_type)
            nprims.append(len(col))
            alphas.extend(a)
            con_coeffs.extend(col)

    # Create the basis object
    coordinates = np.zeros((1, 3))
    shell_map = np.array(shell_map)
    nprims = np.array(nprims)
    shell_types = np.array(shell_types)
    alphas = np.array(alphas)
    con_coeffs = np.array(con_coeffs)
    obasis = GOBasis(coordinates, shell_map, nprims, shell_types, alphas, con_coeffs)
    return obasis


def _read_cp2k_uncontracted_obasis(f):
    """Read an uncontracted basis set from an open CP2K ATOM output file.

    Parameters
    ----------
    f : file
        An open readable file object.

    Returns
    -------
    obasis : GOBasis
             The orbital basis read from the file.
    """
    # Load the relevant data from the file
    basis_desc = []
    shell_type = None
    for line in f:
        if line.startswith(' *******************'):
            break
        elif line[3:13] == 'Exponents:':
            shell_type = str_to_shell_types(line[1:2], pure=True)[0]
        words = line.split()
        if len(words) >= 2:
            # read the exponent
            alpha = float(words[-1])
            basis_desc.append((shell_type, alpha))

    # Convert the basis into HORTON format
    shell_map = []
    shell_types = []
    nprims = []
    alphas = []
    con_coeffs = []

    # fill in arrays
    for shell_type, alpha in basis_desc:
        correction = _get_cp2k_norm_corrections(abs(shell_type), alpha)
        shell_map.append(0)
        shell_types.append(shell_type)
        nprims.append(1)
        alphas.append(alpha)
        con_coeffs.append(1.0 / correction)

    # Create the basis object
    centers = np.zeros((1, 3))
    shell_map = np.array(shell_map)
    nprims = np.array(nprims)
    shell_types = np.array(shell_types)
    alphas = np.array(alphas)
    con_coeffs = np.array(con_coeffs)
    obasis = GOBasis(centers, shell_map, nprims, shell_types, alphas, con_coeffs)
    return obasis


def _read_cp2k_obasis(f):
    """Read a basis set from an open CP2K ATOM output file."""
    f.next()         # Skip empty line
    line = f.next()  # Check for contracted versus uncontracted
    if line == ' ********************** Contracted Gaussian Type Orbitals '\
               '**********************\n':
        return _read_cp2k_contracted_obasis(f)
    elif line == ' ********************* Uncontracted Gaussian Type Orbitals '\
                 '*********************\n':
        return _read_cp2k_uncontracted_obasis(f)
    else:
        raise IOError('Could not find basis set in CP2K ATOM output.')


def _read_cp2k_occupations_energies(f, restricted):
    """Read orbital occupation numbers and energies from an open CP2K ATOM output file.

    Parameters
    ----------
    f : file
        An open readable file object.
    restricted : bool
                 Is wavefunction restricted or unrestricted?

    Returns
    -------
    oe_alpha, oe_beta : list
                        A list with orbital properties. Each element is a tuple with the
                        following info: (angular_momentum l, spin component: 'alpha' or
                        'beta', occupation number, orbital energy).
    """
    oe_alpha = []
    oe_beta = []
    empty = 0
    while empty < 2:
        line = f.next()
        words = line.split()
        if len(words) == 0:
            empty += 1
            continue
        empty = 0
        s = int(words[0])
        l = int(words[2 - restricted])
        occ = float(words[3 - restricted])
        ener = float(words[4 - restricted])
        if restricted or words[1] == 'alpha':
            oe_alpha.append((l, s, occ, ener))
        else:
            oe_beta.append((l, s, occ, ener))
    return oe_alpha, oe_beta


def _read_cp2k_orbital_coeffs(f, oe):
    """Read the expansion coefficients of the orbital from an open CP2K ATOM output.

    Parameters
    ----------
    f : file
        An open readable file object.
    oe : list
         The orbital occupation numbers and energies read with
         ``_read_cp2k_occupations_energies``.

    Returns
    -------
    result : dict
             Key is an (l, s) pair and value is an array with orbital coefficients.
    """
    coeffs = {}
    f.next()
    while len(coeffs) < len(oe):
        line = f.next()
        assert line.startswith("    ORBITAL      L =")
        words = line.split()
        l = int(words[3])
        s = int(words[6])
        c = []
        while True:
            line = f.next()
            if len(line.strip()) == 0:
                break
            c.append(float(line))
        coeffs[(l, s)] = np.array(c)
    return coeffs


def _get_norb_nel(oe):
    """Return number of orbitals and electrons.

    Parameters
    ----------
    oe : list
         The orbital occupation numbers and energies read with
         ``_read_cp2k_occupations_energies``.
    """
    norb = 0
    nel = 0
    for row in oe:
        norb += 2*row[0] + 1
        nel += row[2]
    return norb, nel


def _fill_exp(exp, oe, coeffs, shell_types, restricted):
    """Fill in orbital coefficients, energies and occupation numbers in ``exp``.

    Parameters
    ----------
    exp : DenseExpansion
          An object to represent the orbitals
    oe : list
         The orbital occupation numbers and energies read with
         ``_read_cp2k_occupations_energies``.
    coeffs : dict
             The orbital coefficients read with ``_read_cp2k_orbital_coeffs``.
    shell_types : np.ndarray
                  The array with shell types of the GOBasis instance.
    restricted : bool
                 Is wavefunction restricted or unrestricted?
    """
    # Find the offsets for each angular momentum
    offset = 0
    offsets = []
    ls = abs(shell_types)
    for l in sorted(set(ls)):
        offsets.append(offset)
        offset += (2*l + 1)*(l == ls).sum()
    del offset

    # Fill in the coefficients
    iorb = 0
    for l, s, occ, ener in oe:
        cs = coeffs.get((l, s))
        stride = 2*l + 1
        for m in xrange(-l, l+1):
            im = m + l
            exp.energies[iorb] = ener
            exp.occupations[iorb] = occ/float((restricted + 1)*(2*l + 1))
            for ic in xrange(len(cs)):
                exp.coeffs[offsets[l] + stride*ic + im, iorb] = cs[ic]
            iorb += 1


def load_atom_cp2k(filename, lf):
    """Load data from a CP2K ATOM computation.

    Parameters
    ---------

    filename : str
               The name of the cp2k out file
    lf : LinalgFactory
         A linear-algebra factory.

    Returns
    -------
    results : dict
              Contains: ``obasis``, ``exp_alpha``, ``coordinates``, ``numbers``,
              ``energy``, ``pseudo_numbers``. May contain: ``exp_beta``.


    Notes
    -----

    This function assumes that the following subsections are present in the CP2K
    ATOM input file, in the section ``ATOM%PRINT``:

    .. code-block:: text

      &PRINT
        &POTENTIAL
        &END POTENTIAL
        &BASIS_SET
        &END BASIS_SET
        &ORBITALS
        &END ORBITALS
      &END PRINT
    """
    with open(filename) as f:
        # Find the element number
        number = None
        for line in f:
            if line.startswith(' Atomic Energy Calculation'):
                number = int(line[-5:-1])
                break
        if number is None:
            raise IOError('Could not find atomic number in CP2K ATOM output: %s.' % filename)

        # Go to the all-electron basis set and read it.
        for line in f:
            if line.startswith(' All Electron Basis'):
                break
        ae_obasis = _read_cp2k_obasis(f)

        # Go to the pseudo basis set and read it.
        for line in f:
            if line.startswith(' Pseudopotential Basis'):
                break
        pp_obasis = _read_cp2k_obasis(f)

        # Search for (un)restricted
        restricted = None
        for line in f:
            if line.startswith(' METHOD    |'):
                if 'U' in line:
                    restricted = False
                    break
                elif 'R' in line:
                    restricted = True
                    break

        # Search for the core charge (pseudo number)
        pseudo_number = None
        for line in f:
            if line.startswith('          Core Charge'):
                pseudo_number = float(line[70:])
                assert pseudo_number == int(pseudo_number)
                break
            elif line.startswith(' Electronic structure'):
                pseudo_number = float(number)
                break
        if pseudo_number is None:
            raise IOError('Could not find effective core charge in CP2K ATOM output:'
                          ' %s' % filename)

        # Select the correct basis
        if pseudo_number == number:
            obasis = ae_obasis
        else:
            obasis = pp_obasis
        if lf.default_nbasis is not None and lf.default_nbasis != obasis.nbasis:
            raise IOError('The value of lf.default_nbasis does not match nbasis '
                          'reported in CP2K ATOM output: %s' % filename)
        lf.default_nbasis = obasis.nbasis

        # Search for energy
        for line in f:
            if line.startswith(' Energy components [Hartree]           Total Energy ::'):
                energy = float(line[60:])
                break

        # Read orbital energies and occupations
        for line in f:
            if line.startswith(' Orbital energies'):
                break
        f.next()
        oe_alpha, oe_beta = _read_cp2k_occupations_energies(f, restricted)

        # Read orbital expansion coefficients
        line = f.next()
        if (line != " Atomic orbital expansion coefficients [Alpha]\n") and \
           (line != " Atomic orbital expansion coefficients []\n"):
            raise IOError('Could not find orbital coefficients in CP2K ATOM output: '
                          '%s' % filename)
        coeffs_alpha = _read_cp2k_orbital_coeffs(f, oe_alpha)

        if not restricted:
            line = f.next()
            if line != " Atomic orbital expansion coefficients [Beta]\n":
                raise IOError('Could not find beta orbital coefficient in CP2K ATOM '
                              'output: %s' % filename)
            coeffs_beta = _read_cp2k_orbital_coeffs(f, oe_beta)

        # Turn orbital data into a HORTON orbital expansions
        if restricted:
            norb, nel = _get_norb_nel(oe_alpha)
            assert nel % 2 == 0
            exp_alpha = lf.create_expansion(obasis.nbasis, norb)
            exp_beta = None
            _fill_exp(exp_alpha, oe_alpha, coeffs_alpha, obasis.shell_types, restricted)
        else:
            norb_alpha = _get_norb_nel(oe_alpha)[0]
            norb_beta = _get_norb_nel(oe_beta)[0]
            assert norb_alpha == norb_beta
            exp_alpha = lf.create_expansion(obasis.nbasis, norb_alpha)
            exp_beta = lf.create_expansion(obasis.nbasis, norb_beta)
            _fill_exp(exp_alpha, oe_alpha, coeffs_alpha, obasis.shell_types, restricted)
            _fill_exp(exp_beta, oe_beta, coeffs_beta, obasis.shell_types, restricted)

    result = {
        'obasis': obasis,
        'lf': lf,
        'exp_alpha': exp_alpha,
        'coordinates': obasis.centers,
        'numbers': np.array([number]),
        'energy': energy,
        'pseudo_numbers': np.array([pseudo_number]),
    }
    if exp_beta is not None:
        result['exp_beta'] = exp_beta
    return result
