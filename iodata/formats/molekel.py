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
"""Molekel file format.

This format is used by two programs:
`Molekel <http://ugovaretto.github.io/molekel/wiki/pmwiki.php/Main/HomePage.html>`_ and
`Orca <https://sites.google.com/site/orcainputlibrary/>`_.
"""

from typing import Tuple, List, TextIO

import numpy as np

from .molden import CONVENTIONS, _fix_molden_from_buggy_codes
from ..basis import angmom_sti, angmom_its, convert_conventions, MolecularBasis, Shell
from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.mkl']


def _load_helper_charge_spinpol(lit: LineIterator) -> List[int]:
    charge, spinmult = [int(word) for word in next(lit).split()]
    spinpol = spinmult - 1
    return charge, spinpol


def _load_helper_charges(lit: LineIterator) -> dict:
    atcharges = []
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        atcharges.append(float(line))

    return {'mulliken': np.array(atcharges)}


def _load_helper_atoms(lit: LineIterator) -> Tuple[np.ndarray, np.ndarray]:
    atnums = []
    atcoords = []
    for line in lit:
        if line.strip() == '$END':
            break
        words = line.split()
        atnums.append(int(words[0]))
        atcoords.append([float(words[1]), float(words[2]), float(words[3])])
    atnums = np.array(atnums, int)
    atcoords = np.array(atcoords) * angstrom
    return atnums, atcoords


def _load_helper_obasis(lit: LineIterator) -> MolecularBasis:
    shells = []
    icenter = 0
    while True:
        line = next(lit).strip()
        if line == '$END':
            break
        if line == "":
            continue
        if line == '$$':
            icenter += 1
            continue
        # Shell header, always assuming pure functions
        words = line.split()
        angmom = angmom_sti(words[1])
        nbasis_shell = int(words[0])
        if nbasis_shell == len(CONVENTIONS[(angmom, 'c')]):
            kind = 'c'
        elif nbasis_shell == len(CONVENTIONS[(angmom, 'p')]):
            kind = 'p'
        else:
            lit.error('Cannot interpret angmom={} with nbasis_shell={}'.format(
                angmom, nbasis_shell))
        exponents = []
        coeffs = []
        for line in lit:
            words = line.split()
            if len(words) != 2:
                lit.back(line)
                break
            exponents.append(float(words[0]))
            coeffs.append([float(words[1])])
        shells.append(Shell(icenter, [angmom], [kind], np.array(exponents), np.array(coeffs)))
    return MolecularBasis(shells, CONVENTIONS, 'L2')


def _load_helper_coeffs(lit: LineIterator, nbasis: int) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = []
    energies = []
    irreps = []

    in_orb = 0
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        if in_orb == 0:
            # read a1g line
            words = line.split()
            ncol = len(words)
            assert ncol > 0
            for word in words:
                irreps.append(word)
            cols = [np.zeros((nbasis, 1), float) for _ in range(ncol)]
            in_orb = 1
        elif in_orb == 1:
            # read energies
            words = line.split()
            assert len(words) == ncol
            for word in words:
                energies.append(float(word))
            in_orb = 2
            ibasis = 0
        elif in_orb == 2:
            # read expansion coefficients
            words = line.split()
            assert len(words) == ncol
            for icol in range(ncol):
                cols[icol][ibasis] = float(words[icol])
            ibasis += 1
            if ibasis == nbasis:
                in_orb = 0
                coeffs.extend(cols)

    return np.hstack(coeffs), np.array(energies), irreps


def _load_helper_occ(lit: LineIterator) -> np.ndarray:
    occs = []
    for line in lit:
        line = line.strip()
        if line == '$END':
            break
        for word in line.split():
            occs.append(float(word))
    return np.array(occs)


# pylint: disable=too-many-branches,too-many-statements
@document_load_one("Molekel", ['atcoords', 'atnums', 'mo', 'obasis'], ['atcharges'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    charge = None
    atnums = None
    atcoords = None
    obasis = None
    coeffsa = None
    energiesa = None
    occsa = None
    coeffsb = None
    energiesb = None
    occsb = None
    atcharges = None
    irrepsa = None
    irrepsb = None
    # Using a loop because we're not entirely sure if sections in an MKL file
    # have a fixed order.
    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            # There is no file-end marker we can use, so we only stop when
            # reaching the end of the file.
            break
        if line == '$CHAR_MULT':
            charge, spinpol = _load_helper_charge_spinpol(lit)
        elif line == '$CHARGES':
            atcharges = _load_helper_charges(lit)
        elif line == '$COORD':
            atnums, atcoords = _load_helper_atoms(lit)
        elif line == '$BASIS':
            obasis = _load_helper_obasis(lit)
        elif line == '$COEFF_ALPHA':
            coeffsa, energiesa, irrepsa = _load_helper_coeffs(lit, obasis.nbasis)
        elif line == '$OCC_ALPHA':
            occsa = _load_helper_occ(lit)
        elif line == '$COEFF_BETA':
            coeffsb, energiesb, irrepsb = _load_helper_coeffs(lit, obasis.nbasis)
        elif line == '$OCC_BETA':
            occsb = _load_helper_occ(lit)

    if charge is None:
        lit.error('Charge and spin polarization not found.')
    if atcoords is None:
        lit.error('Coordinates not found.')
    if obasis is None:
        lit.error('Orbital basis not found.')
    if coeffsa is None:
        lit.error('Alpha orbitals not found.')
    if occsa is None:
        lit.error('Alpha occupation numbers not found.')

    nelec = atnums.sum() - charge
    if coeffsb is None:
        # restricted closed-shell
        assert nelec % 2 == 0
        assert abs(occsa.sum() - nelec) < 1e-7
        mo = MolecularOrbitals(
            'restricted', coeffsa.shape[1], coeffsa.shape[1],
            occsa, coeffsa, energiesa, irrepsa)
    else:
        if occsb is None:
            lit.error('Beta occupation numbers not found in mkl file while '
                      'beta orbitals were present.')
        nalpha = int(np.round(occsa.sum()))
        nbeta = int(np.round(occsb.sum()))
        assert abs(spinpol - abs(nalpha - nbeta)) < 1e-7
        assert nelec == nalpha + nbeta
        assert coeffsa.shape == coeffsb.shape
        assert energiesa.shape == energiesb.shape
        assert occsa.shape == occsb.shape
        mo = MolecularOrbitals(
            'unrestricted',
            coeffsa.shape[1],
            coeffsb.shape[1],
            np.concatenate((occsa, occsb), axis=0),
            np.concatenate((coeffsa, coeffsb), axis=1),
            np.concatenate((energiesa, energiesb), axis=0),
            irrepsa + irrepsb)

    result = {
        'atcoords': atcoords,
        'atnums': atnums,
        'obasis': obasis,
        'mo': mo,
        'atcharges': atcharges,
    }
    _fix_molden_from_buggy_codes(result, lit)
    return result


@document_dump_one("Molekel", ['atcoords', 'atnums', 'mo', 'obasis'], ['atcharges'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # Header
    f.write('$MKL\n')
    f.write('#\n')
    f.write('# MKL format file produced by IOData\n')
    f.write('#\n')

    # CHAR_MUL
    f.write('$CHAR_MULT\n')
    f.write('  {:.0f} {:.0f}\n'.format(data.charge, data.spinpol + 1))
    f.write('$END\n')
    f.write('\n')

    # COORD
    atcoords = data.atcoords / angstrom
    f.write('$COORD\n')
    for n, coord in zip(data.atnums, atcoords):
        f.write('   {:d}   {: ,.6f}  {: ,.6f}  {: ,.6f}\n'.format(n, coord[0], coord[1], coord[2]))
    f.write('$END\n')
    f.write('\n')

    # CHARGES
    if 'mulliken' in data.atcharges:
        f.write('$CHARGES\n')
        for charge in data.atcharges['mulliken']:
            f.write('  {: ,.6f}\n'.format(charge))
        f.write('$END\n')
        f.write('\n')

    # BASIS
    f.write('$BASIS\n')
    iatom_last = 0
    for shell in data.obasis.shells:
        iatom_new = shell.icenter
        if iatom_new != iatom_last:
            f.write('$$\n')
        for iangmom, (angmom, kind) in enumerate(zip(shell.angmoms, shell.kinds)):
            iatom_last = shell.icenter
            nbasis = len(CONVENTIONS[(angmom, kind)])
            f.write(' {} {:1s} 1.00\n'.format(nbasis, angmom_its(angmom).capitalize()))
            for exponent, coeff in zip(shell.exponents, shell.coeffs[:, iangmom]):
                f.write('{:20.10f} {:17.10f}\n'.format(exponent, coeff))
    f.write('\n')
    f.write('$END\n')
    f.write('\n')

    if data.mo.kind == 'restricted':
        # COEFF_ALPHA
        f.write('$COEFF_ALPHA\n')
        _dump_helper_coeffs(f, data, spin='a')

        # OCC_ALPHA
        f.write('$OCC_ALPHA\n')
        _dump_helper_occ(f, data, spin='ab')

    # Not taking into account generalized.
    elif data.mo.kind == 'unrestricted':
        # COEFF_ALPHA
        f.write('$COEFF_ALPHA\n')
        _dump_helper_coeffs(f, data, spin='a')

        # OCC_ALPHA
        f.write('$OCC_ALPHA\n')
        _dump_helper_occ(f, data, spin='a')
        f.write('\n')

        # COEFF_BETA
        f.write('$COEFF_BETA\n')
        _dump_helper_coeffs(f, data, spin='b')

        # OCC_BETA
        f.write('$OCC_BETA\n')
        _dump_helper_occ(f, data, spin='b')

    else:
        raise ValueError(f"The MKL format does not support {data.mo.kind} orbitals.")


# Defining help dumping functions
def _dump_helper_coeffs(f, data, spin=None):
    permutation, signs = convert_conventions(data.obasis, CONVENTIONS)
    if spin == 'a':
        norb = data.mo.norba
        coeff = data.mo.coeffsa[permutation] * signs.reshape(-1, 1)
        ener = data.mo.energiesa
        if data.mo.irreps is not None:
            irreps = data.mo.irreps[:norb]
        else:
            irreps = ['a1g'] * norb
    elif spin == 'b':
        norb = data.mo.norbb
        coeff = data.mo.coeffsb[permutation] * signs.reshape(-1, 1)
        ener = data.mo.energiesb
        if data.mo.irreps is not None:
            irreps = data.mo.irreps[norb:]
        else:
            irreps = ['a1g'] * norb
    else:
        raise IOError('A spin must be specified')

    for j in range(0, norb, 5):
        en = ' '.join(['   {: ,.12f}'.format(e) for e in ener[j:j + 5]])
        irre = ' '.join(['{}'.format(irr) for irr in irreps[j:j + 5]])
        f.write(irre + '\n')
        f.write(en + '\n')
        for orb in coeff[:, j:j + 5]:
            coeffs = ' '.join(['  {: ,.12f}'.format(c) for c in orb])
            f.write(coeffs + '\n')

    f.write(' $END\n')
    f.write('\n')


def _dump_helper_occ(f, data, spin=None):
    if spin == 'a':
        norb = data.mo.norba
        occ = data.mo.occsa
    elif spin == 'b':
        norb = data.mo.norbb
        occ = data.mo.occsb
    elif spin == 'ab':
        norb = data.mo.norba
        occ = data.mo.occs
    else:
        raise IOError('A spin must be specified')

    for j in range(0, norb, 5):
        occs = ' '.join(['  {: ,.7f}'.format(o) for o in occ[j:j + 5]])
        f.write(occs + '\n')
    f.write(' $END\n')
