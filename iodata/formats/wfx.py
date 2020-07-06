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
"""AIM/AIMAll WFX file format.

See http://aim.tkgristmill.com/wfxformat.html
"""

from typing import TextIO
import warnings

import numpy as np

from ..docstrings import document_load_one, document_dump_one
from ..orbitals import MolecularOrbitals
from ..periodic import num2sym
from ..iodata import IOData
from ..utils import LineIterator
from ..overlap import gob_cart_normalization
from ..basis import MolecularBasis, Shell

from .wfn import build_obasis, get_mocoeff_scales, CONVENTIONS

__all__ = []

PATTERNS = ['*.wfx']


def _wfx_labels() -> tuple:
    """Build labels for wfx parser."""
    # labels of various sections in WFX file grouped based on their data type

    # section labels with string data types
    labels_str = {
        '<Title>': 'title',
        '<Keywords>': 'keywords',
        '<Model>': 'model_name',
    }
    # section labels with integer number data types
    labels_int = {
        '<Number of Nuclei>': 'num_atoms',
        '<Number of Occupied Molecular Orbitals>': 'num_occ_mo',
        '<Number of Perturbations>': 'num_perturbations',
        '<Number of Electrons>': 'num_electrons',
        '<Number of Core Electrons>': 'num_core_electrons',
        '<Number of Alpha Electrons>': 'num_alpha_electron',
        '<Number of Beta Electrons>': 'num_beta_electron',
        '<Number of Primitives>': 'num_primitives',
        '<Electronic Spin Multiplicity>': 'spin_multi',
    }
    # section labels with float number data types
    labels_float = {
        '<Net Charge>': 'charge',
        '<Energy = T + Vne + Vee + Vnn>': 'energy',
        '<Virial Ratio (-V/T)>': 'virial_ratio',
        '<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>': 'nuc_viral',
        '<Full Virial Ratio, -(V - W)/T>': 'full_virial_ratio',
    }
    # section labels with array of integer data types
    labels_array_int = {
        '<Atomic Numbers>': 'atnums',
        '<Primitive Centers>': 'centers',
        '<Primitive Types>': 'types',
        '<MO Numbers>': 'mo_numbers',  # This is constructed in parse_wfx.
    }
    # section labels with array of float data types
    labels_array_float = {
        '<Nuclear Cartesian Coordinates>': 'atcoords',
        '<Nuclear Charges>': 'nuclear_charge',
        '<Primitive Exponents>': 'exponents',
        '<Molecular Orbital Energies>': 'mo_energies',
        '<Molecular Orbital Occupation Numbers>': 'mo_occs',
        '<Molecular Orbital Primitive Coefficients>': 'mo_coeffs',
    }
    # section labels with other data types
    labels_other = {
        '<Nuclear Names>': 'nuclear_names',
        '<Molecular Orbital Spin Types>': 'mo_spins',
        '<Nuclear Cartesian Energy Gradients>': 'nuclear_gradient',
    }

    # list of tags corresponding to required sections based on WFX format specifications
    required_tags = list(labels_str) + list(labels_int) + list(labels_float)
    required_tags += list(labels_array_float) + list(labels_array_int) + list(labels_other)
    # remove tags corresponding to optional sections
    required_tags.remove('<Model>')
    required_tags.remove('<Number of Core Electrons>')
    required_tags.remove('<Electronic Spin Multiplicity>')
    required_tags.remove('<Atomic Numbers>')
    required_tags.remove('<Full Virial Ratio, -(V - W)/T>')
    required_tags.remove('<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>')
    required_tags.remove('<Nuclear Cartesian Energy Gradients>')

    return (labels_str, labels_int, labels_float, labels_array_int, labels_array_float,
            labels_other, required_tags)


def load_data_wfx(lit: LineIterator) -> dict:
    """Process loaded WFX data."""
    # get all section labels and required labels for WFX files
    lbs_str, lbs_int, lbs_float, lbs_aint, lbs_afloat, lbs_other, required_tags = _wfx_labels()
    # load sections in WFX and check required tags exists
    data = parse_wfx(lit, required_tags)

    # process raw data to convert them to proper type based on their label
    result = {}
    for key, value in data.items():
        if key in lbs_str:
            assert len(value) == 1
            result[lbs_str[key]] = value[0]
        elif key in lbs_int:
            assert len(value) == 1
            result[lbs_int[key]] = int(value[0])
        elif key in lbs_float:
            assert len(value) == 1
            result[lbs_float[key]] = float(value[0])
        elif key in lbs_afloat:
            result[lbs_afloat[key]] = np.fromstring(" ".join(value), dtype=np.float, sep=" ")
        elif key in lbs_aint:
            result[lbs_aint[key]] = np.fromstring(" ".join(value), dtype=np.int, sep=" ")
        elif key in lbs_other:
            result[lbs_other[key]] = value
        else:
            warnings.warn("Not recognized section label, skip {0}".format(key))

    # reshape some arrays
    result['atcoords'] = result['atcoords'].reshape(-1, 3)
    result['mo_coeffs'] = result['mo_coeffs'].reshape(result['num_primitives'], -1, order='F')
    # process nuclear gradient, if present
    if 'nuclear_gradient' in result:
        gradient_mix = np.array([i.split() for i in result.pop('nuclear_gradient')]).reshape(-1, 4)
        gradient_atoms = gradient_mix[:, 0].astype(np.unicode_)
        index = [result['nuclear_names'].index(atom) for atom in gradient_atoms]
        result['atgradient'] = np.full((len(result['nuclear_names']), 3), np.nan)
        result['atgradient'][index] = gradient_mix[:, 1:].astype(float)
    # check keywords & number of perturbations
    perturbation_check = {'GTO': 0, 'GIAO': 3, 'CGST': 6}
    key = result['keywords']
    num = result['num_perturbations']
    if key not in perturbation_check.keys():
        lit.error(f"The keywords is {key}, but it should be either GTO, GIAO or CGST")
    if num != perturbation_check[key]:
        lit.error(f"Number of perturbations of {key} is {num}, expected {perturbation_check[key]}")
    return result


def parse_wfx(lit: LineIterator, required_tags: list = None) -> dict:
    """Load data in all sections existing in the given WFX file LineIterator."""
    data = {}
    mo_start = "<Molecular Orbital Primitive Coefficients>"
    section_start = None

    while True:
        # get a new line
        try:
            line = next(lit).strip()
        except StopIteration:
            break
        # check whether line is the start of a section
        if section_start is None and line.startswith("<"):
            # set start & end of the section and add it to data dictionary
            section_start = line
            if section_start in data.keys():
                lit.error("Section with tag={} is repeated!".format(section_start))
            data[section_start] = []
            section_end = line[:1] + "/" + line[1:]
            # special handling of <Molecular Orbital Primitive Coefficients> section
            if section_start == mo_start:
                data['<MO Numbers>'] = []
        # check whether line is the (correct) end of the section
        elif section_start is not None and line.startswith("</"):
            # In some cases, closing tags have a different number of spaces. 8-[
            if line.replace(" ", "") != section_end.replace(" ", ""):
                lit.error("Expecting line {} but got {}.".format(section_end, line))
            # reset section_start variable to signal that section ended
            section_start = None
        # handle <MO Number> line under <Molecular Orbital Primitive Coefficients> section
        elif section_start == mo_start and line == '<MO Number>':
            # add MO Number to list
            data['<MO Numbers>'].append(next(lit).strip())
            # skip '</MO Number>' line
            next(lit)
        # add section content to the corresponding list in data dictionary
        else:
            data[section_start].append(line)

    # check if last section was closed
    if section_start is not None:
        lit.error("Section {} is not closed at end of file.".format(section_start))
    # check required section tags
    if required_tags is not None:
        for section_tag in required_tags:
            if section_tag not in data.keys():
                lit.error(f'Section {section_tag} is missing from loaded WFX data.')
    return data


@document_load_one("WFX", ['atcoords', 'atgradient', 'atnums', 'energy',
                           'exrtra', 'mo', 'obasis', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # get data contained in WFX file with the proper type & shape
    data = load_data_wfx(lit)

    # Build molecular basis
    # ---------------------
    # build molecular basis and permutation needed to regroup shells
    obasis, permutation = build_obasis(
        data['centers'] - 1, data['types'] - 1, data['exponents'], lit)

    # Build molecular orbitals
    # ------------------------
    # re-order MO coefficients because the loaded expansion coefficients from WFX typically
    # corresponds to basis sets grouped based on their type; that is, all MO coefficients of px
    # basis functions are listed together first, then MO coefficients of py basis functions, and
    # finally MO coefficients of pz (the same is true for higher angular momentum). However, IOData
    # stores basis shells (instead of basis functions), so the p shell with angmom=1 represents
    # the px, py, pz basis functions. These shells are used by MolecularBasis (obasis) in
    # constructing the basis function. If that is the case for the loaded MO coefficients from WFX,
    # they need to be permuted to match obasis expansion of basis set (i.e. to appear in shells).
    data['mo_coeffs'] = data['mo_coeffs'][permutation]
    # fix normalization because the loaded expansion coefficients from WFX corresponds to
    # un-normalized primitives for each normalized MO (which means the primitive normalization
    # constants has been included in the MO coefficients). However, IOData expects normalized
    # primitives (either L2 or L1 as recorded in MolecularBasis primitive types), so we need to
    # divide the MO coefficients by the primitive normalization constants to have them correspond
    # to expansion coefficients for normalized primitives. Here, we assume primitives are
    # L2-normalized (as stored in obasis.primitive_normalization) which is used in scaling MO
    # coefficients to be stored in MolecularOrbitals instance.
    data['mo_coeffs'] /= get_mocoeff_scales(obasis).reshape(-1, 1)

    # process mo_spins and convert it into restricted or unrestricted & count alpha/beta orbitals
    # we do not using the <Model> section for this because it is not guaranteed to be present

    # check whether restricted case with "Alpha and Beta" in mo_spins
    if any("and" in word for word in data['mo_spins']):
        # count number of alpha & beta molecular orbitals
        norbb = data['mo_spins'].count("Alpha and Beta")
        norba = norbb + data['mo_spins'].count("Alpha")
        # check that mo_spin list contains no surprises
        if data['mo_spins'] != ["Alpha and Beta"] * norbb + ["Alpha"] * (norba - norbb):
            lit.error("Unsupported <Molecular Orbital Spin Types> values.")
        if norba != data['mo_coeffs'].shape[1]:
            lit.error("Number of orbitals inconsistent with orbital spin types.")
        # create molecular orbitals, which requires knowing the number of alpha and beta molecular
        # orbitals. These are expected to be the same for 'restricted' case, however, the number of
        # Alpha/Beta counts might not be the same for the restricted WFX (e.g., restricted
        # open-shell calculations that do not print virtual orbitals), so it is safer to use
        # `norba` to denote number of both alpha and beta orbitals in MolecularOrbitals.
        # See orbitals.py for details to see how number of orbitals are dealt with.
        # For restricted wavefunctions, IOData uses the
        # occupation numbers to identify the spin types. IOData also has different
        # conventions for norba and norbb, see orbitals.py for details.
        mo = MolecularOrbitals(
            "restricted", norba, norba,  # This is not a typo!
            data['mo_occs'], data['mo_coeffs'], data['mo_energies'], None)

    # unrestricted case with "Alpha" and "Beta" in mo_spins
    else:
        norba = data['mo_spins'].count("Alpha")
        norbb = data['mo_spins'].count("Beta")
        # check that mo_spin list contains no surprises
        if data['mo_spins'] != ["Alpha"] * norba + ["Beta"] * norbb:
            lit.error("Unsupported molecular orbital spin types.")
        # check that number of orbitals match number of MO coefficients
        if norba + norbb != data['mo_coeffs'].shape[1]:
            lit.error("Number of orbitals inconsistent with orbital spin types.")
        # Create orbitals. For unrestricted wavefunctions, IOData uses the same
        # conventions as WFX.
        mo = MolecularOrbitals(
            "unrestricted", norba, norbb,
            data['mo_occs'], data['mo_coeffs'], data['mo_energies'], None)

    # prepare WFX-specific data for IOData
    extra_labels = ['keywords', 'model_name', 'num_perturbations', 'num_core_electrons',
                    'spin_multi', 'virial_ratio', 'nuc_viral', 'full_virial_ratio', 'mo_spin']
    extra = {label: data.get(label, None) for label in extra_labels}
    extra["permutations"] = permutation

    return {
        'atcoords': data['atcoords'],
        'atgradient': data.get('atgradient'),
        'atnums': data['atnums'],
        'energy': data['energy'],
        'extra': extra,
        'mo': mo,
        'obasis': obasis,
        'title': data['title'],
    }


# todo: check document_dump_one
@document_dump_one("WFX", ['atcoords', 'atgradient', 'atnums', 'energy',
                           'exrtra', 'mo', 'obasis', 'title'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # all the labels
    lbs_str, lbs_int, lbs_float, lbs_aint, lbs_afloat, lbs_other, required_tags = _wfx_labels()
    labels_all = {**lbs_str, **lbs_int, **lbs_float, **lbs_aint, **lbs_afloat, **lbs_other}
    # the required_tags for WFX files
    # required = [labels_all[tag] for tag in required_tags]
    # flip keys and values for further use
    labels_all = {v: k for k, v in labels_all.items()}

    # Creating new obasis in primitive basis set for contracted basis set IOData objects (e.g fchk)
    if data.obasis.shells[0].exponents.shape[0] > 1:
        shells = []
        for shell in data.obasis.shells:
            for i in range(shell.coeffs.shape[1]):
                for j in range(shell.coeffs.shape[0]):
                    shells.append(Shell(shell.icenter, [shell.angmoms[i]], ['c'],
                                        np.array([shell.exponents[j]]),
                                        np.array([shell.coeffs[j][i]])))

        new_obasis = MolecularBasis(shells, CONVENTIONS, 'L2')
        obasis = new_obasis
        data.extra.setdefault("keywords", 'GTO')
        data.extra.setdefault("num_perturbations", 0)
        data.extra.setdefault("model_name", data.lot)
        data.extra.setdefault("spin_multi", int(data.spinpol + 1))
        data.extra.setdefault("virial_ratio", np.nan)
        data.extra.setdefault("nuc_viral", None)
        data.extra.setdefault("full_virial_ratio", None)
        data.extra.setdefault("num_core_electrons", None)
    else:
        obasis = data.obasis

    # write string labels
    # _write_string(data=data, labels_all=labels_all, file=f)
    title = data.title or '<Created with IOData>'
    _write_xml_single(tag=labels_all["title"], info=title, file=f)

    keywords = data.extra["keywords"]
    _write_xml_single(tag=labels_all["keywords"], info=keywords, file=f)

    # number of atoms
    num_atoms = data.natom
    _write_xml_single(tag=labels_all["num_atoms"], info=num_atoms, file=f)

    num_prim = obasis.nbasis
    _write_xml_single(tag=labels_all["num_primitives"], info=num_prim, file=f)

    num_mo = data.mo.occs.shape[0]
    _write_xml_single(tag=labels_all["num_occ_mo"], info=num_mo, file=f)

    num_perturbations = data.extra["num_perturbations"]
    _write_xml_single(tag=labels_all["num_perturbations"], info=num_perturbations, file=f)

    # Nuclear names
    nuclear_names = [num2sym[i] for i in data.atnums]
    _write_xml_iterator(tag=labels_all["nuclear_names"], info=nuclear_names, file=f)

    # atomic  numbers
    atnums = data.atnums
    _write_xml_iterator(tag=labels_all["atnums"], info=atnums, file=f)

    # nuclear charges
    nuclear_charges = data.atcharges
    _write_xml_iterator(tag=labels_all["nuclear_charge"], info=nuclear_charges, file=f)

    # nuclear cartesian coordinates
    atcoords = data.atcoords
    print("<Nuclear Cartesian Coordinates>", file=f)
    for x in range(len(data.atnums)):
        print('{: ,.14E} {: ,.14E} {: ,.14E}'.format(atcoords[x][0],
                                                       atcoords[x][1], atcoords[x][2]), file=f)
    print("</Nuclear Cartesian Coordinates>", file=f)

    # net charge
    net_charge = data.charge
    _write_xml_single_scientific(tag=labels_all["charge"], info=net_charge, file=f)

    # number of electrons
    nelect = data.nelec
    _write_xml_single(tag=labels_all["num_electrons"], info=int(round(nelect)), file=f)

    # number of alpha electrons
    num_alpha_elec = data.mo.occsa[data.mo.occsa > 0.5]
    num_alpha_elec = sum(num_alpha_elec)
    _write_xml_single(tag=labels_all["num_alpha_electron"], info=int(round(num_alpha_elec)), file=f)

    # number of beta electrons
    num_beta_elec = data.mo.occsb[data.mo.occsb > 0.5]
    num_beta_elec = sum(num_beta_elec)
    _write_xml_single(tag=labels_all["num_beta_electron"], info=int(round(num_beta_elec)), file=f)

    # molecular orbital spin types
    if data.extra["spin_multi"] is not None:
        mo_spin = data.extra["spin_multi"]
        _write_xml_single(tag=labels_all["spin_multi"], info=mo_spin, file=f)

    model = data.extra["model_name"]
    _write_xml_single(tag=labels_all["model_name"], info=model, file=f)

    # Primitive centers
    prim_centers = []
    for shell in obasis.shells:
        rang = len(obasis.conventions[shell.angmoms[0], 'c'])
        prim_centers.extend([(shell.icenter + 1) for ao in range(rang)])

    print("<Primitive Centers>", file=f)
    for j in range(0, len(prim_centers), 10):
        print(' '.join(['{:d}'.format(c) for c in prim_centers[j:j + 10]]), file=f)
    print("</Primitive Centers>", file=f)

    # Primitive types
    raw_types = [shell.angmoms[0] for shell in obasis.shells]
    ran_0 = [len(obasis.conventions[angmom, 'c']) for angmom in raw_types]
    ran_1 = [sum([len(obasis.conventions[x, 'c']) for x in range(ang+1)]) for ang in raw_types]

    prim_types = []
    for elem in zip(ran_0, ran_1):
        if elem[0] != elem[1]:
            prim_types.extend(range((elem[1]-elem[0])+1, elem[1]+1))
        else:
            prim_types.append(1)

    print("<Primitive Types>", file=f)
    for j in range(0, len(prim_types), 10):
        print(' '.join(['{:d}'.format(c) for c in prim_types[j:j + 10]]), file=f)
    print("</Primitive Types>", file=f)

    # primitive exponents
    exponents = []
    for shell in obasis.shells:
        rang = len(obasis.conventions[shell.angmoms[0], 'c'])
        exponents.extend([shell.exponents[0] for ex in range(rang)])
    print("<Primitive Exponents>", file=f)
    for j in range(0, len(exponents), 4):
        print(' '.join(['{: ,.14E}'.format(e) for e in exponents[j:j + 4]]), file=f)
    print("</Primitive Exponents>", file=f)

    # molecular orbital occupation numbers
    mo_occs = data.mo.occs
    _write_xml_iterator_scientific(tag=labels_all["mo_occs"], info=mo_occs, file=f)

    # molecular orbital energies
    mo_energies = data.mo.energies
    _write_xml_iterator_scientific(tag=labels_all["mo_energies"], info=mo_energies, file=f)

    # Molecular orbital spin types
    print("<Molecular Orbital Spin Types>", file=f)
    if data.mo.kind == 'restricted':
        for mo in data.mo.occs:
            print('Alpha and Beta ', file=f)
    else:
        for mo in data.mo.occsa:
            print('Alpha', file=f)
        for mo in data.mo.occsb:
            print('Beta', file=f)

    print("</Molecular Orbital Spin Types>", file=f)

    # Molecular Orbital Primitive Coefficients
    if data.obasis.shells[0].exponents.shape[0] > 1:
        shell_info = []
        raw_data = []
        raw_scales = []
        for shell in data.obasis.shells:
            # Creating a list with information of fchk shells. Mostly to get sp split into
            # two different shells. Instead of doing this maybe I can create a new obasis object
            for item in zip(shell.angmoms, shell.kinds, shell.coeffs.T):
                shell_info.append(list(item))
                shell_info[-1].append(shell.exponents)
                shell_info[-1].append(shell.icenter)

        for s in shell_info:
            # Store for each AO its primitives
            rang = len(data.obasis.conventions[s[0], 'c'])
            raw_data.extend(list(s[2]) for _ in range(rang))
            # Copied directly from get wfn.py get_mocoeff_scales. Getting normalization constants
            # for each primitive of each shell
            for name in data.obasis.conventions[(s[0], 'c')]:
                if name == '1':
                    nx, ny, nz = 0, 0, 0
                else:
                    nx = name.count('x')
                    ny = name.count('y')
                    nz = name.count('z')
                raw_scales.append(list(gob_cart_normalization(s[3], np.array([nx, ny, nz]))))

        #Creating arrays for AO-primitive coefficients and normalization constants
        max_prim = len(max(raw_data, key=len))
        prim_coeff = np.zeros((data.obasis.nbasis, len(max(raw_data, key=len))))
        scales = np.zeros((data.obasis.nbasis, len(max(raw_data, key=len))))
        for index in range(len(raw_data)):
            zero = (max_prim - len(raw_data[index])) * ['nan']
            elem_prim = raw_data[index] + zero
            elem_scales = raw_scales[index] + zero
            prim_coeff[index] = elem_prim
            scales[index] = elem_scales

        # Un-normalizing molecular coefficients
        raw_coeffs = data.mo.coeffs.T[:, :, None] * (prim_coeff * scales)
        #raw_coeffs = raw_coeffs[:len(data.mo.occs[data.mo.occs > 0.5])]
        raw_coeffs = raw_coeffs.reshape(len(data.mo.occs), -1)

        # Masking 'nan' values
        coeffs_data = []
        for row in raw_coeffs:
            coeffs_data.append(row[row == np.ma.masked_invalid(row)])

    else:
        coeffs_data = np.copy(data.mo.coeffs)
        coeffs_data *= get_mocoeff_scales(obasis).reshape(-1,1)
        coeffs_data = coeffs_data[data.extra["permutations"]]
        coeffs_data = coeffs_data.T

    print("<Molecular Orbital Primitive Coefficients>", file=f)
    for mo in range(len(data.mo.occs)):
        print("<MO Number>", file=f)
        print(str(mo + 1), file=f)
        print("</MO Number>", file=f)
        for j in range(0, num_prim, 4):
            print(' '.join(['{: ,.14E}'.format(c) for c in coeffs_data[mo][j:j + 4]]), file=f)
    print("</Molecular Orbital Primitive Coefficients>", file=f)

    # energy
    energy = data.energy
    _write_xml_single_scientific(tag=labels_all["energy"], info=energy, file=f)

    # virial ratio
    virial_ratio = data.extra["virial_ratio"]
    _write_xml_single_scientific(tag=labels_all["virial_ratio"], info=virial_ratio, file=f)

    #Nuclear Cartesian Energy Gradients
    if isinstance(data.atgradient, np.ndarray):
        nuc_cart_energy_grad = list(zip(nuclear_names,  data.atgradient))
        print("<Nuclear Cartesian Energy Gradients>", file=f)
        for atom in nuc_cart_energy_grad:
            print(atom[0], '{: ,.14E} {: ,.14E} {: ,.14E}'.format(atom[1][0], atom[1][1], atom[1][2]), file=f)
        print("</Nuclear Cartesian Energy Gradients>", file=f)

    # nuclear virial of energy gradient based forces on nuclei
    if data.extra["nuc_viral"] is not None:
        nuc_viral = data.extra["nuc_viral"]
        _write_xml_single_scientific(tag=labels_all["nuc_viral"], info=nuc_viral, file=f)

    # full virial ratio
    if data.extra["full_virial_ratio"] is not None:
        full_virial_ratio = data.extra["full_virial_ratio"]
        _write_xml_single_scientific(tag=labels_all["full_virial_ratio"], info=full_virial_ratio, file=f)

    # number of core electrons
    if data.extra["num_core_electrons"] is not None:
        num_core_electrons = data.extra["num_core_electrons"]
        _write_xml_single(tag=labels_all["num_core_electrons"], info=num_core_electrons, file=f)


def _write_xml_single(tag: str, info: str, file: TextIO, end: str = "\n") -> None:
    """Write header, tail and the data between them into the file."""
    print(tag, file=file)
    print(info, end=end, file=file)
    tail = '</' + tag.lstrip('<')
    print(tail, file=file)


        # todo: check precision
def _write_xml_single_scientific(tag: str, info: str, file: TextIO, end: str = "\n") -> None:
    """Write header, tail and the data between them into the file."""
    print(tag, file=file)
    print('{: ,.14E}'.format(info), end=end, file=file)
    tail = '</' + tag.lstrip('<')
    print(tail, file=file)


def _write_xml_iterator(tag: str, info: str, file: TextIO, end: str = "\n") -> None:
    """Write list of arrays to file."""
    print(tag, file=file)
    # for list or 1d array works
    for info_line in info:
        print(info_line, end=end, file=file)
    tail = '</' + tag.lstrip('<')
    print(tail, file=file)


    # todo: check precision
def _write_xml_iterator_scientific(tag: str, info: str, file: TextIO, end: str = "\n") -> None:
    """Write list of arrays to file."""
    print(tag, file=file)
    # for list or 1d array works
    for info_line in info:
        print('{: ,.14E}'.format(info_line), end=end, file=file)
    tail = '</' + tag.lstrip('<')
    print(tail, file=file)


    # write array fload labels

    # write other labels

  #  # molecular orbital spin types
   # mo_spin = data.extra["mo_spin"]
   # _write_xml_iterator(tag=labels_all["mo_spin"], info=mo_spin, file=f)
    ## nuclear gradient
   # nuclear_gradient0 = data.atgradient
   # atom_list = np.array([atom + str(idx) for idx, atom in enumerate(nuclear_names)]).reshape(-1, 1)
   # nuclear_gradient = np.concatenate((atom_list, nuclear_gradient0), axis=1)
    #for line in nuclear_gradient:
   #     print(' '.join(line), file=f)
    #

    # todo: check if this is right
    # http://www.computationalscience.org/ccce/Lesson2/Notebook%202%20Lecture.pdf
    # number of primitives
    # spin multiplicity
 #   spin_multi = data.mo.spinol * 2 + 1
 #   _write_xml_single(tag=labels_all["spin_multi"], info=spin_multi, file=f)

    # primitive centers
  #  centers = [shell.icenter + 1 for idx, shell in enumerate(data.obasis.shells)]
  #  print(labels_all["centers"], file=f)
  #  # todo: this is not the best way
  #  for info_line, idx in enumerate(centers):
  #      if idx % 10 == 0:
  #          print(info_line, end="\n", file=f)
  #      else:
  #          print(info_line, end=" ", file=f)

   # tail = '</' + labels_all["centers"].lstrip('<')
   # print(tail, file=f)

    # primitive exponents
    # todo: unit conversions
    # todo: this is not the best way
#    for info_line, idx in enumerate(exponents):
#        if idx % 3 == 0:
#            print(info_line, end="\n", file=f)
#        else:
#            print(info_line, end=" ", file=f)

#    tail = '</' + labels_all["exponents"].lstrip('<')
#    print(tail, file=f)










# def _write_string(data: IOData, labels_all: dict,  file: TextIO) -> None:
#     """Write string data into the file."""
#     title = data.title or '<Created with IOData>'
#     _write_xml_single(tag=labels_all["title"], info=title, file=file)
#     # keywords
#     keywords = data.extra["keywords"]
#     _write_xml_single(tag=labels_all["keywords"], info=keywords, file=file)
#     # model name
#     model = data.extra["model_name"]
#     _write_xml_single(tag=labels_all["model_name"], info=model, file=file)
