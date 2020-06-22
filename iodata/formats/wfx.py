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

from .wfn import build_obasis, get_mocoeff_scales

__all__ = []

PATTERNS = ['*.wfx']


def _wfx_labels() -> tuple:
    """Build labels for wfx parser."""
    labels_str = {
        '<Title>': 'title',
        '<Keywords>': 'keywords',
        '<Model>': 'model_name',
    }
    # integer numbers
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
    # float numbers
    labels_float = {
        '<Net Charge>': 'charge',
        '<Energy = T + Vne + Vee + Vnn>': 'energy',
        '<Virial Ratio (-V/T)>': 'virial_ratio',
        '<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>': 'nuc_viral',
        '<Full Virial Ratio, -(V - W)/T>': 'full_virial_ratio',
    }
    labels_array_int = {
        '<Atomic Numbers>': 'atnums',
        '<Primitive Centers>': 'centers',
        '<Primitive Types>': 'types',
        '<MO Numbers>': 'mo_numbers',  # This is constructed in parse_wfx.
    }
    labels_array_float = {
        '<Nuclear Cartesian Coordinates>': 'atcoords',
        '<Nuclear Charges>': 'nuclear_charge',
        '<Primitive Exponents>': 'exponents',
        '<Molecular Orbital Energies>': 'mo_energies',
        '<Molecular Orbital Occupation Numbers>': 'mo_occs',
        '<Molecular Orbital Primitive Coefficients>': 'mo_coeffs',
    }
    labels_other = {
        '<Nuclear Names>': 'nuclear_names',
        '<Molecular Orbital Spin Types>': 'mo_spins',
        '<Nuclear Cartesian Energy Gradients>': 'nuclear_gradient',
    }

    # list of required section tags
    required_tags = (
            list(labels_str) + list(labels_int) + list(labels_float)
            + list(labels_array_int) + list(labels_array_float) + list(labels_other)
    )
    required_tags.remove('<Model>')
    required_tags.remove('<Number of Core Electrons>')
    required_tags.remove('<Electronic Spin Multiplicity>')
    required_tags.remove('<Atomic Numbers>')
    required_tags.remove('<Full Virial Ratio, -(V - W)/T>')
    required_tags.remove('<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>')
    required_tags.remove('<Nuclear Cartesian Energy Gradients>')

    return labels_str, labels_int, labels_float, labels_array_int, \
           labels_array_float, labels_other, required_tags


labels_str, labels_int, labels_float, labels_array_int, \
labels_array_float, labels_other, required_tags = _wfx_labels()


def load_data_wfx(lit: LineIterator) -> dict:
    """Process loaded WFX data."""
    # load raw data & check required tags
    data = parse_wfx(lit, required_tags)

    # process raw data
    result = {}
    for key, value in data.items():
        if key in labels_str:
            result[labels_str[key]] = value[0]
        elif key in labels_int:
            result[labels_int[key]] = int(value[0])
        elif key in labels_float:
            result[labels_float[key]] = float(value[0])
        elif key in labels_array_float:
            result[labels_array_float[key]] = np.fromstring(" ".join(value),
                                                            dtype=np.float,
                                                            sep=" ")
        elif key in labels_array_int:
            result[labels_array_int[key]] = np.fromstring(" ".join(value),
                                                          dtype=np.int,
                                                          sep=" ")
        elif key in labels_other:
            result[labels_other[key]] = value
        else:
            warnings.warn("Not recognized label, skip {0}".format(key))

    # Reshape some arrays.
    result['atcoords'] = result['atcoords'].reshape(-1, 3)
    result['mo_coeffs'] = result['mo_coeffs'].reshape(result['num_primitives'], -1, order='F')
    # Process nuclear gradient, if present.
    if 'nuclear_gradient' in result:
        gradient_mix = np.array([i.split() for i in result.pop('nuclear_gradient')]).reshape(-1, 4)
        gradient_atoms = gradient_mix[:, 0].astype(np.unicode_)
        index = [result['nuclear_names'].index(atom) for atom in gradient_atoms]
        result['atgradient'] = np.full((len(result['nuclear_names']), 3), np.nan)
        result['atgradient'][index] = gradient_mix[:, 1:].astype(float)
    # Check number of perturbations.
    perturbation_check = {'GTO': 0, 'GIAO': 3, 'CGST': 6}
    if result['num_perturbations'] != perturbation_check[result['keywords']]:
        lit.error("Number of perturbations is not equal to 0, 3 or 6.")
    if result['keywords'] not in ['GTO', 'GIAO', 'CGST']:
        lit.error("The keywords should be one out of GTO, GIAO and CGST.")
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

        if section_start is None and line.startswith("<"):
            section = []
            section_start = line
            data[section_start] = section
            section_end = line[:1] + "/" + line[1:]
            # Special handling of MO coeffs
            if section_start == mo_start:
                mo_numbers = []
                data['<MO Numbers>'] = mo_numbers
        elif section_start is not None and line.startswith("</"):
            # Check if the closing tag is correct. In some cases, closing
            # tags have a different number of spaces. 8-[
            if line.replace(" ", "") != section_end.replace(" ", ""):
                lit.error("Expecting line {} but got {}.".format(section_end, line))
            section_start = None
        elif section_start == mo_start and line == '<MO Number>':
            # Special handling of MO coeffs: read mo number
            mo_numbers.append(next(lit).strip())
            next(lit)  # skip '</MO Number>'
        else:
            section.append(line)

    # check if last section was closed
    if section_start is not None:
        lit.error("Section {} is not closed at end of file.".format(section_start))
    # check required section tags
    if required_tags is not None:
        for section_tag in required_tags:
            if section_tag not in data.keys():
                lit.error(f'Section {section_tag} is missing.')
    return data


@document_load_one("WFX", ['atcoords', 'atgradient', 'atnums', 'energy',
                           'exrtra', 'mo', 'obasis', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    data = load_data_wfx(lit)
    # Build the basis set and the permutation needed to regroup shells.
    obasis, permutation = build_obasis(
        data['centers'] - 1, data['types'] - 1, data['exponents'], lit)

    # Build the molecular orbitals
    # ----------------------------
    # Re-order the mo coefficients.
    data['mo_coeffs'] = data['mo_coeffs'][permutation]
    # Fix normalization
    data['mo_coeffs'] /= get_mocoeff_scales(obasis).reshape(-1, 1)
    # Process mo_spins. Convert this into restricted or unrestricted and
    # corresponding occupation numbers. We are not using the <Model> section
    # because it is not guaranteed to be present.
    if any("and" in word for word in data['mo_spins']):
        # Restricted case.
        norbb = data['mo_spins'].count("Alpha and Beta")
        norba = norbb + data['mo_spins'].count("Alpha")
        # Check that the mo_spin list contains no surprises.
        if data['mo_spins'] != ["Alpha and Beta"] * norbb + ["Alpha"] * (norba - norbb):
            lit.error("Unsupported molecular orbital spin types.")
        if norba != data['mo_coeffs'].shape[1]:
            lit.error("Number of orbitals inconsistent with orbital spin types.")
        # Create orbitals. For restricted wavefunctions, IOData uses the
        # occupation numbers to identify the spin types. IOData also has different
        # conventions for norba and norbb, see orbitals.py for details.
        mo = MolecularOrbitals(
            "restricted", norba, norba,  # This is not a typo!
            data['mo_occs'], data['mo_coeffs'], data['mo_energies'], None)
    else:
        # unrestricted case
        norba = data['mo_spins'].count("Alpha")
        norbb = data['mo_spins'].count("Beta")
        # Check that the mo_spin list contains no surprises
        if data['mo_spins'] != ["Alpha"] * norba + ["Beta"] * norbb:
            lit.error("Unsupported molecular orbital spin types.")
        if norba + norbb != data['mo_coeffs'].shape[1]:
            lit.error("Number of orbitals inconsistent with orbital spin types.")
        # Create orbitals. For unrestricted wavefunctions, IOData uses the same
        # conventions as WFX.
        mo = MolecularOrbitals(
            "unrestricted", norba, norbb,
            data['mo_occs'], data['mo_coeffs'], data['mo_energies'], None)

    # Store WFX-specific data
    extra_labels = ['keywords', 'model_name', 'num_perturbations', 'num_core_electrons',
                    'spin_multi', 'virial_ratio', 'nuc_viral', 'full_virial_ratio', 'mo_spin']
    extra = {label: data.get(label, None) for label in extra_labels}

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
    labels_all = {**labels_str, **labels_int, **labels_float, **labels_array_int,
                  **labels_array_float, **labels_other}
    # the required_tags for WFX files
    # required = [labels_all[tag] for tag in required_tags]
    # flip keys and values for further use
    labels_all = {v: k for k, v in labels_all.items()}
    # todo: check if all mandatory properties are present (not checked)

    # write string labels
    # _write_string(data=data, labels_all=labels_all, file=f)
    title = data.title or '<Created with IOData>'
    _write_xml_single(tag=labels_all["title"], info=title, file=f)

    # keywords
    keywords = data.extra["keywords"]
    _write_xml_single(tag=labels_all["keywords"], info=keywords, file=f)

    # number of atoms
    num_atoms = data.natom
    _write_xml_single(tag=labels_all["num_atoms"], info=num_atoms, file=f)

    # number of primitives
    num_prim = data.obasis.nbasis
    _write_xml_single(tag=labels_all["num_primitives"], info=num_prim, file=f)

    # number of occupied mo
    num_mo = data.mo.occs.shape[0]
    _write_xml_single(tag=labels_all["num_occ_mo"], info=num_mo, file=f)

    # number of perturbations
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

    # todo:  spin multiplicity
    # molecular orbital spin types
    if data.extra["spin_multi"] != None:
        mo_spin = data.extra["spin_multi"]
        _write_xml_single(tag=labels_all["spin_multi"], info=mo_spin, file=f)

    # model name
    model = data.extra["model_name"]
    _write_xml_single(tag=labels_all["model_name"], info=model, file=f)

    # todo: primitive centers and types
    # Primitive centers
    prim_centers = []
    for shell in data.obasis.shells:
        rang = len(data.obasis.conventions[shell.angmoms[0], 'c'])
        prim_centers.extend([(shell.icenter + 1) for ao in range(rang)])

    print("<Primitive Centers>", file=f)
    for j in range(0, len(prim_centers), 10):
        print(' '.join(['{:d}'.format(c) for c in prim_centers[j:j + 10]]), file=f)
    print("</Primitive Centers>", file=f)

    # Primitive types
    raw_types = [shell.angmoms[0] for shell in data.obasis.shells]
    ran_0 = [len(data.obasis.conventions[angmom, 'c']) for angmom in raw_types]
    ran_1 = [sum([len(data.obasis.conventions[x, 'c']) for x in range(ang+1)]) for ang in raw_types]

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

    # todo: unit conversions
    # primitive exponents
    exponents = []
    for shell in data.obasis.shells:
        rang = len(data.obasis.conventions[shell.angmoms[0], 'c'])
        exponents.extend([shell.exponents[0] for ex in range(rang)])
    print("<Primitive Exponents>", file=f)
    print(exponents)
    for j in range(0, len(exponents), 4):
        print(' '.join(['{: ,.14E}'.format(e) for e in exponents[j:j + 4]]), file=f)
    print("</Primitive Exponents>", file=f)

    # todo: MO numbers
    # todo: number of occupied molecular orbitals
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

    # todo: primitive mo numbers and Coefficients, line 213 in wfx.py
    # Molecular Orbital Primitive Coefficients
    coeffs_data = np.copy(data.mo.coeffs)
    coeffs_data *= get_mocoeff_scales(data.obasis).reshape(-1,1)
    print("<Molecular Orbital Primitive Coefficients>", file=f)
    for mo in range(data.mo.coeffs.T.shape[0]):
        print("<MO Number>", file=f)
        print(str(mo + 1), file=f)
        print("</MO Number>", file=f)
        for j in range(0, data.obasis.nbasis, 4):
            print(' '.join(['{: ,.14E}'.format(c) for c in coeffs_data.T[mo][j:j + 4]]), file=f)
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
    if data.extra["nuc_viral"] != None:
        nuc_viral = data.extra["nuc_viral"]
        _write_xml_single_scientific(tag=labels_all["nuc_viral"], info=nuc_viral, file=f)

    # full virial ratio
    if data.extra["full_virial_ratio"] != None:
        full_virial_ratio = data.extra["full_virial_ratio"]
        _write_xml_single_scientific(tag=labels_all["full_virial_ratio"], info=full_virial_ratio, file=f)

    # number of core electrons
    if data.extra["num_core_electrons"] != None:
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
