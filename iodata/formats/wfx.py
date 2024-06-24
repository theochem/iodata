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

from collections.abc import Iterator
from typing import Optional, TextIO
from warnings import warn

import numpy as np

from ..basis import MolecularBasis, Shell, convert_conventions
from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..periodic import num2sym
from ..utils import LineIterator, LoadError, LoadWarning, PrepareDumpError
from .wfn import CONVENTIONS, build_obasis, get_mocoeff_scales

__all__ = []

PATTERNS = ["*.wfx"]


def _wfx_labels() -> tuple:
    """Build labels for wfx parser."""
    # labels of various sections in WFX file grouped based on their data type

    # section labels with string data types
    labels_str = {
        "<Title>": "title",
        "<Keywords>": "keywords",
        "<Model>": "model_name",
    }
    # section labels with integer number data types
    labels_int = {
        "<Number of Nuclei>": "num_atoms",
        "<Number of Occupied Molecular Orbitals>": "num_occ_mo",
        "<Number of Perturbations>": "num_perturbations",
        "<Number of Electrons>": "num_electrons",
        "<Number of Core Electrons>": "num_core_electrons",
        "<Number of Alpha Electrons>": "num_alpha_electron",
        "<Number of Beta Electrons>": "num_beta_electron",
        "<Number of Primitives>": "num_primitives",
        "<Electronic Spin Multiplicity>": "spin_multi",
    }
    # section labels with float number data types
    labels_float = {
        "<Net Charge>": "charge",
        "<Energy = T + Vne + Vee + Vnn>": "energy",
        "<Virial Ratio (-V/T)>": "virial_ratio",
        "<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>": "nuc_viral",
        "<Full Virial Ratio, -(V - W)/T>": "full_virial_ratio",
    }
    # section labels with array of integer data types
    labels_array_int = {
        "<Atomic Numbers>": "atnums",
        "<Primitive Centers>": "centers",
        "<Primitive Types>": "types",
        "<MO Numbers>": "mo_numbers",  # This is constructed in parse_wfx.
    }
    # section labels with array of float data types
    labels_array_float = {
        "<Nuclear Cartesian Coordinates>": "atcoords",
        "<Nuclear Charges>": "nuclear_charge",
        "<Primitive Exponents>": "exponents",
        "<Molecular Orbital Energies>": "mo_energies",
        "<Molecular Orbital Occupation Numbers>": "mo_occs",
        "<Molecular Orbital Primitive Coefficients>": "mo_coeffs",
    }
    # section labels with other data types
    labels_other = {
        "<Nuclear Names>": "nuclear_names",
        "<Molecular Orbital Spin Types>": "mo_spins",
        "<Nuclear Cartesian Energy Gradients>": "nuclear_gradient",
    }

    # list of tags corresponding to required sections based on WFX format specifications
    required_tags = list(labels_str) + list(labels_int) + list(labels_float)
    required_tags += list(labels_array_float) + list(labels_array_int) + list(labels_other)
    # remove tags corresponding to optional sections
    required_tags.remove("<Model>")
    required_tags.remove("<Number of Core Electrons>")
    required_tags.remove("<Electronic Spin Multiplicity>")
    required_tags.remove("<Atomic Numbers>")
    required_tags.remove("<Full Virial Ratio, -(V - W)/T>")
    required_tags.remove("<Nuclear Virial of Energy-Gradient-Based Forces on Nuclei, W>")
    required_tags.remove("<Nuclear Cartesian Energy Gradients>")

    return (
        labels_str,
        labels_int,
        labels_float,
        labels_array_int,
        labels_array_float,
        labels_other,
        required_tags,
    )


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
            result[lbs_afloat[key]] = np.fromstring(" ".join(value), dtype=float, sep=" ")
        elif key in lbs_aint:
            result[lbs_aint[key]] = np.fromstring(" ".join(value), dtype=int, sep=" ")
        elif key in lbs_other:
            result[lbs_other[key]] = value
        else:
            warn(LoadWarning(f"Not recognized section label, skip {key}", lit), stacklevel=2)

    # reshape some arrays
    result["atcoords"] = result["atcoords"].reshape(-1, 3)
    result["mo_coeffs"] = result["mo_coeffs"].reshape(result["num_primitives"], -1, order="F")
    # process nuclear gradient, if present
    if "nuclear_gradient" in result:
        gradient_mix = np.array([i.split() for i in result.pop("nuclear_gradient")]).reshape(-1, 4)
        gradient_atoms = gradient_mix[:, 0].astype(np.str_)
        index = [result["nuclear_names"].index(atom) for atom in gradient_atoms]
        result["atgradient"] = np.full((len(result["nuclear_names"]), 3), np.nan)
        result["atgradient"][index] = gradient_mix[:, 1:].astype(float)
    # check keywords & number of perturbations
    perturbation_check = {"GTO": 0, "GIAO": 3, "CGST": 6}
    key = result["keywords"]
    num = result["num_perturbations"]
    if key not in perturbation_check:
        raise LoadError(f"The keywords is {key}, but it should be either GTO, GIAO or CGST.", lit)
    if num != perturbation_check[key]:
        raise LoadError(
            f"Number of perturbations of {key} is {num}, expected {perturbation_check[key]}.", lit
        )
    return result


def parse_wfx(lit: LineIterator, required_tags: Optional[list] = None) -> dict:
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
            if section_start in data:
                raise LoadError(f"Section with tag={section_start} is repeated.", lit)
            data[section_start] = []
            section_end = line[:1] + "/" + line[1:]
            # special handling of <Molecular Orbital Primitive Coefficients> section
            if section_start == mo_start:
                data["<MO Numbers>"] = []
        # check whether line is the (correct) end of the section
        elif section_start is not None and line.startswith("</"):
            # In some cases, closing tags have a different number of spaces. 8-[
            if line.replace(" ", "") != section_end.replace(" ", ""):
                raise LoadError(f"Expecting line {section_end} but got {line}.", lit)
            # reset section_start variable to signal that section ended
            section_start = None
        # handle <MO Number> line under <Molecular Orbital Primitive Coefficients> section
        elif section_start == mo_start and line == "<MO Number>":
            # add MO Number to list
            data["<MO Numbers>"].append(next(lit).strip())
            # skip '</MO Number>' line
            next(lit)
        # add section content to the corresponding list in data dictionary
        else:
            data[section_start].append(line)

    # check if last section was closed
    if section_start is not None:
        raise LoadError(f"Section {section_start} is not closed at end of file.", lit)
    # check required section tags
    if required_tags is not None:
        for section_tag in required_tags:
            if section_tag not in data:
                raise LoadError(f"Section {section_tag} is missing from loaded WFX data.", lit)
    return data


@document_load_one(
    "WFX", ["atcoords", "atgradient", "atnums", "energy", "extra", "mo", "obasis", "title"]
)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # get data contained in WFX file with the proper type & shape
    data = load_data_wfx(lit)

    # Build molecular basis
    # ---------------------
    # build molecular basis and permutation needed to regroup shells
    obasis, permutation = build_obasis(
        data["centers"] - 1, data["types"] - 1, data["exponents"], lit
    )

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
    data["mo_coeffs"] = data["mo_coeffs"][permutation]
    # fix normalization because the loaded expansion coefficients from WFX corresponds to
    # un-normalized primitives for each normalized MO (which means the primitive normalization
    # constants has been included in the MO coefficients). However, IOData expects normalized
    # primitives (either L2 or L1 as recorded in MolecularBasis primitive types), so we need to
    # divide the MO coefficients by the primitive normalization constants to have them correspond
    # to expansion coefficients for normalized primitives. Here, we assume primitives are
    # L2-normalized (as stored in obasis.primitive_normalization) which is used in scaling MO
    # coefficients to be stored in MolecularOrbitals instance.
    data["mo_coeffs"] /= get_mocoeff_scales(obasis).reshape(-1, 1)

    # process mo_spins and convert it into restricted or unrestricted & count alpha/beta orbitals
    # we do not using the <Model> section for this because it is not guaranteed to be present

    # check whether restricted case with "Alpha and Beta" in mo_spins
    if any("and" in word for word in data["mo_spins"]):
        # count number of alpha & beta molecular orbitals
        norbb = data["mo_spins"].count("Alpha and Beta")
        norba = norbb + data["mo_spins"].count("Alpha")
        # check that mo_spin list contains no surprises
        if data["mo_spins"] != ["Alpha and Beta"] * norbb + ["Alpha"] * (norba - norbb):
            raise LoadError("Unsupported <Molecular Orbital Spin Types> values.", lit)
        if norba != data["mo_coeffs"].shape[1]:
            raise LoadError("Number of orbitals inconsistent with orbital spin types.", lit)
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
            "restricted",
            norba,
            norba,  # This is not a typo!
            data["mo_occs"],
            data["mo_coeffs"],
            data["mo_energies"],
        )

    # unrestricted case with "Alpha" and "Beta" in mo_spins
    else:
        norba = data["mo_spins"].count("Alpha")
        norbb = data["mo_spins"].count("Beta")
        # check that mo_spin list contains no surprises
        if data["mo_spins"] != ["Alpha"] * norba + ["Beta"] * norbb:
            raise LoadError("Unsupported molecular orbital spin types.", lit)
        # check that number of orbitals match number of MO coefficients
        if norba + norbb != data["mo_coeffs"].shape[1]:
            raise LoadError("Number of orbitals inconsistent with orbital spin types.", lit)
        # Create orbitals. For unrestricted wavefunctions, IOData uses the same
        # conventions as WFX.
        mo = MolecularOrbitals(
            "unrestricted", norba, norbb, data["mo_occs"], data["mo_coeffs"], data["mo_energies"]
        )

    # prepare WFX-specific data for IOData
    extra_labels = [
        "keywords",
        "model_name",
        "num_perturbations",
        "num_core_electrons",
        "spin_multi",
        "virial_ratio",
        "nuc_viral",
        "full_virial_ratio",
        "mo_spin",
    ]
    extra = {label: data.get(label, None) for label in extra_labels}
    extra["permutations"] = permutation

    return {
        "atcoords": data["atcoords"],
        "atgradient": data.get("atgradient"),
        "atnums": data["atnums"],
        "atcorenums": data["nuclear_charge"],
        "energy": data["energy"],
        "extra": extra,
        "mo": mo,
        "obasis": obasis,
        "title": data["title"],
    }


def prepare_dump(filename: str, data: IOData):
    """Check the compatibility of the IOData object with the WFX format.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    data
        The IOData instance to be checked.
    """
    if data.mo is None:
        raise PrepareDumpError("The WFX format requires molecular orbitals.", filename)
    if data.obasis is None:
        raise PrepareDumpError("The WFX format requires an orbital basis set.", filename)
    if data.mo.kind == "generalized":
        raise PrepareDumpError("Cannot write WFX file with generalized orbitals.", filename)
    if data.mo.occs_aminusb is not None:
        raise PrepareDumpError("Cannot write WFX file when mo.occs_aminusb is set.", filename)
    for shell in data.obasis.shells:
        if any(kind != "c" for kind in shell.kinds):
            raise PrepareDumpError(
                "The WFX format only supports Cartesian MolecularBasis.", filename
            )


@document_dump_one(
    "WFX",
    ["atcoords", "atnums", "atcorenums", "mo", "obasis", "charge"],
    ["title", "energy", "spinpol", "lot", "atgradient", "extra"],
)
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # get all tags/labels that can be written into a WFX file
    lbs_str, lbs_int, lbs_float, lbs_aint, lbs_afloat, lbs_other, _ = _wfx_labels()
    # put all labels in one dictionary and flip key and value for easier use
    lbs = {**lbs_str, **lbs_int, **lbs_float, **lbs_aint, **lbs_afloat, **lbs_other}
    lbs = {v: k for k, v in lbs.items()}

    # de-contract data.obasis
    # -----------------------
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
    # make a new instance of MolecularBasis with de-contracted basis shells; ideally for WFX we
    # want the primitive basis set, but IOData only supports shells.
    obasis = MolecularBasis(shells, data.obasis.conventions, data.obasis.primitive_normalization)

    # expand mo.coeffs in de-contracted basis primitives
    # --------------------------------------------------
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
            for _j in range(shell.nprim):
                mo_coeffs[index_mo_new : index_mo_new + n] = c
                index_mo_new += n
            index_mo_old += n
    # fix MO coefficients
    # 1) expansion coefficients in WFX correspond to un-normalized primitives, so the primitive
    # normalization constants should be included in the MO coefficients. However, IOData stores
    # normalized primitives (either L2 or L1 as recorded in MolecularBasis primitive types), so
    # we need to multiply the MO coefficients by the primitive normalization constants
    scales = get_mocoeff_scales(obasis)
    # 2) expansion coefficients in WFX represent the primitive basis coefficients, so contraction
    # coefficients needs to be multiplied by the MO expansion coefficients.
    contractions = []
    for shell in obasis.shells:
        contractions.extend(np.repeat(shell.coeffs.ravel(), [shell.nbasis], axis=0))
    contractions = np.array(contractions)
    # update MO coefficients to include primitives contraction coefficients & normalization
    for index in range(mo_coeffs.shape[1]):
        mo_coeffs[:, index] *= contractions * scales

    # write title & keywords
    _write_xml_single(tag=lbs["title"], info=data.title or "<Created with IOData>", file=f)
    _write_xml_single(tag=lbs["keywords"], info=data.extra.get("keywords", "GTO"), file=f)

    # write number of nuclei & number of primitives
    _write_xml_single(tag=lbs["num_atoms"], info=data.natom, file=f)
    _write_xml_single(tag=lbs["num_primitives"], info=obasis.nbasis, file=f)

    # write number of occupied molecular orbitals
    # in practice wfx prints the total number of MO, even though the section title specifies
    # "Number of Occupied Molecular Orbitals", which is different from total number of MO when
    # you print virtual orbitals in wfx file.
    _write_xml_single(tag=lbs["num_occ_mo"], info=data.mo.occs.shape[0], file=f)

    # write number of perturbations
    _write_xml_single(lbs["num_perturbations"], data.extra.get("num_perturbations", 0), file=f)

    # write nuclear names, atomic numbers, and nuclear charges
    # add ghost atom, represented by Bq and atomic number 0
    num2sym.update({0: "Bq"})
    nuclear_names = [f" {num2sym[num]}{index + 1}" for index, num in enumerate(data.atcorenums)]
    _write_xml_iterator(tag=lbs["nuclear_names"], info=nuclear_names, file=f)
    _write_xml_iterator(tag=lbs["atnums"], info=data.atnums, file=f)
    _write_xml_iterator_scientific(tag=lbs["nuclear_charge"], info=data.atcorenums, file=f)

    # write nuclear cartesian coordinates
    print("<Nuclear Cartesian Coordinates>", file=f)
    for item in data.atcoords:
        print(f"{item[0]: ,.14E} {item[1]: ,.14E} {item[2]: ,.14E}", file=f)
    print("</Nuclear Cartesian Coordinates>", file=f)

    # write net charge, number of electrons, number of alpha electrons, and number beta electrons
    _write_xml_single_scientific(tag=lbs["charge"], info=data.charge, file=f)
    _write_xml_single(tag=lbs["num_electrons"], info=int(data.nelec), file=f)
    # wfx expects integer values for number of alpha/beta electrons but int rounds down the float
    # so round is used before turning it to integer to get the correct number.
    _write_xml_single(tag=lbs["num_alpha_electron"], info=int(round(sum(data.mo.occsa))), file=f)
    _write_xml_single(tag=lbs["num_beta_electron"], info=int(round(sum(data.mo.occsb))), file=f)

    # write electronic spin multiplicity and model (both optional)
    if data.spinpol is not None:
        _write_xml_single(tag=lbs["spin_multi"], info=int(data.spinpol + 1), file=f)
    if data.lot is not None:
        _write_xml_single(tag=lbs["model_name"], info=data.lot, file=f)

    # write primitive centers
    prim_centers = [shell.icenter + 1 for shell in obasis.shells for _ in range(shell.nbasis)]
    print("<Primitive Centers>", file=f)
    for j in range(0, len(prim_centers), 10):
        print(" ".join([f"{c:d}" for c in prim_centers[j : j + 10]]), file=f)
    print("</Primitive Centers>", file=f)

    # write primitive types
    angmom_prim = {}
    count = 1
    for angmom in range(max([shell.angmoms[0] for shell in obasis.shells]) + 1):
        angmom_prim[angmom] = [count + i for i in range(len(obasis.conventions[angmom, "c"]))]
        count += len(obasis.conventions[angmom, "c"])
    prim_types = [item for shell in obasis.shells for item in angmom_prim[shell.angmoms[0]]]
    print("<Primitive Types>", file=f)
    for j in range(0, len(prim_types), 10):
        print(" ".join([f"{c:d}" for c in prim_types[j : j + 10]]), file=f)
    print("</Primitive Types>", file=f)

    # write primitive exponents
    exponents = [shell.exponents[0] for shell in obasis.shells for _ in range(shell.nbasis)]
    print("<Primitive Exponents>", file=f)
    for j in range(0, len(exponents), 4):
        print(" ".join([f"{e: ,.14E}" for e in exponents[j : j + 4]]), file=f)
    print("</Primitive Exponents>", file=f)

    # write molecular orbital occupation numbers
    _write_xml_iterator_scientific(tag=lbs["mo_occs"], info=data.mo.occs, file=f)

    # write molecular orbital energies
    _write_xml_iterator_scientific(tag=lbs["mo_energies"], info=data.mo.energies, file=f)

    # write molecular orbital spin types
    if data.mo.kind == "restricted":
        mo_spin = ["Alpha and Beta "] * len(data.mo.occs)
    else:
        mo_spin = ["Alpha"] * len(data.mo.occsa) + ["Beta"] * len(data.mo.occsb)
    _write_xml_iterator(tag=lbs["mo_spins"], info=mo_spin, file=f)

    # write MO primitive coefficients
    print("<Molecular Orbital Primitive Coefficients>", file=f)
    for mo in range(len(data.mo.occs)):
        print("<MO Number>", file=f)
        print(str(mo + 1), file=f)
        print("</MO Number>", file=f)
        for j in range(0, obasis.nbasis, 4):
            print(" ".join([f"{c: ,.14E}" for c in mo_coeffs.T[mo][j : j + 4]]), file=f)
    print("</Molecular Orbital Primitive Coefficients>", file=f)

    # write energy and virial ratio; use ' NAN' when None (not available)
    _write_xml_single_scientific(tag=lbs["energy"], info=data.energy or np.nan, file=f)
    _write_xml_single_scientific(lbs["virial_ratio"], data.extra.get("virial_ratio", np.nan), f)

    # write nuclear Cartesian energy gradients (optional)
    if data.atgradient is not None:
        nuc_cart_energy_grad = list(zip(nuclear_names, data.atgradient))
        print("<Nuclear Cartesian Energy Gradients>", file=f)
        for atom in nuc_cart_energy_grad:
            print(
                atom[0],
                f"{atom[1][0]: ,.14E} {atom[1][1]: ,.14E} {atom[1][2]: ,.14E}",
                file=f,
            )
        print("</Nuclear Cartesian Energy Gradients>", file=f)

    # nuclear virial of energy-gradient-based forces on nuclei (optional)
    if data.extra.get("nuc_viral") is not None:
        _write_xml_single_scientific(tag=lbs["nuc_viral"], info=data.extra["nuc_viral"], file=f)

    # write full virial ratio (optional)
    if data.extra.get("full_virial_ratio") is not None:
        _write_xml_single_scientific(lbs["full_virial_ratio"], data.extra["full_virial_ratio"], f)

    # number of core electrons (optional)
    if data.extra.get("num_core_electrons") is not None:
        _write_xml_single(lbs["num_core_electrons"], data.extra["num_core_electrons"], f)


def _write_xml_single(tag: str, info: [str, int], file: TextIO) -> None:
    """Write header, tail and the data between them into the file."""
    print(tag, file=file)
    print(info, file=file)
    print("</" + tag.lstrip("<"), file=file)


def _write_xml_single_scientific(tag: str, info: float, file: TextIO) -> None:
    """Write header, tail and the data between them into the file."""
    print(tag, file=file)
    print(f"{info: ,.14E}", file=file)
    print("</" + tag.lstrip("<"), file=file)


def _write_xml_iterator(tag: str, info: Iterator, file: TextIO) -> None:
    """Write list of arrays to file."""
    print(tag, file=file)
    for info_line in info:
        print(info_line, file=file)
    print("</" + tag.lstrip("<"), file=file)


def _write_xml_iterator_scientific(tag: str, info: Iterator, file: TextIO) -> None:
    """Write list of arrays to file."""
    print(tag, file=file)
    for info_line in info:
        print(f"{info_line: ,.14E}", file=file)
    print("</" + tag.lstrip("<"), file=file)
