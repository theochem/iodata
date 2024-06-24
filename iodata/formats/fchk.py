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
"""Gaussian FCHK file format."""

from collections.abc import Iterator
from fnmatch import fnmatch
from typing import Optional, TextIO

import numpy as np
from numpy.typing import NDArray

from ..basis import HORTON2_CONVENTIONS, MolecularBasis, Shell, convert_conventions
from ..docstrings import document_dump_one, document_load_many, document_load_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..utils import DumpError, LineIterator, LoadError, PrepareDumpError, amu

__all__ = []


PATTERNS = ["*.fchk", "*.fch"]


CONVENTIONS = {
    (9, "p"): HORTON2_CONVENTIONS[(9, "p")],
    (8, "p"): HORTON2_CONVENTIONS[(8, "p")],
    (7, "p"): HORTON2_CONVENTIONS[(7, "p")],
    (6, "p"): HORTON2_CONVENTIONS[(6, "p")],
    (5, "p"): HORTON2_CONVENTIONS[(5, "p")],
    (4, "p"): HORTON2_CONVENTIONS[(4, "p")],
    (3, "p"): HORTON2_CONVENTIONS[(3, "p")],
    (2, "p"): HORTON2_CONVENTIONS[(2, "p")],
    (0, "c"): ["1"],
    (1, "c"): ["x", "y", "z"],
    (2, "c"): ["xx", "yy", "zz", "xy", "xz", "yz"],
    (3, "c"): ["xxx", "yyy", "zzz", "xyy", "xxy", "xxz", "xzz", "yzz", "yyz", "xyz"],
    (4, "c"): HORTON2_CONVENTIONS[(4, "c")][::-1],
    (5, "c"): HORTON2_CONVENTIONS[(5, "c")][::-1],
    (6, "c"): HORTON2_CONVENTIONS[(6, "c")][::-1],
    (7, "c"): HORTON2_CONVENTIONS[(7, "c")][::-1],
    (8, "c"): HORTON2_CONVENTIONS[(8, "c")][::-1],
    (9, "c"): HORTON2_CONVENTIONS[(9, "c")][::-1],
}


@document_load_one(
    "Gaussian Formatted Checkpoint",
    [
        "atcharges",
        "atcoords",
        "atnums",
        "atcorenums",
        "lot",
        "mo",
        "obasis",
        "obasis_name",
        "run_type",
        "title",
    ],
    ["energy", "atfrozen", "atgradient", "athessian", "atmasses", "one_rdms", "extra", "moments"],
)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    fchk = _load_fchk_low(
        lit,
        [
            "Number of electrons",
            "Number of basis functions",
            "Number of alpha electrons",
            "Number of beta electrons",
            "Atomic numbers",
            "Current cartesian coordinates",
            "Real atomic weights",
            "Shell types",
            "Shell to atom map",
            "Shell to atom map",
            "Number of primitives per shell",
            "Primitive exponents",
            "Contraction coefficients",
            "P(S=P) Contraction coefficients",
            "Alpha Orbital Energies",
            "Alpha MO coefficients",
            "Beta Orbital Energies",
            "Beta MO coefficients",
            "Total Energy",
            "Nuclear charges",
            "Total SCF Density",
            "Spin SCF Density",
            "Total MP2 Density",
            "Spin MP2 Density",
            "Total MP3 Density",
            "Spin MP3 Density",
            "Total CC Density",
            "Spin CC Density",
            "Total CI Density",
            "Spin CI Density",
            "Mulliken Charges",
            "ESP Charges",
            "NPA Charges",
            "MBS Charges",
            "Type 6 Charges",
            "Type 7 Charges",
            "Polarizability",
            "Dipole Moment",
            "Quadrupole Moment",
            "Cartesian Gradient",
            "Cartesian Force Constants",
            "MicOpt",
        ],
    )

    # A) Load a bunch of simple things
    result = {
        "title": fchk["title"],
        # if "Total Energy" is not present in FCHk, None is returned.
        "energy": fchk.get("Total Energy", None),
        "lot": fchk["lot"].lower(),
        "obasis_name": fchk["obasis_name"].lower(),
        "atcoords": fchk["Current cartesian coordinates"].reshape(-1, 3),
        "atnums": fchk["Atomic numbers"],
        "atcorenums": fchk["Nuclear charges"],
    }

    atmasses = fchk.get("Real atomic weights")
    if atmasses is not None:
        result["atmasses"] = atmasses * amu
    atgradient = fchk.get("Cartesian Gradient")
    if atgradient is not None:
        result["atgradient"] = atgradient.reshape(-1, 3)
    athessian = fchk.get("Cartesian Force Constants")
    if athessian is not None:
        result["athessian"] = _triangle_to_dense(athessian)
    atfrozen = fchk.get("MicOpt")
    if atfrozen is not None:
        result["atfrozen"] = atfrozen == -2
    run_types = {"SP": "energy", "FOpt": "opt", "Scan": "scan", "Freq": "freq"}
    run_type = run_types.get(fchk["command"])
    if run_type is not None:
        result["run_type"] = run_type

    # B) Load the orbital basis set
    shell_types = fchk["Shell types"]
    shell_map = fchk["Shell to atom map"] - 1
    nprims = fchk["Number of primitives per shell"]
    exponents = fchk["Primitive exponents"]
    ccoeffs_level1 = fchk["Contraction coefficients"]
    ccoeffs_level2 = fchk.get("P(S=P) Contraction coefficients")

    shells = []
    counter = 0
    # First loop over all shells
    for i, n in enumerate(nprims):
        if shell_types[i] == -1:
            # Special treatment for SP shell type
            shells.append(
                Shell(
                    shell_map[i],
                    [0, 1],
                    ["c", "c"],
                    exponents[counter : counter + n],
                    np.stack(
                        [
                            ccoeffs_level1[counter : counter + n],
                            ccoeffs_level2[counter : counter + n],
                        ],
                        axis=1,
                    ),
                )
            )
        else:
            shells.append(
                Shell(
                    shell_map[i],
                    [abs(shell_types[i])],
                    ["p" if shell_types[i] < 0 else "c"],
                    exponents[counter : counter + n],
                    ccoeffs_level1[counter : counter + n][:, np.newaxis],
                )
            )
        counter += n
    del shell_map
    del shell_types
    del nprims
    del exponents

    result["obasis"] = MolecularBasis(shells, CONVENTIONS, "L2")
    nbasis = fchk["Number of basis functions"]

    # C) Load density matrices
    one_rdms = {}
    _load_dm("Total SCF Density", fchk, one_rdms, "scf")
    _load_dm("Spin SCF Density", fchk, one_rdms, "scf_spin")
    # only one of the lots should be present, hence using the same key
    for lot in "MP2", "MP3", "CC", "CI":
        _load_dm(f"Total {lot} Density", fchk, one_rdms, "post_scf_ao")
        _load_dm(f"Spin {lot} Density", fchk, one_rdms, "post_scf_spin_ao")
    if one_rdms:
        result["one_rdms"] = one_rdms

    # D) Load the wavefunction

    # Load orbitals
    nalpha = fchk["Number of alpha electrons"]
    nbeta = fchk["Number of beta electrons"]
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        raise LoadError("The number of electrons is not positive.", lit)
    if nalpha < nbeta:
        raise LoadError(f"n_alpha={nalpha} < n_beta={nbeta} is invalid.", lit)

    norba = fchk["Alpha Orbital Energies"].shape[0]
    mo_coeffs = np.copy(fchk["Alpha MO coefficients"].reshape(norba, nbasis).T)
    mo_energies = np.copy(fchk["Alpha Orbital Energies"])

    if "Beta Orbital Energies" in fchk:
        # unrestricted
        norbb = fchk["Beta Orbital Energies"].shape[0]
        mo_coeffs_b = np.copy(fchk["Beta MO coefficients"].reshape(norbb, nbasis).T)
        mo_coeffs = np.concatenate((mo_coeffs, mo_coeffs_b), axis=1)
        mo_energies = np.concatenate((mo_energies, np.copy(fchk["Beta Orbital Energies"])), axis=0)
        mo_occs = np.zeros(norba + norbb)
        mo_occs[:nalpha] = 1.0
        mo_occs[norba : norba + nbeta] = 1.0
        mo = MolecularOrbitals("unrestricted", norba, norbb, mo_occs, mo_coeffs, mo_energies)
    else:
        # restricted closed-shell and open-shell
        mo_occs = np.zeros(norba)
        mo_occs[:nalpha] = 1.0
        mo_occs[:nbeta] = 2.0
        # delete dm_full_scf because it is known to be buggy
        if nalpha != nbeta and "one_rdms" in result and "scf" in result["one_rdms"]:
            result["one_rdms"].pop("scf")
        mo = MolecularOrbitals("restricted", norba, norba, mo_occs, mo_coeffs, mo_energies)
    result["mo"] = mo

    # E) Load properties
    if "Polarizability" in fchk:
        result["extra"] = {"polarizability_tensor": _triangle_to_dense(fchk["Polarizability"])}
    moments = {}
    if "Dipole Moment" in fchk:
        moments[(1, "c")] = fchk["Dipole Moment"]
    if "Quadrupole Moment" in fchk:
        # Convert to alphabetical ordering: xx, xy, xz, yy, yz, zz
        moments[(2, "c")] = fchk["Quadrupole Moment"][[0, 3, 4, 1, 5, 2]]
    if moments:
        result["moments"] = moments
    atcharges = {}
    if "Mulliken Charges" in fchk:
        atcharges["mulliken"] = fchk["Mulliken Charges"]
    if "ESP Charges" in fchk:
        atcharges["esp"] = fchk["ESP Charges"]
    if "NPA Charges" in fchk:
        atcharges["npa"] = fchk["NPA Charges"]
    if "MBS Charges" in fchk:
        atcharges["mbs"] = fchk["MBS Charges"]
    if "Type 6 Charges" in fchk:
        atcharges["hirshfeld"] = fchk["Type 6 Charges"]
    if "Type 7 Charges" in fchk:
        atcharges["cm5"] = fchk["Type 7 Charges"]
    if atcharges:
        result["atcharges"] = atcharges

    return result


LOAD_MANY_NOTES = """
Trajectories from a Gaussian optimization, relaxed scan or IRC calculation are written in
groups of frames, called "points" in the Gaussian world, e.g. to discrimininate between
different values of the constraint in a relaxed geometry. In most cases, e.g. IRC or
conventional optimization, there is only one "point". Within one "point", one can have
multiple geometries and their properties. This information is stored in the ``extra``
attribute:

- ``ipoint`` is the counter for a point
- ``npoint`` is the total number of points.
- ``istep`` is the counter within one "point"
- ``nstep`` is the total number of geometries within in a "point".
- ``reaction_coordinate`` is only present in case of an IRC calculation.
"""


@document_load_many(
    "XYZ",
    ["atcoords", "atgradient", "atnums", "atcorenums", "energy", "extra", "title"],
    [],
    {},
    LOAD_MANY_NOTES,
)
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    fchk = _load_fchk_low(
        lit,
        [
            "Atomic numbers",
            "Current cartesian coordinates",
            "Nuclear charges",
            "IRC *",
            "Optimization *",
            "Opt point *",
        ],
    )

    # Determine the type of calculation: IRC or Optimization
    if "IRC Number of geometries" in fchk:
        prefix = "IRC point"
        nsteps = fchk["IRC Number of geometries"]
    elif "Optimization Number of geometries" in fchk:
        prefix = "Opt point"
        nsteps = fchk["Optimization Number of geometries"]
    else:
        raise LoadError("Cannot find IRC or Optimization trajectory in FCHK file.", lit)

    natom = fchk["Atomic numbers"].size
    for ipoint, nstep in enumerate(nsteps):
        results_geoms = fchk[f"{prefix} {ipoint + 1:7d} Results for each geome"]
        trajectory = list(
            zip(
                results_geoms[::2],
                results_geoms[1::2],
                fchk[f"{prefix} {ipoint + 1:7d} Geometries"].reshape(-1, natom, 3),
                fchk[f"{prefix} {ipoint + 1:7d} Gradient at each geome"].reshape(-1, natom, 3),
            )
        )
        assert len(trajectory) == nstep
        for istep, (energy, recor, atcoords, gradients) in enumerate(trajectory):
            data = {
                "title": fchk["title"],
                "atnums": fchk["Atomic numbers"],
                "atcorenums": fchk["Nuclear charges"],
                "energy": energy,
                "atcoords": atcoords,
                "atgradient": gradients,
                "extra": {
                    "ipoint": ipoint,
                    "npoint": len(nsteps),
                    "istep": istep,
                    "nstep": nstep,
                },
            }
            if prefix == "IRC point":
                data["extra"]["reaction_coordinate"] = recor
            yield data


def _load_fchk_low(lit: LineIterator, label_patterns: Optional[list[str]] = None) -> dict:
    """Read selected fields from a formatted checkpoint file.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    label_patterns
        A list of Unix shell-style wildcard patterns of labels to read.

    Returns
    -------
    A dictionary containing data read from the FCHK file.
    Keys are the field names and values are either scalar or array data.
    Arrays are always one-dimensional.

    """
    # Read the two-line header
    result = {"title": next(lit).strip()}
    words = next(lit).split()
    if len(words) == 3:
        result["command"], result["lot"], result["obasis_name"] = words
    elif len(words) == 2:
        result["command"], result["lot"] = words
    else:
        raise LoadError("The second line of the FCHK file should contain two or three words.", lit)

    while True:
        try:
            label, value = _load_fchk_field(lit, label_patterns)
        except StopIteration:
            # We always read until the end of the file.
            break
        result[label] = value
    return result


def _load_fchk_field(lit: LineIterator, label_patterns: list[str]) -> tuple[str, object]:
    """Read a single field matching one of the given label_patterns.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    label_patterns
        A list of Unix shell-style wildcard patterns. The next field matching
        one of the patterns is returned

    Returns
    -------
    label
        The name of the field
    value
        The scalar or array data of the field.

    """
    while True:
        # find a sane header line
        line = next(lit)
        label = line[:43].strip()
        words = line[43:].split()
        if not words:
            continue
        if words[0] == "I":
            datatype = int
        elif words[0] == "R":
            datatype = float
        else:
            continue
        if not (
            label_patterns is None
            or any(fnmatch(label, label_pattern) for label_pattern in label_patterns)
        ):
            continue
        if len(words) == 2:
            try:
                return label, datatype(words[1])
            except ValueError as exc:
                raise LoadError(f"Could not interpret as {datatype}: {words[1]}", lit) from exc
        elif len(words) == 3:
            if words[1] != "N=":
                raise LoadError("Expected N= not found.", lit)
            length = int(words[2])
            value = np.zeros(length, datatype)
            counter = 0
            words = []
            while counter < length:
                if not words:
                    words = next(lit).split()
                word = words.pop(0)
                try:
                    value[counter] = datatype(word)
                except (ValueError, OverflowError) as exc:
                    raise LoadError(f"Could not interpret as {datatype}: {word}", lit) from exc
                counter += 1
            return label, value


def _load_dm(label: str, fchk: dict, result: dict, key: str):
    """Load a density matrix from the FCHK file if present.

    Parameters
    ----------
    label
        The label in the FCHK file.
    fchk
        The dictionary with labels from the FCHK file.
    result
        The output dictionary.
    key:
        The key to be used in the output dictionary.

    """
    if label in fchk:
        result[key] = _triangle_to_dense(fchk[label])


def _triangle_to_dense(triangle: NDArray[float]) -> NDArray[float]:
    """Convert a symmetric matrix in triangular storage to a dense square matrix.

    Parameters
    ----------
    triangle
        A row vector containing all the unique matrix elements of symmetric
        matrix. (Either the lower-triangular part in row major-order or the
        upper-triangular part in column-major order.)

    Returns
    -------
    A square symmetric matrix.

    """
    nrow = int(np.round((np.sqrt(1 + 8 * len(triangle)) - 1) / 2))
    result = np.zeros((nrow, nrow))
    begin = 0
    for irow in range(nrow):
        end = begin + irow + 1
        result[irow, : irow + 1] = triangle[begin:end]
        result[: irow + 1, irow] = triangle[begin:end]
        begin = end
    return result


# The fchk file has a very rigid format, to dump the information are
# theses functions, both scalars and arrays, integer and real(float) variables
def _dump_integer_scalars(name: str, val: int, f: TextIO):
    """Dumper for a scalar integer."""
    print(f"{name:40}   I     {int(val):12d}", file=f)


def _dump_real_scalars(name: str, val: float, f: TextIO):
    """Dumper for a scalar float."""
    print(f"{name:40}   R     {float(val): 16.8E}", file=f)


def _dump_integer_arrays(name: str, val: NDArray[int], f: TextIO):
    """Dumper for a array of integers."""
    nval = val.size
    if nval != 0:
        np.reshape(val, nval)
        print(f"{name:40}   I   N={nval:12}", file=f)
        k = 0
        for i in range(nval):
            print(f"{int(val[i]):12}", file=f, end="")
            k += 1
            if k == 6 or i == nval - 1:
                print("", file=f)
                k = 0


def _dump_real_arrays(name: str, val: NDArray[float], f: TextIO):
    """Dumper for a array of float."""
    nval = val.size
    if nval != 0:
        np.reshape(val, nval)
        print(f"{name:40}   R   N={nval:12}", file=f)
        k = 0
        for i in range(nval):
            print(f"{val[i]: 16.8E}", file=f, end="")
            k += 1
            if k == 5 or i == nval - 1:
                print("", file=f)
                k = 0


def prepare_dump(filename: str, data: IOData):
    """Check the compatibility of the IOData object with the FCHK format.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    data
        The IOData instance to be checked.
    """
    if data.mo is not None:
        if data.mo.kind == "generalized":
            raise PrepareDumpError("Cannot write FCHK file with generalized orbitals.", filename)
        na = int(np.round(np.sum(data.mo.occsa)))
        if not ((data.mo.occsa[:na] == 1.0).all() and (data.mo.occsa[na:] == 0.0).all()):
            raise PrepareDumpError(
                "Cannot dump FCHK because it does not have fully occupied alpha orbitals "
                "followed by fully virtual ones.",
                filename,
            )
        nb = int(np.round(np.sum(data.mo.occsb)))
        if not ((data.mo.occsb[:nb] == 1.0).all() and (data.mo.occsb[nb:] == 0.0).all()):
            raise PrepareDumpError(
                "Cannot dump FCHK because it does not have fully occupied beta orbitals "
                "followed by fully virtual ones.",
                filename,
            )


@document_dump_one(
    "Gaussian Formatted Checkpoint",
    ["atnums", "atcorenums"],
    [
        "atcharges",
        "atcoords",
        "atfrozen",
        "atgradient",
        "athessian",
        "atmasses",
        "charge",
        "energy",
        "lot",
        "mo",
        "one_rdms",
        "obasis_name",
        "extra",
        "moments",
    ],
)
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # write title
    print("{:72}".format(data.title or "FCHK generated by IOData"), file=f)

    # write run type, level of theory, and basis set name (all in uppercase)
    items = [getattr(data, item) or "NA" for item in ["run_type", "lot", "obasis_name"]]
    if items[0] == "energy":
        items[0] = "SP"
    print(f"{items[0].upper():10s}{items[1].upper():30s}{items[2].upper():>33s}", file=f)

    # write basic information
    _dump_integer_scalars("Number of atoms", data.natom, f)
    _dump_integer_scalars("Number of electrons", int(data.nelec), f)
    if data.charge is not None:
        _dump_integer_scalars("Charge", int(data.charge), f)
    if data.mo is not None:
        na = int(np.round(np.sum(data.mo.occsa)))
        nb = int(np.round(np.sum(data.mo.occsb)))
        # assign number of alpha and beta electrons
        multiplicity = abs(na - nb) + 1
        _dump_integer_scalars("Multiplicity", multiplicity, f)
        _dump_integer_scalars("Number of alpha electrons", na, f)
        _dump_integer_scalars("Number of beta electrons", nb, f)

    # write atomic numbers, nuclear charges, and atomic coordinates
    _dump_integer_arrays("Atomic numbers", data.atnums, f)
    _dump_real_arrays("Nuclear charges", data.atcorenums, f)
    if data.atcoords is not None:
        _dump_real_arrays("Current cartesian coordinates", data.atcoords.flatten(), f)

    # write atomic weights
    if data.atmasses is not None:
        masses = data.atmasses / amu
        _dump_integer_arrays("Integer atomic weights", masses.round(), f)
        _dump_real_arrays("Real atomic weights", masses, f)

    # write molecular orbital basis set
    if data.obasis is not None:
        # number of primitives per shell
        nprims = np.array([shell.nprim for shell in data.obasis.shells])
        exponents = np.array([item for shell in data.obasis.shells for item in shell.exponents])
        coeffs = np.array([s.coeffs[i][0] for s in data.obasis.shells for i in range(s.nprim)])
        coordinates = np.array([data.atcoords[shell.icenter] for shell in data.obasis.shells])
        shell_to_atom = np.array([shell.icenter + 1 for shell in data.obasis.shells])

        # get list of shell types: 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f...
        shell_types = []
        for shell in data.obasis.shells:
            if shell.ncon == 1 and shell.kinds == ["c"]:
                shell_types.append(shell.angmoms[0])
            elif shell.ncon == 1 and shell.kinds == ["p"]:
                shell_types.append(-1 * shell.angmoms[0])
            elif shell.ncon == 2 and shell.angmoms == [0, 1]:
                shell_types.append(-1)
            else:
                raise DumpError("Cannot identify type of shell!", f)

        num_pure_d_shells = sum([1 for st in shell_types if st == 2])
        num_pure_f_shells = sum([1 for st in shell_types if st == 3])

        _dump_integer_scalars("Number of basis functions", data.obasis.nbasis, f)
        _dump_integer_scalars("Number of independent functions", data.obasis.nbasis, f)
        _dump_integer_scalars("Number of contracted shells", len(data.obasis.shells), f)
        _dump_integer_scalars("Number of primitive shells", nprims.sum(), f)
        _dump_integer_scalars("Pure/Cartesian d shells", num_pure_d_shells, f)
        _dump_integer_scalars("Pure/Cartesian f shells", num_pure_f_shells, f)
        _dump_integer_scalars("Highest angular momentum", np.amax(np.abs(shell_types)), f)
        _dump_integer_scalars("Largest degree of contraction", np.amax(nprims), f)

        _dump_integer_arrays("Shell types", np.array(shell_types), f)
        _dump_integer_arrays("Number of primitives per shell", nprims, f)
        _dump_integer_arrays("Shell to atom map", shell_to_atom, f)

        _dump_real_arrays("Primitive exponents", exponents, f)
        _dump_real_arrays("Contraction coefficients", coeffs, f)

        if -1 in shell_types:
            sp_coeffs = []
            for shell, shell_type in zip(data.obasis.shells, shell_types):
                if shell_type == -1:
                    sp_coeffs.extend([shell.coeffs[i][1] for i in range(shell.nprim)])
                else:
                    sp_coeffs.extend([0.0] * shell.nprim)
            _dump_real_arrays("P(S=P) Contraction coefficients", np.array(sp_coeffs), f)
        _dump_real_arrays("Coordinates of each shell", coordinates.flatten(), f)

    # write energy
    if data.energy is not None:
        _dump_real_scalars("SCF Energy", data.energy, f)
        _dump_real_scalars("Total Energy", data.energy, f)

    # write MO energies & coefficients
    if data.mo is not None:
        # convert to FCHK basis conventions
        permutation, signs = convert_conventions(data.obasis, CONVENTIONS)
        coeffsa = data.mo.coeffsa[permutation] * signs.reshape(-1, 1)
        _dump_real_arrays("Alpha Orbital Energies", data.mo.energiesa, f)
        _dump_real_arrays("Alpha MO coefficients", coeffsa.transpose().flatten(), f)
        if data.mo.kind == "unrestricted":
            coeffsb = data.mo.coeffsb[permutation] * signs.reshape(-1, 1)
            _dump_real_arrays("Beta Orbital Energies", data.mo.energiesb, f)
            _dump_real_arrays("Beta MO coefficients", coeffsb.transpose().flatten(), f)

    # write reduced density matrix, if available
    # get level of theory, use 'NA' if not available
    level = data.lot.upper() if data.lot is not None else "NA"
    for item in ["MP2", "MP3", "CC", "CI"]:
        if item in level:
            level = item
    for key, arr in data.one_rdms.items():
        # get lower triangular elements of RDM
        mat = arr[np.tril_indices(arr.shape[0])]

        # identify type of RDMs
        if key == "scf":
            title = "Total SCF Density"
        elif key == "scf_spin":
            title = "Spin SCF Density"
        elif key == "post_scf_ao":
            title = f"Total {level} Density"
        elif key == "post_scf_spin_ao":
            title = f"Spin {level} Density"
        else:
            title = "Total SCF Density"
        _dump_real_arrays(title, mat, f)

    # write atomic charges
    if "mulliken" in data.atcharges:
        _dump_real_arrays("Mulliken Charges", data.atcharges["mulliken"], f)
    if "esp" in data.atcharges:
        _dump_real_arrays("ESP Charges", data.atcharges["esp"], f)
    if "npa" in data.atcharges:
        _dump_real_arrays("NPA Charges", data.atcharges["npa"], f)
    if "mbs" in data.atcharges:
        _dump_real_arrays("MBS Charges", data.atcharges["mbs"], f)
    if "hirshfeld" in data.atcharges:
        _dump_real_arrays("Type 6 Charges", data.atcharges["hirshfeld"], f)
    if "cm5" in data.atcharges:
        _dump_real_arrays("Type 7 Charges", data.atcharges["cm5"], f)

    # write atomic gradient
    if data.atgradient is not None:
        _dump_real_arrays("Cartesian Gradient", data.atgradient.flatten(), f)

    # write atomic hessian
    if data.athessian is not None:
        arr = data.athessian[np.tril_indices(data.athessian.shape[0])]
        _dump_real_arrays("Cartesian Force Constants", arr, f)

    # write moments
    if (1, "c") in data.moments:
        _dump_real_arrays("Dipole Moment", data.moments[(1, "c")], f)
    if (2, "c") in data.moments and len(data.moments[(2, "c")]) != 0:
        # quadrupole moments are stored as XX, XY, XZ, YY, YZ, ZZ in IOData, so they need to
        # be permuted to have XX, YY, ZZ, XY, XZ, YZ order for FCHK.
        quadrupole = data.moments[(2, "c")][[0, 3, 5, 1, 2, 4]]
        _dump_real_arrays("Quadrupole Moment", quadrupole, f)

    # write polarizability tensor
    if "polarizability_tensor" in data.extra:
        arr = data.extra["polarizability_tensor"]
        arr = arr[np.tril_indices(arr.shape[0])]
        _dump_real_arrays("Polarizability", arr, f)
