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
"""QCSchema JSON file format.

QCSchema defines four different subschema:

- :ref:`Molecule <json_schema_molecule>`: specifying a molecular system
- :ref:`Input <json_schema_input>`: specifying QC program input for a specific Molecule
- :ref:`Output <json_schema_output>`: specifying QC program output for a specific Molecule
- Basis: specifying a basis set for a specific Molecule

General Usage
-------------
The QCSchema format is intended to be a catch-all file format for storing and sharing QC calculation
data. Due to the wide number of possibilities of the data contained in a single file, not every
field in a QCSchema file directly corresponds to an IOData attribute. For example,
``qcschema_output`` files allow for many fields capturing different energy contributions, especially
for coupled-cluster calculations. To accommodate this fact, IOData does not always assume the intent
of the user; instead, IOData ensures that every field in the file is stored in a structured manner.
When a QCSchema field does not correspond to an IOData attribute, that data is instead stored in the
``extra`` dict, in a dictionary corresponding to the subschema where that data was found. In cases
where multiple subschema contain the relevant field (e.g. the Output subschema contains the entirety
of the Input subschema), the data will be found in the smallest subschema (for the example above, in
``IOData.extra["input"]``, not ``IOData.extra["output"]``).

Dumping an IOData instance to a QCSchema file involves adding relevant required (and optional, if
needed) fields to the necessary dictionaries in the ``extra`` dict. One exception is the
``provenance`` field: if the only desired provenance data is the creation of the file by IOData,
that data will be added automatically.

The following sections will describe the requirements of each subschema and the behaviour to expect
from IOData when loading in or dumping out a QCSchema file.

Schema Definitions
------------------

.. _json_schema_provenance:

Provenance Information
^^^^^^^^^^^^^^^^^^^^^^
The provenance field contains information about how the associated QCSchema object and its
attributes were generated, provided, and manipulated. A provenance entry expects these fields:

========= ===========
Field     Description
========= ===========
creator   **Required**. The program that generated, provided, or manipulated this file.
version   The version of the creator.
routine   The routine of the creator.
========= ===========

In QCElemental, only a single provenance entry is permitted. When generating a QCSchema file for use
with QCElemental, the easiest way to ensure compliance is to leave the provenance field blank, to
allow the ``dump_one`` function to generate the correct provenance information. However, allowing
only one entry for provenance information limits the ability to properly trace a file through
several operations during complex workflows. With this in mind, IOData supports an enhanced
provenance field, in the form of a list of provenance entries, with new entries appended to the end
of the list.

.. _json_schema_molecule:

Molecule Schema
^^^^^^^^^^^^^^^
The ``qcschema_molecule`` subschema describes a molecular system, and contains the data necessary to
specify a molecular system and support I/O and manipulation processes.

The following is an example of a minimal ``qcschema_molecule`` file:

.. code-block :: JSON

    {
      "schema_name": "qcschema_molecule",
      "schema_version": 2,
      "symbols":  ["Li", "Cl"],
      "geometry": [0.000000, 0.000000, -1.631761, 0.000000, 0.000000, 0.287958],
      "molecular_charge": 0,
      "molecular_multiplicity": 1,
      "provenance": {
        "creator": "HORTON3",
        "routine": "Manual validation"
      }
    }


The required fields and corresponding types for a ``qcschema_molecule`` file are:

====================== ============ ============ =================================================
Field                  Type         IOData attr. Description
====================== ============ ============ =================================================
schema_name            str          N/A          The name of the QCSchema subschema. Fixed as
                                                 ``qcschema_molecule``.
schema_version         str          N/A          The version of the subschema specification.
                                                 2.0 is the current version.
symbols                list(N_at)   ``atnums``   An array of the atomic symbols for the system.
geometry               list(3*N_at) ``atcoords`` An ordered array of XYZ atomic coordinates,
                                                 corresponding to the order of ``symbols``. The
                                                 first three elements correspond to atom one,
                                                 the second three to atom two, etc.
molecular_charge       float        ``charge``   The net electrostatic charge of the molecule.
                                                 Some writers assume a default of 0.
molecular_multiplicity int          ``spinpol``  The total multiplicity of this molecule.
                                                 Some writers assume a default of 1.
provenance             dict or list N/A          Information about the file was generated,
                                                 provided, and manipulated. See
                                                 :ref:`Provenance section <json_schema_provenance>`
                                                 above for more details.
====================== ============ ============ =================================================

Note: N_at corresponds to the number of atoms in the molecule, as defined by the length of
``symbols``.

The optional fields and corresponding types for a ``qcschema_molecule`` file are:

======================= ============ ============== ================================================
Field                   Type         IOData attr.   Description
======================= ============ ============== ================================================
atom_labels             list(N_at)   N/A            Additional per-atom labels. Typically used for
                                                    model conversions, not user assignment. The
                                                    indices of this array correspond to the
                                                    ``symbols`` ordering.
atomic_numbers          list(N_at)   ``atnums``     An array of atomic numbers for each atom.
                                                    Typically inferred from ``symbols``.
comment                 str          N/A            Additional comments for this molecule. These
                                                    comments are intended for user information, not
                                                    any computational tasks.
connectivity            list         ``bonds``      The connectivity information between each atom
                                                    in the ``symbols`` array. Each entry in this
                                                    array is a 3-item array,
                                                    ``[index_a, index_b, bond_order]``,
                                                    where the indices correspond to the atom indices
                                                    in ``symbols``.
extras                  dict         N/A            Extra information to associate with this
                                                    molecule.
fix_symmetry            str          ``g_rot``      Maximal point group symmetry with which the
                                                    molecule should be treated.
fragments               list(N_fr)   N/A            An array that designates which sets of atoms are
                                                    fragments within the molecule. This is a nested
                                                    array, with the indices of the base array
                                                    corresponding to the values in
                                                    ``fragment_charges`` and
                                                    ``fragment_multiplicities`` and the values in
                                                    the nested arrays corresponding to the indices
                                                    of ``symbols``.
fragment_charges        list(N_fr)   N/A            The total charge of each fragment in
                                                    ``fragments``. The indices of this array
                                                    correspond to the ``fragments`` ordering.
fragment_multiplicities list(N_fr)   N/A            The multiplicity of each fragment in
                                                    ``fragments``. The indices of this array
                                                    correspond to the ``fragments`` ordering.
id                      str          N/A            A unique identifier for this molecule.
identifiers             dict         N/A            Additional identifiers by which this molecule
                                                    can be referenced, such as INCHI, SMILES, etc.
real                    list(N_at)   ``atcorenums`` An array indicating whether each atom is real
                                                    (true) or a ghost/virtual atom (false). The
                                                    indices of this array correspond to the
                                                    ``symbols`` ordering.
mass_numbers            list(N_at)   ``atmasses``   An array of atomic mass numbers for each atom.
                                                    The indices of this array correspond to the
                                                    ``symbols`` ordering.
masses                  list(N_at)   ``atmasses``   An array of atomic masses [u] for each atom.
                                                    Typically inferred from ``symbols``. The indices
                                                    of this array correspond to the ``symbols``
                                                    ordering.
name                    str          ``title``      An arbitrary, common, or human-readable name to
                                                    assign to this molecule.
======================= ============ ============== ================================================

Note: N_at corresponds to the number of atoms in the molecule, as defined by the length of
``symbols``; N_fr corresponds to the number of fragments in the molecule, as defined by the length
of ``fragments``. Fragment data is stored in a sub-dictionary, ``fragments``.

The following are additional optional keywords used in QCElemental's QCSchema implementation. These
keywords mostly correspond to specific QCElemental functionality, and may not necessarily produce
similar results in other QCSchema parsers.

======================= ============ ==================================================
Field                   Type         Description
======================= ============ ==================================================
fix_com                 bool         An indicator to prevent pre-processing the
                                     molecule by translating the COM to (0,0,0) in
                                     Euclidean coordinate space.
fix_orientation         bool         An indicator to prevent pre-processing the
                                     molecule by orienting via the inertia tensor.
validated               bool         An indicator that the input molecule data has been
                                     previously checked for schema and physics (e.g.
                                     non-overlapping atoms, feasible multiplicity)
                                     compliance. Generally should only be true when set
                                     by a trusted validator.
======================= ============ ==================================================

.. _json_schema_input:

Input Schema
^^^^^^^^^^^^
The ``qcschema_input`` subschema describes all data necessary to generate and parse a QC program
input file for a given molecule.

The following is an example of a minimal ``qcschema_input`` file:

.. code-block :: JSON

    {
      "schema_name": "qcschema_input",
      "schema_version": 2.0,
      "molecule": {
        "schema_name": "qcschema_molecule",
        "schema_version": 2.0,
        "symbols":  ["Li", "Cl"],
        "geometry": [0.000000, 0.000000, -1.631761, 0.000000, 0.000000, 0.287958],
        "molecular_charge": 0.0,
        "molecular_multiplicity": 1,
        "provenance": {
          "creator": "HORTON3",
          "routine": "Manual validation"
        }
      },
      "driver": "energy",
      "model": {
        "method": "B3LYP",
        "basis": "Def2TZVP"
      }
    }

The required fields and corresponding types for a ``qcschema_input`` file are:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
schema_name             str          N/A          The QCSchema specification to which this model
                                                  conforms. Fixed as ``qcschema_input``.
schema_version          float        N/A          The version number of ``schema_name`` to which
                                                  this model conforms, currently 2.
molecule                dict         N/A          :ref:`QCSchema Molecule <json_schema_molecule>`
                                                  instance.
driver                  str          N/A          The type of calculation being performed. One of
                                                  ``energy``, ``gradient``, ``hessian``, or
                                                  ``properties``.
model                   dict         N/A          The quantum chemistry model specification for a
                                                  given operation to compute against. See
                                                  :ref:`Model section <json_schema_model>` below.
======================= ============ ============ ==================================================

The optional fields and corresponding types for a `qcschema_input` file are:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
extras                  dict         N/A          Extra information associated with the input.
id                      str          N/A          An identifier for the input object.
keywords                dict         N/A          QC program-specific keywords to be used for a
                                                  computation. See details below for IOData-specific
                                                  usages.
protocols               dict         N/A          Protocols regarding the manipulation of the output
                                                  that results from this input. See
                                                  :ref:`Protocols section <json_schema_protocols>`
                                                  below.
provenance              dict or list N/A          Information about the file was generated,
                                                  provided, and manipulated. See
                                                  :ref:`Provenance section <json_schema_provenance>`
                                                  above for more information.
======================= ============ ============ ==================================================

IOData currently supports the following keywords for ``qcschema_input`` files:

======================= ============ ============ ==================================================
Keyword                 Type         IOData attr. Description
======================= ============ ============ ==================================================
run_type                str          ``run_type`` The type of calculation that lead to the results
                                                  stored in IOData, which must be one of the
                                                  following: ``energy``, ``energy_force``, ``opt``,
                                                  ``scan``, ``freq`` or None.
======================= ============ ============ ==================================================

.. _json_schema_model:

Model Subschema
^^^^^^^^^^^^^^^
The ``model`` dict contains the following fields:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
method                  str          ``lot``      The level of theory used for the computation (e.g.
                                                  B3LYP, PBE, CCSD(T), etc.)
basis                   str or dict  N/A          The quantum chemistry basis set to evaluate (e.g.
                                                  6-31G, cc-pVDZ, etc.) Can be 'none' for methods
                                                  without basis sets. Must be either a string
                                                  specifying the basis set name (the same as its
                                                  name in the Basis Set Exchange, when possible) or
                                                  a qcschema_basis instance.
======================= ============ ============ ==================================================

.. _json_schema_protocols:

Protocols Subschema
^^^^^^^^^^^^^^^^^^^
The ``protocols`` dict contains the following fields:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
wavefunction            str          N/A          Specification of the wavefunction properties to
                                                  keep from the resulting output. One of ``all``,
                                                  ``orbitals_and_eigenvalues``, ``return_results``,
                                                  or ``none``.
keep_stdout             bool         N/A          An indicator to keep the output file from the
                                                  resulting output.
======================= ============ ============ ==================================================

.. _json_schema_output:

Output Schema
^^^^^^^^^^^^^
The ``qcschema_output`` subschema describes all data necessary to generate and parse a QC program's
output file for a given molecule.

The following is an example of a minimal ``qcschema_output`` file:

.. code-block :: JSON

    {
      "schema_name": "qcschema_output",
      "schema_version": 2.0,
      "molecule": {
        "schema_name": "qcschema_molecule",
        "schema_version": 2.0,
        "symbols":  ["Li", "Cl"],
        "geometry": [0.000000, 0.000000, -1.631761, 0.000000, 0.000000, 0.287958],
        "molecular_charge": 0.0,
        "molecular_multiplicity": 1,
        "provenance": {
          "creator": "HORTON3",
          "routine": "Manual validation"
        }
      },
      "driver": "energy",
      "model": {
        "method": "HF",
        "basis": "STO-4G"
      },
      "properties": {},
      "return_result": -464.626219879,
      "success": true
    }

The required fields and corresponding types for a ``qcschema_output`` file are:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
schema_name             str          N/A          The QCSchema specification to which this model
                                                  conforms. Fixed as ``qcschema_output``.
schema_version          float        N/A          The version number of ``schema_name`` to which
                                                  this model conforms, currently 2.
molecule                dict         N/A          QCSchema Molecule instance.
driver                  str          N/A          The type of calculation being performed. One of
                                                  ``energy``, ``gradient``, ``hessian``, or
                                                  ``properties``.
model                   dict         N/A          The quantum chemistry model specification for a
                                                  given operation to compute against.
properties              dict         N/A          Named properties of quantum chemistry
                                                  computations. See
                                                  :ref:`Properties section <json_schema_properties>`
                                                  below.
return_result           varies       N/A          The result requested by the ``driver``. The type
                                                  depends on the ``driver``.
success                 bool         N/A          An indicator for the success of the QC program's
                                                  execution.
======================= ============ ============ ==================================================

The optional fields and corresponding types for a ``qcschema_output`` file are:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
error                   dict         N/A          A complete description of an error-terminated
                                                  computation. See
                                                  :ref:`Error section <json_schema_error>` below.
extras                  dict         N/A          Extra information associated with the input. Also
                                                  specified for
                                                  :ref:`qcschema_input <json_schema_input>`.
id                      str          N/A          An identifier for the input object. Also specified
                                                  for :ref:`qcschema_input <json_schema_input>`.
keywords                dict         N/A          QC program-specific keywords to be used for a
                                                  computation. See details below for IOData-specific
                                                  usages. Also specified for
                                                  :ref:`qcschema_input <json_schema_input>`.
protocols               dict         N/A          Protocols regarding the manipulation of the output
                                                  that results from this input. See
                                                  :ref:`Protocols section <json_schema_protocols>`
                                                  above. Also specified for
                                                  :ref:`qcschema_input <json_schema_input>`.
provenance              dict or list N/A          Information about the file was generated,
                                                  provided, and manipulated. See Provenance section
                                                  above for more information. Also specified for
                                                  :ref:`qcschema_input <json_schema_input>`.
stderr                  str          N/A          The standard error (stderr) of the associated
                                                  computation.
stdout                  str          N/A          The standard output (stdout) of the associated
                                                  computation.
wavefunction            dict         N/A          The wavefunction properties of a QC computation.
                                                  All matrices appear in column-major order. See
                                                  :ref:`Wavefunction <json_schema_wavefunction>`
                                                  section below.
======================= ============ ============ ==================================================

.. _json_schema_properties:

Properties Subschema
^^^^^^^^^^^^^^^^^^^^
The ``properties`` dict contains named properties of quantum chemistry computations. Due to the
variability possible for the contents of an output file, IOData does not guess at which properties
are desired by the user, and stores all properties in the ``extra["output]["properties"]`` dict for
easy retrieval. The current QCSchema standard provides names for the following properties:

======================================== ===========================================================
Field                                    Description
======================================== ===========================================================
calcinfo_nbasis                          The number of basis functions for the computation.
calcinfo_nmo                             The number of molecular orbitals for the computation.
calcinfo_nalpha                          The number of alpha electrons in the computation.
calcinfo_nbeta                           The number of beta electrons in the computation.
calcinfo_natom                           The number of atoms in the computation.
nuclear_repulsion_energy                 The nuclear repulsion energy term.
return_energy                            The energy of the requested method, identical to
                                         ``return_value`` for energy computations.
scf_one_electron_energy                  The one-electron (core Hamiltonian) energy contribution to
                                         the total SCF energy.
scf_two_electron_energy                  The two-electron energy contribution to the total SCF
                                         energy.
scf_vv10_energy                          The VV10 functional energy contribution to the total SCF
                                         energy.
scf_xc_energy                            The functional (XC) energy contribution to the total SCF
                                         energy.
scf_dispersion_correction_energy         The dispersion correction appended to an underlying
                                         functional when a DFT-D method is requested.
scf_dipole_moment                        The X, Y, and Z dipole components.
scf_total_energy                         The total electronic energy of the SCF stage of the
                                         calculation.
scf_iterations                           The number of SCF iterations taken before convergence.
mp2_same_spin_correlation_energy         The portion of MP2 doubles correlation energy from
                                         same-spin (i.e. triplet) correlations.
mp2_opposite_spin_correlation_energy     The portion of MP2 doubles correlation energy from
                                         opposite-spin (i.e. singlet) correlations.
mp2_singles_energy                       The singles portion of the MP2 correlation energy. Zero
                                         except in ROHF.
mp2_doubles_energy                       The doubles portion of the MP2 correlation energy including
                                          same-spin and opposite-spin correlations.
mp2_total_correlation_energy             The MP2 correlation energy.
mp2_correlation_energy                   The MP2 correlation energy.
mp2_total_energy                         The total MP2 energy (MP2 correlation energy + HF energy).
mp2_dipole_moment                        The MP2 X, Y, and Z dipole components.
ccsd_same_spin_correlation_energy        The portion of CCSD doubles correlation energy from
                                         same-spin (i.e. triplet) correlations.
ccsd_opposite_spin_correlation_energy    The portion of CCSD doubles correlation energy from
                                         opposite-spin (i.e. singlet) correlations
ccsd_singles_energy                      The singles portion of the CCSD correlation energy. Zero
                                         except in ROHF.
ccsd_doubles_energy                      The doubles portion of the CCSD correlation energy
                                         including same-spin and opposite-spin correlations.
ccsd_correlation_energy                  The CCSD correlation energy.
ccsd_total_energy                        The total CCSD energy (CCSD correlation energy + HF
                                         energy).
ccsd_dipole_moment                       The CCSD X, Y, and Z dipole components.
ccsd_iterations                          The number of CCSD iterations taken before convergence.
ccsd_prt_pr_correlation_energy           The CCSD(T) correlation energy.
ccsd_prt_pr_total_energy                 The total CCSD(T) energy (CCSD(T) correlation energy + HF
                                         energy).
ccsd_prt_pr_dipole_moment                The CCSD(T) X, Y, and Z dipole components.
ccsd_prt_pr_iterations                   The number of CCSD(T) iterations taken before convergence.
ccsdt_correlation_energy                 The CCSDT correlation energy.
ccsdt_total_energy                       The total CCSDT energy (CCSDT correlation energy + HF
                                         energy).
ccsdt_dipole_moment                      The CCSDT X, Y, and Z dipole components.
ccsdt_iterations                         The number of CCSDT iterations taken before convergence.
ccsdtq_correlation_energy                The CCSDTQ correlation energy.
ccsdtq_total_energy                      The total CCSDTQ energy (CCSDTQ correlation energy + HF
                                         energy).
ccsdtq_dipole_moment                     The CCSDTQ X, Y, and Z dipole components.
ccsdtq_iterations                        The number of CCSDTQ iterations taken before convergence.
======================================== ===========================================================

.. _json_schema_error:

Error Subschema
^^^^^^^^^^^^^^^
The ``error`` dict contains the following fields:

======================= ============ ============ ==================================================
Field                   Type         IOData attr. Description
======================= ============ ============ ==================================================
error_type              str          N/A          The type of error raised during the computation.
error_message           str          N/A          Additional information related to the error, such
                                                  as the backtrace.
extras                  dict         N/A          Additional data associated with the error.
======================= ============ ============ ==================================================

.. _json_schema_wavefunction:

Wavefunction subschema
^^^^^^^^^^^^^^^^^^^^^^
The wavefunction subschema contains the wavefunction properties of a QC computation. All matrices
appear in column-major order. The current QCSchema standard provides names for the following
wavefunction properties:

.. _libint: https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API#solid-harmonic-gaussians-ordering-and-normalization

======================================== ===========================================================
Field                                    Description
======================================== ===========================================================
basis                                    A ``qcschema_basis`` instance for the one-electron AO basis
                                         set. AO basis functions are ordered according to the CCA
                                         standard as implemented in `libint`_.
restricted                               An indicator for a restricted calculation (alpha == beta).
                                         When true, all beta quantites are omitted, since quantity_b
                                         == quantity_a
h_core_a                                 Alpha-spin core (one-electron) Hamiltonian.
h_core_b                                 Beta-spin core (one-electron) Hamiltonian.
h_effective_a                            Alpha-spin effective core (one-electron) Hamiltonian.
h_effective_b                            Beta-spin effective core (one-electron) Hamiltonian.
scf_orbitals_a                           Alpha-spin SCF orbitals.
scf_orbitals_b                           Beta-spin SCF orbitals.
scf_density_a                            Alpha-spin SCF density matrix.
scf_density_b                            Beta-spin SCF density matrix.
scf_fock_a                               Alpha-spin SCF Fock matrix.
scf_fock_b                               Beta-spin SCF Fock matrix.
scf_eigenvalues_a                        Alpha-spin SCF eigenvalues.
scf_eigenvalues_b                        Beta-spin SCF eigenvalues.
scf_occupations_a                        Alpha-spin SCF orbital occupations.
scf_occupations_b                        Beta-spin SCF orbital occupations.
orbitals_a                               Keyword for the primary return alpha-spin orbitals.
orbitals_b                               Keyword for the primary return beta-spin orbitals.
density_a                                Keyword for the primary return alpha-spin density.
density_b                                Keyword for the primary return beta-spin density.
fock_a                                   Keyword for the primary return alpha-spin Fock matrix.
fock_b                                   Keyword for the primary return beta-spin Fock matrix.
eigenvalues_a                            Keyword for the primary return alpha-spin eigenvalues.
eigenvalues_b                            Keyword for the primary return beta-spin eigenvalues.
occupations_a                            Keyword for the primary return alpha-spin orbital
                                         occupations.
occupations_b                            Keyword for the primary return beta-spin orbital
                                         occupations.
======================================== ===========================================================


"""

import json
from typing import TextIO, Union
from warnings import warn

import numpy as np

from .. import __version__
from ..docstrings import document_dump_one, document_load_one
from ..iodata import IOData
from ..periodic import num2sym, sym2num
from ..utils import DumpError, DumpWarning, LineIterator, LoadError, LoadWarning, PrepareDumpError

__all__ = []


PATTERNS = ["*.json"]


@document_load_one(
    "QCSchema",
    ["atnums", "atcorenums", "atcoords", "charge", "nelec", "spinpol"],
    ["atmasses", "bonds", "energy", "g_rot", "lot", "obasis", "obasis_name", "title", "extra"],
)
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Use python standard lib json module to read the file to a dict
    json_in = json.load(lit.fh)
    return _parse_json(json_in, lit)


def _parse_json(json_in: dict, lit: LineIterator) -> dict:
    """Parse data from QCSchema JSON input file.

    QCSchema supports four different schema types: :ref:`qcschema_molecule <json_schema_molecule>`,
    specifying one or more molecules in a single system; `qcschema_basis`, specifying a basis set
    for a molecular system, :ref:`qcschema_input <json_schema_input>`, specifying input to a QC
    program for a specific system; and :ref:`qcschema_output <json_schema_output>`, specifying
    results of a QC program calculation for a specific system along with the input information.

    Parameters
    ----------
    lit
        The line iterator holding the file data.
    json_in
        The JSON dict loaded from file.

    Returns
    -------
    Dictionary with IOData attributes.

    """
    # Remove all null entries and empty dicts in json
    # QCEngine seems to add null entries and empty dicts even for optional and empty keys
    fix_keys = {k: v for k, v in json_in.items() if v is not None}
    fix_subkeys = {}
    for key in fix_keys:
        if isinstance(fix_keys[key], dict):
            fix_subkeys[key] = {k: v for k, v in fix_keys[key].items() if v is not None}
    result = {**fix_keys, **fix_subkeys}
    # Remove empty dicts
    keys = list(result.keys())
    for key in keys:
        if isinstance(result[key], dict) and not bool(result[key]):
            del result[key]

    # Determine schema type
    if "schema_name" in result:
        # Correct for qc_schema vs. qcschema, due to inconsistencies in prior versions
        schema_name = result["schema_name"].replace("qc_schema", "qcschema")
        if schema_name not in {
            "qcschema_molecule",
            "qcschema_basis",
            "qcschema_input",
            "qcschema_output",
        }:
            del result["schema_name"]
    if "schema_name" not in result:
        # Attempt to determine schema type, since some QCElemental files omit this
        warn(
            LoadWarning(
                "QCSchema files should have a `schema_name` key."
                "Attempting to determine schema type...",
                lit.filename,
            ),
            stacklevel=2,
        )
        # Geometry is required in any molecule schema
        if "geometry" in result:
            schema_name = "qcschema_molecule"
        # Check if BSE file, which is too different
        elif "molssi_bse_schema" in result:
            raise LoadError(
                "IOData does not currently support MolSSI BSE Basis JSON.", lit.filename
            )
        # Center_data is required in any basis schema
        elif "center_data" in result:
            schema_name = "qcschema_basis"
        elif "driver" in result:
            schema_name = "qcschema_output" if "return_result" in result else "qcschema_input"
        else:
            raise LoadError("Could not determine `schema_name`.", lit.filename)
    if "schema_version" not in result:
        warn(
            LoadWarning(
                "QCSchema files should have a `schema_version` key."
                "Attempting to load without version number.",
                lit.filename,
            ),
            stacklevel=2,
        )

    if schema_name == "qcschema_molecule":
        return _load_qcschema_molecule(result, lit)
    if schema_name == "qcschema_basis":
        return _load_qcschema_basis(result, lit)
    if schema_name == "qcschema_input":
        return _load_qcschema_input(result, lit)
    if schema_name == "qcschema_output":
        return _load_qcschema_output(result, lit)
    raise LoadError(
        f"Invalid QCSchema type {result['schema_name']}, should be one of "
        "`qcschema_molecule`, `qcschema_basis`, `qcschema_input`, or `qcschema_output`.",
        lit.filename,
    )


def _load_qcschema_molecule(result: dict, lit: LineIterator) -> dict:
    """Load :ref:`qcschema_molecule <json_schema_molecule>` properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    molecule_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``, ``extra``,
        ``nelec`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``g_rot`` & ``title`` keys and corresponding values
        as well.

    """
    # All Topology properties are found in the "molecule" key
    molecule_dict = _parse_topology_keys(result, lit)

    # Move extra keys to molecule dict, for consistency with input/output
    extra_dict = {"molecule": molecule_dict["extra"]}
    molecule_dict["extra"] = extra_dict
    molecule_dict["extra"]["schema_name"] = "qcschema_molecule"

    return molecule_dict


def _parse_topology_keys(mol: dict, lit: LineIterator) -> dict:
    """Load topology properties from old QCSchema Molecule specifications.

    The qcschema_molecule v2 specification requires a topology for every file, specified in the
    ``molecule`` key, containing at least the keys ``schema_name``, ``schema_version``, ``symbols``,
    ``geometry``, ``molecular_charge``, ``molecular_multiplicity``, and ``provenance``. This schema
    is currently used in QCElemental (and thus the QCArchive ecosystem).

    qcschema_molecule v1 only exists as the specification on the QCSchema website, and seems never
    to have been implemented in QCArchive. It is possible to accept v1 input, since all required
    keys for v2 exist as keys in v1, but it is preferable to convert those files to v2 explicitly.

    Parameters
    ----------
    mol
        The 'molecule' key from the QCSchema input or output file, or the full result for a QCSchema
        Molecule file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    topology_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``, ``extra``,
        ``nelec`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``g_rot`` & ``title`` keys and corresponding values
        as well.

    """
    # Make sure required topology properties are present
    # NOTE: currently molecular_charge and molecular_multiplicity have default values,
    # so some programs may not provide those required keys
    # QCEngineRecords files follow Molecule v1 for some reason, despite being uploaded in 2019
    should_be_required_keys = {"schema_name", "schema_version", "provenance"}
    topology_keys = {
        "symbols",
        "geometry",
    }
    for key in should_be_required_keys:
        if key not in mol:
            warn(
                LoadWarning(f"QCSchema files should have a '{key}' key.", lit.filename),
                stacklevel=2,
            )
    for key in topology_keys:
        if key not in mol:
            raise LoadError(f"QCSchema topology requires '{key}' key", lit.filename)

    topology_dict = {}
    extra_dict = {}

    # Save schema name & version
    extra_dict["schema_name"] = "qcschema_molecule"
    extra_dict["schema_version"] = _version_check(mol, 2, "qcschema_molecule", lit)

    # Geometry is in a flattened list, convert to N x 3
    topology_dict["atcoords"] = np.array(mol["geometry"]).reshape(-1, 3)
    atnums = np.array([sym2num[symbol.title()] for symbol in mol["symbols"]])
    topology_dict["atnums"] = atnums
    atcorenums = atnums.astype(float)
    topology_dict["atcorenums"] = atcorenums
    # Check for missing charge, warn that this is a required field
    if "molecular_charge" not in mol:
        warn(
            LoadWarning(
                "Missing 'molecular_charge' key."
                "Some QCSchema writers omit this key for default value 0.0,"
                "Ensure this value is correct.",
                lit.filename,
            ),
            stacklevel=2,
        )
        formal_charge = 0.0
    else:
        formal_charge = mol["molecular_charge"]
    # Determine number of electrons from atomic numbers and formal charge
    topology_dict["charge"] = formal_charge
    topology_dict["nelec"] = np.sum(atnums) - formal_charge
    # Check for missing mult, warn that this is a required field
    if "molecular_multiplicity" not in mol:
        warn(
            LoadWarning(
                "Missing 'molecular_multiplicity' key."
                "Some QCSchema writers omit this key for default value 1,"
                "Ensure this value is correct.",
                lit.filename,
            ),
            stacklevel=2,
        )
        topology_dict["spinpol"] = 0
    else:
        mult = mol["molecular_multiplicity"]
        topology_dict["spinpol"] = mult - 1

    # Provenance is optional in v1
    if "provenance" in mol:
        extra_dict["provenance"] = _parse_provenance(
            mol["provenance"], lit, "qcschema_molecule", False
        )

    # Check for optional keys
    # Load ghost atoms as atoms with zero effective core charge
    if "real" in mol:
        atcorenums[~np.array(mol["real"])] = 0.0
    # Load atom masses to array, canonical weights assumed if masses not given
    if "masses" in mol and "mass_numbers" in mol:
        warn(
            LoadWarning(
                "Both `masses` and `mass_numbers` given. "
                "Both values will be written to `extra` dict.",
                lit.filename,
            ),
            stacklevel=2,
        )
        extra_dict["mass_numbers"] = np.array(mol["mass_numbers"])
        extra_dict["masses"] = np.array(mol["masses"])
    elif "masses" in mol:
        topology_dict["atmasses"] = np.array(mol["masses"])
    elif "mass_numbers" in mol:
        topology_dict["atmasses"] = np.array(mol["mass_numbers"])
    # Load bonds: list of tuple (atom1, atom2, bond_order)
    # Note: The QCSchema spec allows for non-integer bond_orders, these are forced to integers here
    # in accordance with current IOData specification
    if "connectivity" in mol:
        topology_dict["bonds"] = np.array(mol["connectivity"], dtype=int)
    # Check for fragment keys
    # List fragment indices in nested list (likely is a jagged array)
    if "fragments" in mol:
        fragments = mol["fragments"]
        extra_dict["fragments"] = {"indices": [np.array(fragment) for fragment in fragments]}
        if "fragment_charges" in mol:
            extra_dict["fragments"]["charges"] = np.array(mol["fragment_charges"])
        if "fragment_multiplicities" in mol:
            extra_dict["fragments"]["multiplicities"] = np.array(mol["fragment_multiplicities"])
    if "fix_symmetry" in mol:
        topology_dict["g_rot"] = mol["fix_symmetry"]
    if "fix_orientation" in mol:
        extra_dict["fix_orientation"] = mol["fix_orientation"]

    # Check for other extras
    # name, comment, fix_com, fix_symmetry, fix_orientation
    if "name" in mol:
        topology_dict["title"] = mol["name"]
    if "comment" in mol:
        extra_dict["comment"] = mol["comment"]
    if "fix_com" in mol:
        extra_dict["fix_com"] = mol["fix_com"]
    if "identifiers" in mol:
        extra_dict["identifiers"] = mol["identifiers"]
    if "validated" in mol:
        extra_dict["qcel_validated"] = mol["validated"]
    if "atom_labels" in mol:
        extra_dict["atom_labels"] = mol["atom_labels"]
    if "atomic_numbers" in mol:
        extra_dict["atomic_numbers"] = np.array(mol["atomic_numbers"])
    if "id" in mol:
        extra_dict["id"] = mol["id"]
    if "extras" in mol:
        extra_dict["extras"] = mol["extras"]

    if extra_dict:
        topology_dict["extra"] = extra_dict

    molecule_keys = {
        "schema_name",
        "schema_version",
        "validated",
        "symbols",
        "geometry",
        "name",
        "identifiers",
        "comment",
        "molecular_charge",
        "molecular_multiplicity",
        "masses",
        "real",
        "atom_labels",
        "atomic_numbers",
        "mass_numbers",
        "connectivity",
        "fragments",
        "fragment_charges",
        "fragment_multiplicities",
        "fix_com",
        "fix_orientation",
        "fix_symmetry",
        "provenance",
        "id",
        "extras",
    }
    passthrough_dict = _find_passthrough_dict(mol, molecule_keys)
    if passthrough_dict:
        topology_dict["extra"]["unparsed"] = passthrough_dict

    return topology_dict


def _version_check(result: dict, max_version: float, schema_name: str, lit: LineIterator) -> str:
    """Check whether the QCSchema version is a known version.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    max_version
        The highest (most recent) known version for the QCSchema type.
    schema_name
        The ``schema_name`` key of a QCSchema file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    version
        The version of the QCSchema file, -1 if unknown version.
    """
    try:
        version = result["schema_version"]
    except KeyError:
        version = -1
    if float(version) < 0 or float(version) > max_version:
        warn(
            LoadWarning(
                f"Unknown {schema_name} version {version}, " "loading may produce invalid results",
                lit.filename,
            ),
            stacklevel=2,
        )
    return version


def _find_passthrough_dict(result: dict, keys: set) -> dict:
    """Find all keys not specified for a given schema.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    keys
        The set of expected keys for a given schema type.

    Returns
    -------
    passthrough_dict
        All unparsed keys remaining in the parsed dict.
    """
    # Avoid altering original dict
    result = result.copy()

    passthrough_dict = {}
    parsed_keys = keys.intersection(result.keys())
    for key in parsed_keys:
        del result[key]
    if len(result) > 0:
        passthrough_dict = result

    return passthrough_dict


def _load_qcschema_basis(_result: dict, _lit: LineIterator) -> dict:
    """Load qcschema_basis properties.

    Parameters
    ----------
    _result
        The JSON dict loaded from file.
    _lit
        The line iterator holding the file data.

    Returns
    -------
    basis_dict
        ...

    Raises
    ------
    NotImplementedError
        QCSchema Basis schema is not yet implemented in IOData.

    """
    # basis_dict = {}
    # return basis_dict
    raise NotImplementedError("qcschema_basis is not yet implemented in IOData.")


def _parse_basis_keys(_basis: dict, _lit: LineIterator) -> dict:
    """Parse basis keys for a QCSchema input, output, or basis file.

    Parameters
    ----------
    _basis
        The basis dictionary from a QCSchema basis file or QCSchema input or output 'method' key.
    _lit
        The line iterator holding the file data.

    Returns
    -------
    basis_dict
        Dictionary containing ...

    Raises
    ------
    NotImplementedError
        QCSchema Basis schema is not yet implemented in IOData.

    """
    raise NotImplementedError("qcschema_basis is not yet implemented in IOData.")


def _load_qcschema_input(result: dict, lit: LineIterator) -> dict:
    """Load qcschema_input properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    input_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``, ``extra``, ``lot``,
        ``nelec``, ``obasis_name`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``g_rot``, ``obasis``, ``run_type`` & ``title``
        keys and corresponding values as well.
    """
    extra_dict = {}
    input_dict = _parse_input_keys(result, lit)
    extra_dict["input"] = input_dict["extra"]

    if "molecule" not in result:
        raise LoadError("QCSchema Input requires 'molecule' key", lit.filename)
    molecule_dict = _parse_topology_keys(result["molecule"], lit)
    input_dict.update(molecule_dict)
    extra_dict["molecule"] = molecule_dict["extra"]
    input_dict["extra"] = extra_dict
    input_dict["extra"]["schema_name"] = "qcschema_input"

    return input_dict


def _parse_input_keys(result: dict, lit: LineIterator) -> dict:
    """Parse input keys for QCSchema input or output files.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    input_dict
        Output dictionary containing ``extra``, ``lot`` and ``obasis_name`` keys and corresponding
        values.
        It may contain ``obasis`` & ``run_type`` keys and corresponding values as well.

    """
    # QCEngineRecords input files don't actually specify a name or version
    should_be_required_keys = {"schema_name", "schema_version"}
    input_keys = {"molecule", "driver", "model"}
    for key in should_be_required_keys:
        if key not in result:
            warn(
                LoadWarning(f"QCSchema files should have a '{key}' key.", lit.filename),
                stacklevel=2,
            )
    for key in input_keys:
        if key not in result:
            raise LoadError(f"QCSchema `qcschema_input` file requires '{key}' key", lit.filename)
    # Store all extra keys in extra_dict and gather at end
    input_dict = {}
    extra_dict = {}

    # Save schema name & version
    extra_dict["schema_name"] = "qcschema_input"
    extra_dict["schema_version"] = _version_check(result, 1, "qcschema_input", lit)

    # Load driver
    extra_dict["driver"] = _parse_driver(result["driver"], lit)

    # Load model & call basis helper if needed
    model = _parse_model(result["model"], lit)
    input_dict.update(model)
    extra_dict["model"] = model["extra"]

    # Load keywords & store
    # Currently, only the IOData run_type attribute is specifically parsed from keywords, but this
    # is a good space for passing additional IOData-specific keywords, given that the official spec
    # treats this as program-specific territory. If run_type is not one of the values expected by
    # IOData, it will be stored only in the extra_dict.
    if "keywords" in result:
        keywords_dict = result["keywords"]
        if "run_type" in keywords_dict and keywords_dict["run_type"].lower() in {
            "energy",
            "energy_force",
            "opt",
            "scan",
            "freq",
        }:
            input_dict["run_type"] = keywords_dict["run_type"]
        extra_dict["keywords"] = keywords_dict
    # Check for extras
    if "extras" in result:
        extra_dict["extras"] = result["extras"]
    # Check for ID
    if "id" in result:
        extra_dict["id"] = result["id"]
    # Load protocols
    if "protocols" in result:
        extra_dict["protocols"] = _parse_protocols(result["protocols"], lit)
    # Check for provenance
    if "provenance" in result:
        extra_dict["provenance"] = _parse_provenance(result["provenance"], lit, "qcschema_input")

    input_dict["extra"] = extra_dict

    input_keys = {
        "schema_name",
        "schema_version",
        "molecule",
        "driver",
        "model",
        "extras",
        "id",
        "keywords",
        "protocols",
        "provenance",
    }
    passthrough_dict = _find_passthrough_dict(result, input_keys)
    if passthrough_dict:
        input_dict["extra"]["unparsed"] = passthrough_dict

    return input_dict


def _parse_driver(driver: str, lit: LineIterator) -> str:
    """Load driver properties from QCSchema.

    Parameters
    ----------
    driver
        The ``driver`` key from the QCSchema input.
    lit
        The line iterator holding the file data.

    Returns
    -------
    driver_dict
        The driver for the QCSchema file, specifying what type of calculation is being performed.

    Raises
    ------
    LoadError
        If driver is not one of {"energy", "gradient", "hessian", "properties"}.

    Notes
    -----
    This keyword is similar to, but not really interchangeable with, the ``run_type`` IOData
    attribute. In order to specify the ``run_type``, add it to the ``keywords`` dictionary.

    """
    if driver not in ["energy", "gradient", "hessian", "properties"]:
        raise LoadError(
            "QCSchema driver must be one of `energy`, `gradient`, `hessian`, or `properties`",
            lit.filename,
        )
    return driver


def _parse_model(model: dict, lit: LineIterator) -> dict:
    """Load :ref:`model <json_schema_model>` properties from QCSchema.

    Parameters
    ----------
    model
        The dictionary corresponding to the 'model' key for a QCSchema input or output file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    model_dict
        Output dictionary containing ``lot`` and ``obasis_name`` keys and corresponding values.
        It may contain ``obasis`` and ``extra`` keys and corresponding values as well.

    """
    model_dict = {}
    extra_dict = {}

    if "method" not in model:
        raise LoadError("QCSchema `model` requires a `method`", lit.filename)
    model_dict["lot"] = model["method"]
    # QCEngineRecords doesn't give an empty string for basis-free methods, omits req'd key instead
    if "basis" not in model:
        warn(
            LoadWarning(
                "Model `basis` key should be given. Assuming basis-free method.", lit.filename
            ),
            stacklevel=2,
        )
    elif isinstance(model["basis"], str):
        if model["basis"] == "":
            warn(
                LoadWarning(
                    "QCSchema `basis` could not be read and will be omitted."
                    "Unless model is for a basis-free method, check input file.",
                    lit.filename,
                ),
                stacklevel=2,
            )
        else:
            model_dict["obasis_name"] = model["basis"]
    elif isinstance(model["basis"], dict):
        basis = _parse_basis_keys(model["basis"], lit)
        model_dict.update(basis)
        extra_dict["basis"] = basis["extra"]

    model_dict["extra"] = extra_dict
    return model_dict


def _parse_protocols(protocols: dict, lit: LineIterator) -> dict:
    """Load :ref:`protocols <json_schema_protocols>` properties from QCSchema.

    Parameters
    ----------
    protocols
        Protocols key from a QCSchema input or output file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    protocols_dict
        Protocols dictionary containing instructions for the manipulation of output generated from
        this input.

    """
    if "wavefunction" not in protocols:
        warn(
            LoadWarning(
                "Protocols `wavefunction` key not specified, no properties will be kept.",
                lit.filename,
            ),
            stacklevel=2,
        )
        wavefunction = "none"
    else:
        wavefunction = protocols["wavefunction"]
    if "stdout" not in protocols:
        warn(
            LoadWarning("Protocols `stdout` key not specified, stdout will be kept.", lit.filename),
            stacklevel=2,
        )
        keep_stdout = True
    else:
        keep_stdout = protocols["stdout"]
    protocols_dict = {}
    if wavefunction not in {"all", "orbitals_and_eigenvalues", "return_results", "none"}:
        raise LoadError("Invalid `protocols` `wavefunction` keyword.", lit.filename)
    protocols_dict["keep_wavefunction"] = wavefunction
    if not isinstance(keep_stdout, bool):
        raise LoadError("`protocols` `stdout` option must be a boolean.", lit.filename)
    protocols_dict["keep_stdout"] = keep_stdout
    return protocols_dict


def _load_qcschema_output(result: dict, lit: LineIterator) -> dict:
    """Load :ref:`qcschema_output <json_schema_output>` properties.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    output_dict
        Output dictionary containing ``atcoords``, ``atnums``, ``charge``, ``extra``, ``lot``,
        ``nelec``, ``obasis_name`` & ``spinpol`` keys and corresponding values.
        It may contain ``atmasses``, ``bonds``, ``energy``, ``g_rot``, ``obasis``, ``run_type`` &
        ``title`` keys and corresponding values as well.

    """
    extra_dict = {}
    output_dict = _parse_output_keys(result, lit)
    extra_dict["output"] = output_dict["extra"]

    if "molecule" not in result:
        raise LoadError("QCSchema Input requires 'molecule' key", lit.filename)
    molecule_dict = _parse_topology_keys(result["molecule"], lit)
    output_dict.update(molecule_dict)
    extra_dict["molecule"] = molecule_dict["extra"]

    input_dict = _parse_input_keys(result, lit)
    output_dict.update(input_dict)
    extra_dict["input"] = input_dict["extra"]
    output_dict["extra"] = extra_dict
    output_dict["extra"]["schema_name"] = "qcschema_output"

    return output_dict


def _parse_output_keys(result: dict, lit: LineIterator) -> dict:
    """Parse output keys for QCSchema output files.

    Parameters
    ----------
    result
        The JSON dict loaded from file.
    lit
        The line iterator holding the file data.

    Returns
    -------
    output_dict
        Output dictionary containing ``extra`` key and corresponding values.
        It may contain ``energy`` key and corresponding values as well.

    """
    should_be_required_keys = {"schema_name", "schema_version"}
    output_keys = {"provenance", "properties", "success", "return_result"}
    for key in should_be_required_keys:
        if key not in result:
            warn(
                LoadWarning(f"QCSchema files should have a '{key}' key.", lit.filename),
                stacklevel=2,
            )
    for key in output_keys:
        if key not in result:
            raise LoadError(f"QCSchema `qcschema_output` file requires '{key}' key", lit.filenam)

    # Store all extra keys in extra_dict and gather at end
    output_dict = {}
    extra_dict = {}

    extra_dict["schema_name"] = "qcschema_output"
    extra_dict["schema_version"] = _version_check(result, 2, "qcschema_output", lit)

    extra_dict["return_result"] = result["return_result"]
    extra_dict["success"] = result["success"]

    # Parse properties
    properties = result["properties"]
    if "return_energy" in properties:
        output_dict["energy"] = properties["return_energy"]
    extra_dict["properties"] = properties

    if "error" in result:
        extra_dict["error"] = result["error"]
    if "stderr" in result:
        extra_dict["stderr"] = result["stderr"]
    if "stdout" in result:
        extra_dict["stderr"] = result["stdout"]
    if "wavefunction" in result:
        extra_dict["wavefunction"] = result["wavefunction"]

    output_dict["extra"] = extra_dict

    output_keys = {
        "schema_name",
        "schema_version",
        "molecule",
        "driver",
        "model",
        "extras",
        "id",
        "keywords",
        "protocols",
        "provenance",
        "properties",
        "success",
        "return_result",
    }
    passthrough_dict = _find_passthrough_dict(result, output_keys)
    if passthrough_dict:
        output_dict["extra"]["unparsed"] = passthrough_dict

    return output_dict


def _parse_provenance(
    provenance: Union[list[dict], dict], lit: LineIterator, source: str, append=True
) -> Union[list[dict], dict]:
    """Load :ref:`provenance <json_schema_provenance>` properties from QCSchema.

    Parameters
    ----------
    provenance
        QCSchema JSON provenance dictionary.
    lit
        The line iterator holding the file data.
    source
        The schema type {``qcschema_molecule``, ``qcschema_input``, ``qcschema_output``} associated
        with this provenance data.
    append
        Append IOData provenance entry to provenance list?

    Returns
    -------
    base_provenance
        The provenance data for a QCSchema file.
    """
    if isinstance(provenance, dict):
        if "creator" not in provenance:
            raise LoadError(f"`{source}` provenance requires `creator` key", lit.filename)
        if append:
            base_provenance = [provenance]
        else:
            return provenance
    elif isinstance(provenance, list):
        for prov in provenance:
            if "creator" not in prov:
                raise LoadError(f"`{source}` provenance requires `creator` key", lit.filename)
        base_provenance = provenance
    else:
        raise LoadError(f"Invalid `{source}` provenance type", lit.filename)
    if append:
        base_provenance.append(
            {"creator": "IOData", "version": __version__, "routine": "iodata.formats.json.load_one"}
        )
    return base_provenance


def prepare_dump(filename: str, data: IOData):
    """Check the compatibility of the IOData object with QCScheme.

    Parameters
    ----------
    filename
        The file to be written to, only used for error messages.
    data
        The IOData instance to be checked.

    """
    if "schema_name" not in data.extra:
        raise PrepareDumpError(
            "Cannot write qcschema file without 'schema_name' defined.", filename
        )
    schema_name = data.extra["schema_name"]
    if schema_name == "qcschema_basis":
        raise PrepareDumpError(f"{schema_name} not yet implemented in IOData.", filename)


@document_dump_one(
    "QCSchema",
    ["atnums", "atcoords", "charge", "spinpol"],
    ["title", "atcorenums", "atmasses", "bonds", "g_rot", "extra"],
)
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    schema_name = data.extra["schema_name"]

    if schema_name == "qcschema_molecule":
        return_dict = _dump_qcschema_molecule(f, data)
    elif schema_name == "qcschema_basis":
        raise NotImplementedError(f"{schema_name} not yet implemented in IOData.")
        # return_dict = _dump_qcschema_basis(f, data)
    elif schema_name == "qcschema_input":
        return_dict = _dump_qcschema_input(f, data)
    elif schema_name == "qcschema_output":
        return_dict = _dump_qcschema_output(f, data)
    else:
        raise DumpError(
            "'schema_name' must be one of 'qcschema_molecule', 'qcschema_basis'"
            "'qcschema_input' or 'qcschema_output'.",
            f,
        )
    json.dump(return_dict, f, indent=4)


def _dump_qcschema_molecule(f: TextIO, data: IOData) -> dict:
    """Dump relevant attributes from IOData to :ref:`qcschema_molecule <json_schema_molecule>`.

    Parameters
    ----------
    f
        The file being written, used for error and warning messages only.
    data
        The IOData instance to dump to file.

    Returns
    -------
    molecule_dict
        The dict that will produce the QCSchema JSON file.

    """
    molecule_dict = {"schema_name": "qcschema_molecule", "schema_version": 2.0}

    # Gather required field data
    if data.atnums is None or data.atcoords is None:
        raise DumpError("qcschema_molecule requires `atnums` and `atcoords` fields.", f)
    molecule_dict["symbols"] = [num2sym[num] for num in data.atnums]
    molecule_dict["geometry"] = list(data.atcoords.flatten())

    # Should be required field data
    if data.charge is None or data.spinpol is None:
        warn(
            DumpWarning(
                "`charge` and `spinpol` should be given to write qcschema_molecule file:"
                "QCSchema defaults to charge = 0 and multiplicity = 1 if no values given.",
                f,
            ),
            stacklevel=2,
        )
    if data.charge is not None:
        molecule_dict["molecular_charge"] = data.charge
    if data.spinpol is not None:
        molecule_dict["molecular_multiplicity"] = data.spinpol + 1

    # Check for other QCSchema keys from IOData keys
    if data.title:
        molecule_dict["name"] = data.title
    molecule_dict["real"] = [bool(atcorenum != 0) for atcorenum in data.atcorenums]
    # "masses" could be overwritten below (for QCSchema passthrough)
    if data.atmasses is not None:
        molecule_dict["masses"] = data.atmasses.tolist()
    if data.bonds is not None:
        molecule_dict["connectivity"] = [[int(i) for i in bond] for bond in data.bonds]
    if data.g_rot:
        molecule_dict["fix_symmetry"] = data.g_rot

    # Check for other QCSchema keys from IOData extra dict
    if "qcel_validated" in data.extra["molecule"]:
        molecule_dict["validated"] = data.extra["molecule"]["qcel_validated"]
    if "identifiers" in data.extra["molecule"]:
        molecule_dict["identifiers"] = data.extra["molecule"]["identifiers"]
    if "comment" in data.extra["molecule"]:
        molecule_dict["comment"] = data.extra["molecule"]["comment"]
    if "atom_labels" in data.extra["molecule"]:
        molecule_dict["atom_labels"] = data.extra["molecule"]["atom_labels"]
    if "atomic_numbers" in data.extra["molecule"]:
        molecule_dict["atomic_numbers"] = data.extra["molecule"]["atomic_numbers"].tolist()
    if "masses" in data.extra["molecule"]:
        molecule_dict["masses"] = data.extra["molecule"]["masses"].tolist()
    if "mass_numbers" in data.extra["molecule"]:
        molecule_dict["mass_numbers"] = data.extra["molecule"]["mass_numbers"].tolist()
    if "fragments" in data.extra["molecule"]:
        if "indices" in data.extra["molecule"]["fragments"]:
            molecule_dict["fragments"] = [
                fragment.tolist() for fragment in data.extra["molecule"]["fragments"]["indices"]
            ]
        if "indices" in data.extra["molecule"]["fragments"]:
            molecule_dict["fragment_charges"] = data.extra["molecule"]["fragments"][
                "charges"
            ].tolist()
        if "indices" in data.extra["molecule"]["fragments"]:
            molecule_dict["fragment_multiplicities"] = data.extra["molecule"]["fragments"][
                "multiplicities"
            ].tolist()
    if "fix_com" in data.extra["molecule"]:
        molecule_dict["fix_com"] = data.extra["molecule"]["fix_com"]
    if "fix_orientation" in data.extra["molecule"]:
        molecule_dict["fix_orientation"] = data.extra["molecule"]["fix_orientation"]
    molecule_dict["provenance"] = _dump_provenance(f, data, "molecule")
    if "id" in data.extra["molecule"]:
        molecule_dict["id"] = data.extra["molecule"]["id"]
    if "extras" in data.extra["molecule"]:
        molecule_dict["extras"] = data.extra["molecule"]["extras"]
    if "unparsed" in data.extra["molecule"]:
        for k in data.extra["molecule"]["unparsed"]:
            molecule_dict[k] = data.extra["molecule"]["unparsed"][k]

    return molecule_dict


def _dump_provenance(f: TextIO, data: IOData, source: str) -> Union[list[dict], dict]:
    """Generate the :ref:`provenance <json_schema_provenance>` information.

    This is used when dumping an IOData instance to QCSchema.

    Parameters
    ----------
    f
        The file being written, used for error and warning messages only.
    data
        The IOData instance to dump to file.
    source
        The `extra` dict location for the dump file, to find provenance data.

    Returns
    -------
    provenance
        The provenance information for the IOData instance.

    """
    new_provenance = {
        "creator": "IOData",
        "version": __version__,
        "routine": "iodata.formats.json.dump_one",
    }
    if "provenance" in data.extra[source]:
        provenance = data.extra[source]["provenance"]
        if isinstance(provenance, dict):
            return [provenance, new_provenance]
        if isinstance(provenance, list):
            provenance.append(new_provenance)
            return provenance
        raise DumpError("QCSchema provenance must be either a dict or list of dicts.", f)
    return new_provenance


def _dump_qcschema_input(f: TextIO, data: IOData) -> dict:
    """Dump relevant attributes from IOData to :ref:`qcschema_input <json_schema_input>`.

    Using this function requires keywords to be stored in two locations in the ``extra`` dict:
    a ``molecule`` dict for the QCSchema Molecule extra keys, and an ``input`` dict for the QCSchema
    Input extra keys.

    Parameters
    ----------
    f
        The file being written, used for error and warning messages only.
    data
        The IOData instance to dump to file.

    Returns
    -------
    input_dict
        The dict that will produce the QCSchema JSON file.

    """
    input_dict = {"schema_name": "qcschema_input", "schema_version": 2.0}

    # Gather required field data
    input_dict["molecule"] = _dump_qcschema_molecule(f, data)
    if "driver" not in data.extra["input"]:
        raise DumpError("qcschema_input requires `driver` field in extra['input'].", f)
    if data.extra["input"]["driver"] not in {"energy", "gradient", "hessian", "properties"}:
        raise DumpError(
            "QCSchema driver must be one of `energy`, `gradient`, `hessian`, or `properties`", f
        )
    input_dict["driver"] = data.extra["input"]["driver"]
    if "model" not in data.extra["input"]:
        raise DumpError("qcschema_input requires `model` field in extra['input'].", f)
    input_dict["model"] = {}
    if data.lot is None:
        raise DumpError("qcschema_input requires specifed `lot`.", f)
    input_dict["model"]["method"] = data.lot
    if data.obasis_name is None and "basis" not in data.extra["input"]["model"]:
        input_dict["model"]["basis"] = ""
    if "basis" in data.extra["input"]["model"]:
        raise NotImplementedError("qcschema_basis is not yet supported in IOData.")
    input_dict["model"]["basis"] = data.obasis_name
    if "keywords" in data.extra["input"]:
        input_dict["keywords"] = data.extra["input"]["keywords"]
    if "extras" in data.extra["input"]:
        input_dict["extras"] = data.extra["input"]["extras"]
    if "id" in data.extra["input"]:
        input_dict["id"] = data.extra["input"]["id"]
    if "protocols" in data.extra["input"]:
        input_dict["protocols"] = {}
        # Remove 'keep_' from protocols keys (added in IOData for readability)
        for keep in data.extra["input"]["protocols"]:
            input_dict["protocols"][keep[5:]] = data.extra["input"]["protocols"][keep]
    input_dict["provenance"] = _dump_provenance(f, data, "input")
    if "unparsed" in data.extra["input"]:
        for k in data.extra["input"]["unparsed"]:
            input_dict[k] = data.extra["input"]["unparsed"][k]

    return input_dict


def _dump_qcschema_output(f: TextIO, data: IOData) -> dict:
    """Dump relevant attributes from IOData to :ref:`qcschema_output <json_schema_output>`.

    Using this function requires keywords to be stored in three locations in the ``extra`` dict:
    a ``molecule`` dict for the QCSchema Molecule extra keys, an ``input`` dict for the QCSchema
    Input extra keys, and an ``output`` dict for the QCSchema Output extra keys.

    Parameters
    ----------
    f
        The file being written, used for error and warning messages only.
    data
        The IOData instance to dump to file.

    Returns
    -------
    output_dict
        The dict that will produce the QCSchema JSON file.

    """
    output_dict = {"schema_name": "qcschema_output", "schema_version": 2.0}

    # Gather required field data
    # Gather required field data
    output_dict["molecule"] = _dump_qcschema_molecule(f, data)
    if "driver" not in data.extra["input"]:
        raise DumpError("qcschema_output requires `driver` field in extra['input'].", f)
    if data.extra["input"]["driver"] not in {"energy", "gradient", "hessian", "properties"}:
        raise DumpError(
            "QCSchema driver must be one of `energy`, `gradient`, `hessian`, or `properties`", f
        )
    output_dict["driver"] = data.extra["input"]["driver"]
    if "model" not in data.extra["input"]:
        raise DumpError("qcschema_output requires `model` field in extra['input'].", f)
    output_dict["model"] = {}
    if data.lot is None:
        raise DumpError("qcschema_output requires specifed `lot`.", f)
    output_dict["model"]["method"] = data.lot
    if data.obasis_name is None and "basis" not in data.extra["input"]["model"]:
        warn(
            DumpWarning(
                "No basis name given. QCSchema assumes this signifies a basis-free method; to"
                "avoid this warning, specify `obasis_name` as an empty string.",
                f,
            ),
            stacklevel=2,
        )
    if "basis" in data.extra["input"]["model"]:
        raise NotImplementedError("qcschema_basis is not yet supported in IOData.")
    output_dict["model"]["basis"] = data.obasis_name
    if "properties" not in data.extra["output"]:
        raise DumpError("qcschema_output requires `properties` field in extra['output'].", f)
    output_dict["properties"] = data.extra["output"]["properties"]
    if data.energy is not None:
        output_dict["properties"]["return_energy"] = data.energy
        if output_dict["driver"] == "energy":
            output_dict["return_result"] = data.energy
    if "return_result" not in output_dict and "return_result" not in data.extra["output"]:
        raise DumpError("qcschema_output requires `return_result` field in extra['output'].", f)
    if "return_result" in data.extra["output"]:
        output_dict["return_result"] = data.extra["output"]["return_result"]
    if "keywords" in data.extra["input"]:
        output_dict["keywords"] = data.extra["input"]["keywords"]
    if "extras" in data.extra["input"]:
        output_dict["extras"] = data.extra["input"]["extras"]
    if "id" in data.extra["input"]:
        output_dict["id"] = data.extra["input"]["id"]
    if "protocols" in data.extra["input"]:
        output_dict["protocols"] = {}
        # Remove 'keep_' from protocols keys (added in IOData for readability)
        for keep in data.extra["input"]["protocols"]:
            output_dict["protocols"][keep[5:]] = data.extra["input"]["protocols"][keep]
    if "error" in data.extra["output"]:
        output_dict["error"] = data.extra["output"]["error"]
    if "stderr" in data.extra["output"]:
        output_dict["stderr"] = data.extra["output"]["stderr"]
    if "stdout" in data.extra["output"]:
        output_dict["stderr"] = data.extra["output"]["stdout"]
    if "wavefunction" in data.extra["output"]:
        output_dict["wavefunction"] = data.extra["output"]["wavefunction"]
    output_dict["provenance"] = _dump_provenance(f, data, "input")
    if "unparsed" in data.extra["input"]:
        for k in data.extra["input"]["unparsed"]:
            output_dict[k] = data.extra["input"]["unparsed"][k]

    return output_dict
