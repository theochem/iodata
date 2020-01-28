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
"""Module for handling input/output from different file formats."""


import attr
import numpy as np

from .basis import MolecularBasis
from .orbitals import MolecularOrbitals
from .utils import Cube


__all__ = ['IOData']


def convert_array_to(dtype):
    """Return a function to convert arrays to the given type."""
    def converter(array):
        if array is None:
            return None
        return np.array(array, copy=False, dtype=dtype)
    return converter


def validate_shape(*shape):
    """Return a function to validate the shape of an array."""
    def validator(obj, attrname, value):
        if value is None:
            return
        myshape = tuple([obj.natom if size == 'natom' else size for size in shape])
        if len(myshape) != len(value.shape):
            raise TypeError('Expect ndim {} for attribute {}, got {}'.format(
                len(myshape), attrname, len(value.shape)))
        for axis, size in enumerate(myshape):
            if size is None:
                continue
            if size != value.shape[axis]:
                raise TypeError(
                    'Expect size {} for axis {} of attribute {}, got {}'.format(
                        size, axis, attrname, value.shape[axis]))
    return validator


# pylint: disable=too-many-instance-attributes
@attr.s(auto_attribs=True, slots=True)
class IOData:
    """A container class for data loaded from (or to be written to) a file.

    In principle, the constructor accepts any keyword argument, which is
    stored as an attribute. All attributes are optional. Attributes can be
    set are removed after the IOData instance is constructed. The following
    attributes are supported by at least one of the io formats:

    Attributes
    ----------
    atcharges
        A dictionary where keys are names of charge definitions and values are
        arrays with atomic charges (size N).
    atcoords
        A (N, 3) float array with Cartesian coordinates of the atoms.
    atcorenums
        A (N,) float array with pseudo-potential core charges. The matrix
        elements corresponding to ghost atoms are zero.
    atffparams
        A dictionary with arrays of atomic force field parameters (typically
        non-bonded). Keys include 'charges', 'vdw_radii', 'sigmas', 'epsilons',
        'alphas' (atomic polarizabilities), 'c6s', 'c8s', 'c10s', 'buck_as',
        'buck_bs', 'lj_as', 'core_charges', 'valence_charges', 'valence_widths',
        etc. Not all of them have to be present, depending on the use case.
    atfrozen
        A (N,) bool array with frozen atoms. (All atoms are free if this
        attribute is not set.)
    atgradient
        A (N, 3) float array with the first derivatives of the energy w.r.t.
        Cartesian atomic displacements.
    athessian
        A (3*N, 3*N) array containing the energy Hessian w.r.t Cartesian atomic
        displacements.
    atmasses
        A (N,) float array with atomic masses
    atnums
        A (N,) int vector with the atomic numbers.
    basisdef
        A basis set definition, i.e. a dictionary whose keys are symbols (of
        chemical elements), atomic numbers (similar to previous, str to make
        distinction with following) or an atom index (integer referring to a
        specific atom in a molecule). The format of the values is to be decided
        when implementing a load function for basis set definitions.
    bonds
        An (nbond, 3) array with the list of covalent bonds. Each row represents
        one bond and consists of three integers: first atom index (starting
        from zero), second atom index & an optional bond type (0: not known, 1:
        single, 2: double, 3: triple, 4: conjugated).
    cellvecs
        A (NP, 3) array containing the (real-space) cell vectors describing
        periodic boundary conditions. A single vector corresponds to a 1D cell,
        e.g. for a wire. Two vectors describe a 2D cell, e.g. for a membrane.
        Three vectors describe a 3D cell, e.g. a crystalline solid.
    charge
        The net charge of the system.
    core_energy
        The Hartree-Fock energy due to the core orbitals
    cube
        An instance of Cube, describing the volumetric data from a cube (or
        similar) file.
    energy
        The total energy (electronic + nn)
    extcharges
        Array with values of external charges, with shape (nextcharge, 4). First
        three columns for Cartesian X, Y and Z coordinates, last column for the
        actual charge.
    extra
        A dictionary with additional data loaded from a file. Any data which
        cannot be assigned to the other attributes belongs here. It may be
        decided in future to move some of the results from this dictionary to
        IOData attributes, with a more final name.
    g_rot
        The rotational symmetry number of the molecule.
    lot
        The level of theory used to compute the orbitals (and other properties).
    mo
        An instance of MolecularOrbitals.
    moments
        A dictionary with electrostatic multipole moments. Keys are (angmom,
        kind) tuples where angmom is an integer for the angular momentum and
        kind is 'c' for Cartesian or 'p' for pure functions (only for angmom >=
        2). The corresponding values are 1D numpy arrays. The order of the
        components of the multipole moments follows the HORTON2_CONVENTIONS from
        iodata/basis.py
    nelec
        The number of electrons.
    obasis
        An OrderedDict containing parameters to instantiate a GOBasis class.
    obasis_name
        A name or DOI describing the basis set used for the orbitals in the
        mo attribute (if applicable). Should be consistent with
        www.basissetexchange.org.
    one_ints
        Dictionary where keys are names and values are numpy arrays with
        one-body operators, typically integrals of a one-body operator
        with a pair of (Gaussian) basis functions. Names can start with ``olp``
        (overlap), ``kin`` (kinetic energy), ``na`` (nuclear attraction),
        ``core`` (core hamiltonian), etc. When relevant, these names must have a
        suffix ``_ao`` or ``_mo`` to clarify in which basis the integrals are
        computed. ``_ao`` is used to denote integrals in a non-orthogonal
        (atomic orbital) basis. ``_mo`` is used to denote an orthogonal
        (molecular orbital) basis. For the overlap integrals, this suffix can be
        omitted because it is only useful to compute them in the atomic-orbital
        basis.
    one_rdms
        Dictionary where keys are names and values are one-particle density
        matrices. Names can be ``scf``, ``post_scf``, ``scf_spin``,
        ``post_scf_spin``. These matrices are always expressed in the AO basis.
    run_type
        The type of calculation that lead to the results stored in IOData, e.g.
        'energy', 'energy_force', 'opt', 'freq', ...
    spinpol
        The spin polarization. By default, its value is derived from the
        molecular orbitals (mo attribute), as abs(nalpha - nbeta). In this case,
        spinpol cannot be set. When no molecular orbitals are present, this
        attribute can be set.
    title
         A suitable name for the data.
    two_ints
        Dictionary where keys are names and values are numpy arrays with
        two-body operators, typically integrals of two-body operator
        with four of (Gaussian) basis functions. Names can start with ``er``
        (electron repulsion) or ``two`` (general pairswise interaction). When
        relevant, these names must have a suffix ``_ao`` or ``_mo`` to clarify
        in which basis the integrals are computed, see one_ints for more
        details. Array indexes are in physicist's notation.
    two_rdms
        Dictionary where keys are names and values are two-particle density
        matrices. Names can be ``post_scf`` or ``post_scf_spin``. These matrices
        are always expressed in the AO basis. Array indexes are in physicist's
        notation.

    """

    atcharges: dict = {}
    atcoords: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape('natom', 3))
    _atcorenums: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape('natom'))
    atffparams: dict = {}
    atfrozen: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(bool),
        validator=validate_shape('natom'))
    atgradient: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape('natom', 3))
    athessian: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape(None, None))
    atmasses: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape('natom'))
    atnums: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(int),
        validator=validate_shape('natom'))
    basisdef: str = None
    bonds: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(int),
        validator=validate_shape(None, 3))
    cellvecs: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape(None, 3))
    _charge: float = None
    core_energy: float = None
    cube: Cube = None
    energy: float = None
    extcharges: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=validate_shape(None, 4))
    extra: dict = {}
    g_rot: float = None
    lot: str = None
    mo: MolecularOrbitals = None
    moments: dict = {}
    _nelec: float = None
    obasis: MolecularBasis = None
    obasis_name: str = None
    one_ints: dict = {}
    one_rdms: dict = {}
    run_type: str = None
    _spinpol: float = None
    title: str = None
    two_ints: dict = {}
    two_rdms: dict = {}

    # Public interfaces to private attributes

    @property
    def atcorenums(self) -> np.ndarray:
        """Return effective core charges."""
        result = self._atcorenums
        if result is None and self.atnums is not None:
            # Known bug in pylint. See
            # https://stackoverflow.com/questions/47972143/using-attr-with-pylint
            # https://github.com/PyCQA/pylint/issues/1694
            # pylint: disable=no-member
            result = self.atnums.astype(float)
            self._atcorenums = result
        return result

    @atcorenums.setter
    def atcorenums(self, atcorenums):
        self._atcorenums = atcorenums

    @property
    def charge(self) -> float:
        """Return the net charge of the system."""
        if self.atcorenums is None:
            return self._charge
        return self.atcorenums.sum() - self.nelec

    @charge.setter
    def charge(self, charge: float):
        if self.atcorenums is None:
            self._charge = charge
        else:
            self.nelec = self.atcorenums.sum() - charge

    @property
    def natom(self) -> int:
        """Return the number of atoms."""
        natom = None
        if self.atcoords is not None:
            natom = len(self.atcoords)
        elif self.atcorenums is not None:
            natom = len(self.atcorenums)
        elif self.atgradient is not None:
            natom = len(self.atgradient)
        elif self.atfrozen is not None:
            natom = len(self.atfrozen)
        elif self.atmasses is not None:
            natom = len(self.atmasses)
        elif self.atnums is not None:
            natom = len(self.atnums)
        return natom

    @property
    def nelec(self) -> float:
        """Return the number of electrons."""
        if self.mo is None:
            return self._nelec
        return self.mo.nelec

    @nelec.setter
    def nelec(self, nelec: float):
        if self.mo is None:
            self._nelec = nelec
        else:
            raise TypeError("nelec cannot be set when orbitals are present.")

    @property
    def spinpol(self) -> float:
        """Return the spin polarization.

        Warning: for restricted wavefunctions, it is assumed that an occupation
        number in ]0, 2[ implies spin polarizaiton, which may not always be a
        valid assumption.
        """
        if self.mo is None:
            return self._spinpol
        return self.mo.spinpol

    @spinpol.setter
    def spinpol(self, spinpol: float):
        if self.mo is None:
            self._spinpol = spinpol
        else:
            raise TypeError("spinpol cannot be set when orbitals are present.")
