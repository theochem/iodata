..
    : IODATA is an input and output module for quantum chemistry.
    :
    : Copyright (C) 2011-2019 The IODATA Development Team
    :
    : This file is part of IODATA.
    :
    : IODATA is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : IODATA is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --

.. _file_formats:

Supported File Formats
######################

Molecular geometry file formats
===============================

.. |br| raw:: html

    <br />

The ``.xyz`` format
-------------------

======================== =======================================================
Load                     Yes
Dump                     Yes
Recognized by            File extension ``.xyz``
Interoperation           Nearly all molecular simulation codes and `Open Babel <http://openbabel.org/>`_
Always **loading**       ``title`` ``numbers`` ``coordinates``
Derived when **loading** ``natom`` ``pseudo_numbers``
Required for **dumping** ``numbers`` ``coordinates``
Optional for **dumping** ``title``
======================== =======================================================


The ``POSCAR`` format
---------------------

======================== =======================================================
Load                     Yes
Dump                     Yes
Recognized by            File prefix ``POSCAR``
Interoperation           `VASP 5.X <https://www.vasp.at/>`_, `VESTA <http://jp-minerals.org/vesta/en/>`_
Always **loading**       ``title`` ``numbers`` ``coordinates`` ``cell``
Derived when **loading** ``natom`` ``pseudo_numbers``
Required for **dumping** ``numbers`` ``coordinates`` ``cell``
Optional for **dumping** ``title``
======================== =======================================================


.. _ref_file_formats_cube:

Cube file formats
=================

The Gaussian ``.cube`` format
-----------------------------

======================== =======================================================
Load                     Yes
Dump                     Yes
Recognized by            File extension ``.cube``
Interoperation           `Gaussian <http://www.gaussian.com/>`_, `CP2K <http://www.cp2k.org/>`_, `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_, `Q-Chem <http://www.q-chem.com/>`_`, ...
Always **loading**       ``title`` ``numbers`` ``pseudo_numbers`` ``coordinates`` ``cell`` ``grid`` ``cube_data``
Derived when **loading** ``natom``
Required for **dumping** ``numbers``  ``coordinates`` ``cell`` ``grid`` ``cube_data``
Optional for **dumping** ``title`` ``pseudo_numbers``
======================== =======================================================

.. note::

    The second column in the geometry specification of the cube file is used
    for the pseudo-numbers.

The VASP ``CHGCAR`` and ``LOCPOT`` formats
------------------------------------------

======================== =======================================================
Load                     Yes
Dump                     No
Recognized by            File prefix ``CHGCAR`` or ``LOCPOT``
Interoperation           `VASP 5.X <https://www.vasp.at/>`_, `VESTA <http://jp-minerals.org/vesta/en/>`_
Always **loading**       ``title`` ``coordinates`` ``numbers`` ``cell`` ``grid`` ``cube_data``
Derived when **loading** ``natom`` ``pseudo_numbers``
======================== =======================================================

.. note::

    Even though the ``CHGCAR`` and ``LOCPOT`` files look very similar, they
    require different conversions to atomic units.


.. _ref_file_formats_wfn:

Wavefunction formats (using a Gaussian basis set)
=================================================

All wavefunction formats share the following behavior

* In case of a restricted wavefunction, only the alpha orbitals are loaded.
* In case of an unrestricted wavefunction, both the alpha and beta orbitals are
  loaded.
* Some formats also `load` a ``permutation`` and/or a ``signs`` attribute. These are
  generated when loading the file, such that appropriate permutations and sign changes can be
  applied to convert to the proper HORTON conventions for Gaussian basis
  functions. These conventions are `fixed` in the ``from_file`` method. This
  allows you to fix also the order of elements in arrays loaded from another
  file. For example, you can load an ``.fchk`` and a ``.log`` file at the same
  time:

  .. code-block:: python

        mol = IOData.from_file('foo.fchk', 'foo.log')

  In this case, ``permutation`` is deduced from the file ``foo.fchk`` but is
  also applied to reorder the matrix elements loaded from ``foo.log``, for the
  sake of consistency.


The Gaussian ``.fchk`` format
-----------------------------

======================== =======================================================
Load                     Yes
Dump                     No
Recognized by            File extension ``.fchk``
Interoperation           `Gaussian <http://www.gaussian.com/>`_
Always **loading**       ``title`` ``coordinates`` ``numbers`` ``obasis`` ``orb_alpha`` ``permutation`` |br|
                         ``energy`` ``pseudo_numbers`` ``mulliken_charges``
**loading** if present   ``npa_charges`` ``esp_charges`` ``orb_beta`` ``dm_full_mp2`` ``dm_spin_mp2`` |br|
                         ``dm_full_mp3`` ``dm_spin_mp3`` ``dm_full_cc`` ``dm_spin_cc`` ``dm_full_ci`` |br|
                         ``dm_spin_ci`` ``dm_full_scf`` ``dm_spin_scf`` ``polar`` ``dipole_moment`` |br|
                         ``quadrupole_moment``
Derived when **loading** ``natom``
======================== =======================================================


The ``.molden`` format
----------------------

======================== =======================================================
Load                     Yes
Dump                     Yes
Recognized by            File extension ``.molden``
Interoperation           `Molpro <https://www.molpro.net/>`_,
                         `Orca <https://orcaforum.cec.mpg.de/>`_,
                         `PSI4 <http://www.psicode.org/>`_,
                         `Molden <http://www.cmbi.ru.nl/molden/>`_,
                         `Turbomole <http://www.turbomole.com/>`_
Always **loading**       ``coordinates`` ``numbers`` ``obasis`` ``orb_alpha``
                         ``pseudo_numbers`` ``signs``
**loading** if present   ``title`` ``orb_beta``
Derived when **loading** ``natom``
Required for **dumping** ``coordinates`` ``numbers`` ``obasis`` ``orb_alpha``
Optional for **dumping** ``title`` ``orb_beta`` ``pseudo_numbers``
======================== =======================================================


The ``.mkl`` (Molekel) format
-----------------------------

======================== =======================================================
Load                     Yes
Dump                     No
Recognized by            File extension ``.mkl``
Interoperation           `Molekel <http://ugovaretto.github.io/molekel/wiki/pmwiki.php/Main/HomePage.html>`_,
                         `Orca <https://orcaforum.cec.mpg.de/>`_,
Always **loading**       ``coordinates`` ``numbers`` ``obasis`` ``orb_alpha``
**loading** if present   ``orb_beta`` ``signs``
Derived when **loading** ``natom``
======================== =======================================================


The ``.wfn`` format
-------------------

======================== =======================================================
Load                     Yes
Dump                     No
Recognized by            File extension ``.wfn``
Interoperation           `GAMESS <http://www.msg.ameslab.gov/gamess/>`_,
                         `Gaussian <http://www.gaussian.com/>`_,
Always **loading**       ``title`` ``coordinates`` ``numbers`` ``obasis`` ``orb_alpha`` ``energy``
**loading** if present   ``orb_beta``
Derived when **loading** ``natom``
======================== =======================================================

.. note ::

    Only use this format if the program that generated it does not offer any
    alternatives that HORTON can load. The WFN format has the disadvantage that
    it cannot represent contractions and therefore expands all orbitals into
    a decontracted basis. This makes the post-processing less efficient compared
    to formats that do support contractions of Gaussian functions.


.. _ref_file_formats_ham:

Hamiltonian file formats
========================


The Molpro 2012 ``FCIDUMP`` format
----------------------------------

======================== =======================================================
Load                     Yes
Dump                     Yes
Recognized by            File name contains ``FCIDUMP``
Interoperation           `Molpro <https://www.molpro.net/>`_,
                         `PSI4 <http://www.psicode.org/>`_
Always **loading**       ``nelec`` ``ms2`` ``one_mo`` ``two_mo`` ``core_energy``
Required for **dumping** ``one_mo`` ``two_mo``
Optional for **dumping** ``core_energy`` ``nelec`` ``ms``
======================== =======================================================


The Gaussian ``.log`` file
--------------------------

======================== =======================================================
Load                     Yes
Dump                     No
Recognized by            File extension ``.log``
Interoperation           `Gaussian <http://www.gaussian.com/>`_,
**loading** if present   ``olp`` ``kin`` ``na`` ``er``
======================== =======================================================

In order to let Gaussian print out all the matrix elements (Gaussian integrals),
the following commands must be used in the Gaussian input file:

.. code-block:: text

    scf(conventional) iop(3/33=5) extralinks=l316 iop(3/27=999)

Just keep in mind that this feature in Gaussian only works for a low number of
basis functions. The ``FCIDUMP`` files generated with Molpro or PSI4 are more
reliable and also have the advantage that all integrals are stored in double
precision.


.. _ref_file_formats_internal:

HORTON's internal file format
=============================

The user is free to store any additional data
not covered by the file formats above. Many (not all) Python data types can
dumped into the internal format:

* ``int``

* ``float``

* ``str``

* Any NumPy array

* A dictionary with strings as keys and any mixture of the above data types as
  values.
