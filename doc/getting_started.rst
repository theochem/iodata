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

Getting Started
===============

IOData can be used to read and write different quantum chemistry file formats.


Script usage
------------

The simplest way to use IOData, without writing any code is to use the ``iodata-convert`` script.

.. code-block:: bash

    iodata-convert in.fchk out.molden

See the :code:`--help` option for more details on usage.


Code usage
----------

More complex use cases can be implemented in Python, using IOData as a library.
IOData stores an object containing the data read from the file.


Reading
^^^^^^^

To read a file, use something like this:

.. code-block:: python

    from iodata import load_one

    mol = load_one('water.xyz')  # XYZ files contain atomic coordinates in Angstrom
    print(mol.atcoords)  # print coordinates in Bohr.

**Note that IOData will automatically convert units from the file format's
official specification to atomic units (which is the format used throughout
HORTON3).**

The file format is inferred from the extension, but one can override the
detection mechanism by manually specifying the format:

.. code-block:: python

    from iodata import load_one

    mol = load_one('water.foo', 'xyz')  # XYZ file with unusual extension
    print(mol.atcoords)

IOData also has basic support for loading databases of molecules. For example,
the following will iterate over all frames in an XYZ file:

.. code-block:: python

    from iodata import load_many

    # print the title line from each frame in the trajectory.
    for mol in load_many('trajectory.xyz'):
        print(mol.title)



Writing
^^^^^^^

IOData can also be used to write different file formats:

.. code-block:: python

    from iodata import load_one, dump_one

    mol = load_one('water.fchk')
    # Here you may put some code to manipulate mol before writing it the data
    # to a different file.
    dump_one(mol, 'water.molden')


One could also convert (and manipulate) an entire trajectory. The following
example converts a geometry optimization trajectory from a Gaussian FCHK file
to an XYZ file:

.. code-block:: python

    from iodata import load_many, dump_many

    # Conversion without manipulation.
    dump_many((mol for mol in load_many('water_opt.fchk')), 'water_opt.xyz')

If you wish to perform some manipulations before writing the trajectory, the
simplest way is to load the entire trajectory in a list of IOData objects and
dump it later:

.. code-block:: python

    from iodata import load_many, dump_many

    # Read the trajectory
    trj = list(load_many('water_opt.fchk'))
    # Manipulate if desired
    # ...
    # Write the trajectory
    dump_many(trj, 'water_opt.xyz')


For very large trajectories, you may want to avoid loading it as a whole in
memory. For this, one should avoid making the ``list`` object in the above
example. The following approach would be more memory efficient.

.. code-block:: python

    from iodata import load_many, dump_many

    def itermols():
        for mol in load_many("traj1.xyz"):
            # Do some manipulations
            yield modified_mol

    dump_many(itermols(), "traj2.xyz")


Input files
^^^^^^^^^^^

IOData can be used to write input files for quantum-chemistry software. By
default minimal settings are used, which can be changed if needed. For example,
the following will prepare a Gaussian input for a HF/STO-3G calculation from
a PDB file:

.. code-block:: python

    from iodata import load_one, write_input

    write_input(load_one("water.pdb"), "water.com", fmt="gaussian")

The level of theory and other settings can be modified by setting corresponding
attributes in the IOData object:

.. code-block:: python

    from iodata import load_one, write_input

    mol = load_one("water.pdb")
    mol.lot = "B3LYP"
    mol.obasis_name = "6-31g*"
    mol.run_type = "opt"
    write_input(mol, "water.com", fmt="gaussian")

The run types can be any of the following: ``energy``, ``energy_force``,
``opt``, ``scan`` or ``freq``. These are translated into program-specific
keywords when the file is written.

It is possible to define a custom input file template to allow for specialized
commands. This is done by passing a template string using the optional ``template`` keyword,
placing each IOData attribute (or additional keyword, as shown below) in curly brackets:

.. code-block:: python

    from iodata import load_one, write_input

    mol = load_one("water.pdb")
    mol.lot = "B3LYP"
    mol.obasis_name = "Def2QZVP"
    mol.run_type = "opt"
    custom_template = """\
    %NProcShared=4
    %mem=16GB
    %chk=B3LYP_def2qzvp_H2O
    #n {lot}/{obasis_name} scf=(maxcycle=900,verytightlineq,xqc) integral=(grid=ultrafinegrid) pop=(cm5, hlygat, mbs, npa, esp)

    {title}

    {charge} {spinmult}
    {geometry}

    """
    write_input(mol, "water.com", fmt="gaussian", template=custom_template)

The input file template may also include keywords that are not part of the IOData
object:

.. code-block:: python

    from iodata import load_one, write_input

    mol = load_one("water.pdb")
    mol.lot = "B3LYP"
    mol.obasis_name = "Def2QZVP"
    mol.run_type = "opt"
    custom_template = """\
    %chk={chk_name}
    #n {lot}/{obasis_name} {run_type}

    {title}

    {charge} {spinmult}
    {geometry}

    """
    # Custom keywords as arguments (best for few extra arguments)
    write_input(mol, "water.com", fmt="gaussian", template=custom_template, chk_name="B3LYP_def2qzvp_water")

    # Custom keywords from a dict (in cases with many extra arguments)
    custom_keywords = {"chk_name": "B3LYP_def2qzvp_waters"}
    write_input(mol, "water.com", fmt="gaussian", template=custom_template, **custom_keywords)

In some cases, it may be preferable to load the template from file, instead of
defining it in the script:

.. code-block:: python

    from iodata import load_one, write_input

    mol = load_one("water.pdb")
    mol.lot = "B3LYP"
    mol.obasis_name = "6-31g*"
    mol.run_type = "opt"
    write_input(mol, "water.com", fmt="gaussian", template=open("my_template.com", "r").read())


Data storage
^^^^^^^^^^^^

IOData can be used to store data in a consistent format for writing at a future point.

.. code-block:: python

    import numpy as np
    from iodata import IOData

    mol = IOData(title="water")
    mol.atnums = np.array([8, 1, 1])
    mol.atcoords = np.array([[0, 0, 0,], [0, 1, 0,], [0, -1, 0,]])  # in Bohr


.. _units:


Unit conversion
^^^^^^^^^^^^^^^

IOData always represents all quantities in atomic units and unit conversion
constants are defined in ``iodata.utils``. Conversion *to* atomic units is done
by *multiplication* with a unit constant. This convention can be easily
remembered with the following examples:

- When you say "this bond length is 1.5 Å", the IOData equivalent is
  ``bond_length = 1.5 * angstrom``.

- The conversion from atomic units is similar to axes labels in old papers.
  For example. a bond length in angstrom is printed as "Bond length / Å".
  Expressing this with IOData's conventions gives
  ``print("Bond length in Angstrom:", bond_length /  angstrom)``

(This is rather different from the ASE conventions.)
