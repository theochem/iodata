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


Installation
============

Latest PyPI version
-------------------

Python 3 (>=3.9) must be installed before you can install IOData.
In addition, IOData has the following dependencies:

- numpy >= 1.22: https://numpy.org/
- scipy >= 1.8: https://scipy.org/
- attrs >= 21.3.0: https://www.attrs.org/en/stable/index.html

Normally, you don't need to install these dependencies manually. They will be
installed automatically when you follow the instructions below.

1. The cleanest option is to install IOData in a `virtual environment`_:

   .. code-block:: bash

       # Create a virtual environment in ~/horton3
       # Feel free to change the path.
       python3 -m venv ~/horton3

       # Activate the virtual environment.
       source ~/horton3/bin/activate

       # Install IOData in the venv horton3.
       pip3 install qc-iodata
       # alternative: python3 -m pip install qc-iodata

       # For developers, install with developer dependencies
       pip3 install qc-iodata[dev]
       # alternative: python3 -m pip install qc-iodata[dev]

   The ``source`` command needs to be entered every time you open a new virtual terminal,
   which can be inconvenient.
   You may put this line in your `.bashrc` or `.bash_profile` to avoid that repetition.
   You may also automatically load environments when entering specific directories
   with tools like `direnv`_.

2. You can install IOData into your ``${HOME}`` directory, without creating a virtual
   environment.

   .. code-block:: bash

       # Install the stable release in your home directory.
       pip3 install qc-iodata
       # alternative: python3 -m pip install qc-iodata

   This is by far the simplest method, ideal to get started, but you have only
   one home directory. If the installation breaks due to some experimentation,
   it is harder to make a clean start in comparison to the virtual environment,
   which you can simply delete.

In case the ``pip3`` executable is not found, pip may be installed in a
directory which is not included in your ``${PATH}`` variable. This seems to be a
common issue on macOS. A simple workaround is to replace ``pip3`` by ``python3
-m pip``.

In case Python and your operating system are up to date, you may also use
``pip`` instead of ``pip3`` or ``python`` instead of ``python3``. The ``3`` is
only used to avoid potential confusion with Python 2. Note that the ``3`` is
only present in names of executables, not names of Python modules.


.. _install-latest-git-revision:

Latest Git revision
-------------------

This section shows how one can install the latest revision of IOData from the
git repository. This kind of installation comes with some risks (sudden API
changes, bugs, ...) and so be prepared to accept them when using the following
installation instructions.

There are two installation methods:

1. **Quick and dirty.** Of this method, there are two variants, depending on
   the correctness of your ``PATH`` variable.
   These different scenarios are explained in more detail in the previous section.

   .. code-block:: bash

       # correct PATH
       pip install git+https://github.com/theochem/iodata.git
       # broken PATH
       python3 -m pip install git+https://github.com/theochem/iodata.git

2. **Slow and smart.** In addition to the two variations in the quick and dirty
   method, the slow and smart can be useful when you plan to tinker with the source code.
   You also have the options to use *SSH* or *HTTPS* protocols to
   clone the git repository. Pick whichever works best for you.

   .. code-block:: bash

        # A) Clone git repo with https OR ssh:
        # The second one only works if you have ssh set up for Github
        #  A1) https
        git clone https://github.com/theochem/iodata.git
        #  A2) ssh
        git clone git@github.com:theochem/iodata.git
        # B) Actual install, 2 different methods.
        #  B1) correct PATH
        pip3 install .
        #  B2) broken PATH
        python3 -m pip install .


Testing
-------

The tests are automatically run after each change in the main branch on GitHub,
but you may try them again on your own machine after installation.
For this to work, you also need to install the development dependencies, as shown below.

.. code-block:: bash

    # Install pytest
    pip3 install qc-iodata[dev]

    # Finally, run the tests.
    pytest --pyargs iodata -n auto


.. _virtual environment: https://docs.python.org/3/tutorial/venv.html
.. _direnv: https://direnv.net/
