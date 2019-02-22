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

.. _dev_building:

Publishing Contributions
========================

IOData, like all HORTON 3 modules, has a robust build system included which
builds off the Conda architecture and Travis-CI. This ensures our codebase
meets certain quality standards.

When making a new feature for IOData, you should **fork the main repository
and then make a pull request from your fork to the master branch**. The pull
request will then be verified by Travis-CI. Travis checks for several things:

* PEP-8 (code style) compliance
* PEP-257 (docstring) compliance
* Unit tests pass
* Unit testing coverage
* Sufficient documentation
* Linux/OSX compatibility

You can examine the Travis-CI error log if your code fails and make the
appropriate changes. Every commit to your pull request will be tested
automatically.

Tips
----

You should check your code before making a pull request to save yourself
some trouble. Using a modern IDE like Pycharm will go a long way to
making sure your code complies with PEP-8, PEP-257, and so forth. Use the
code refactoring features.

You will also find that testing locally can save you some time.
Run pytest on your own machine before submitting your PR.

Building with Conda on your own machine will emulate lots of the build
tests and give you a virtualenv that will be more reliable.

The instructions for compiling with Conda are listed in
:ref:`install_from_source`. This is the build which will be executed
by Travis-CI when making a pull request.

Documentation changes
---------------------

Most of the documentation builds happen automatically. The ``.rst`` files within ``doc/``
are processed by Sphinx and associated scripts. Docstring changes in the code are automatically
included in the new docs. If write user documentation for your feature (which would be
greatly appreciated), you can add a new ``.rst`` or modify an existing one. The docs will then
be built by ReadTheDocs upon merging to master.

You can test your docs beforehand by running this command in the ``docs`` directory. Note that
this will not work unless you have all the doc building dependencies installed.

.. code-block:: bash

    ./gen_docs.sh
    make html

Since Sphinx cannot parse docstring unless it imports the module, this means:

1. Your code must be syntactically correct and importable
2. If your code contains Cython modules, they must be built before building the docs.

More concretely, if you ran ``cleanfiles.sh`` and ``gen_docs.sh`` immediately afterwards, you
will get errors. Also, if you build using Conda-build, your docs will not build unless you install
the package into your environment first.

Introducing new dependencies
----------------------------

Sometimes a contribution will introduce a dependency on another library. This will need to be added
to the conda virtualenv when building the package for testing. This can be done in
``tools/conda.recipe/meta.yaml``. Depending on whether your dependency is a Python library, or a
compiled C/C++ library, it needs to go in different sections.

The ``host`` section is for packages which will be installed into the build environment. This is for
C/C++ dependencies.

.. code-block:: yaml

    host:
    - python ={{ MYCONDAPY }}
    - libint
    - cython >=0.24.1
    - numpy
    - setuptools
    - pytest

The ``run`` section is for installing dependencies on the user's machine. This is for Python
dependencies. This is also for libraries which need to be dynamically linked. In theory the Conda
build tools will automatically add the linked libraries from ``host`` here as well, but in practice
the process is not reliable. You are advised to add them in as well.

.. code-block:: yaml

    run:
    - python >=3
    - numpy
    - scipy
    - pytest
    - libint

For details on the ``meta.yaml`` file, read the
`conda-build documentation
<https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html>`_.
Some commands within the documentation are incorrect/out-of-date. You have been forewarned...
