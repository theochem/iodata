Install
=======

Dependencies
------------

IOData (like all HORTON3 packages) is built using conda:

It is provided in the conda **theochem** channel.


Installation (from Conda)
-------------------------

To install IOData:

.. code-block:: bash

    $ conda -c theochem install iodata

.. _install_from_source:

Installation (from source)
--------------------------

If you wish to build from source, you will need the **conda-build** package
to build it.

You must set the PROJECT_VERSION and MYCONDAPY environmental variables to
emulate the travis build environment.

From project root, issue some variation of:

.. code-block:: bash

    $ PROJECT_VERSION=0.0.0 MYCONDAPY=3.7 conda-build -c theochem tools/conda.recipe

Installation (by-hand)
----------------------

Advanced developers may build by hand using the dependencies listed below,
but the procedure is entirely unsupported.

The following dependencies will be necessary for IOData to build properly,

* Python >= 3.7
* NumPy >= 1.9.1
* Scipy
* Nosetests
* gcc/clang
* Cython


Testing
-------

The tests are automatically run when building with conda, but you may try
them again on your own machine:

.. code-block:: bash

    $ nosetests -v iodata