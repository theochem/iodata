IOData
======
|Travis|
|Conda|
|Pypi|
|Codecov|
|Version|
|CondaVersion|


About
-----


License
-------

IOData is distributed under GPL License version 3 (GPLv3).


Dependencies
------------

The following dependencies will be necessary for IOData to build properly,

* Python >= 2.7, >= 3.5: http://www.python.org/
* SciPy >= 0.11.0: http://www.scipy.org/
* NumPy >= 1.9.1: http://www.numpy.org/
* Nosetests >= 1.1.2: http://readthedocs.org/docs/nose/en/latest/


Installation
------------

To install IOData using a conda environment:

* Install miniconda: https://conda.io/miniconda.html
* OR anaconda: https://www.anaconda.com/download

.. code-block:: bash

    $ conda create -n iodata_env
    $ source activate iodata_env
    (iodata_env) $ conda install -c theochem iodata

Alternatively, to install IOData:

.. code-block:: bash

    $ python ./setup install --user


Testing
-------

Requires nosetests to be installed.

To run tests using the iodata_env conda environment:

.. code-block:: bash

    (iodata_env) $ conda install nose
    (iodata_env) $ nosetests -v iodata


Alternatively, to run tests:

.. code-block:: bash

    $ nosetests -v iodata

.. |Travis| image:: https://travis-ci.org/theochem/iodata.svg?branch=master
    :target: https://travis-ci.org/theochem/iodata
.. |Version| image:: https://img.shields.io/pypi/pyversions/iodata.svg
.. |Pypi| image:: https://img.shields.io/pypi/v/iodata.svg
    :target: https://pypi.python.org/pypi/iodata/0.1.3
.. |Codecov| image:: https://img.shields.io/codecov/c/github/theochem/iodata/master.svg
    :target: https://codecov.io/gh/theochem/iodata
.. |Conda| image:: https://img.shields.io/conda/v/theochem/iodata.svg
    :target: https://anaconda.org/theochem/iodata
.. |CondaVersion| image:: https://img.shields.io/conda/pn/theochem/iodata.svg
    :target: https://anaconda.org/theochem/iodata
