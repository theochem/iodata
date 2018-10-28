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
IOData is a HORTON 3 module for input/output of quantum chemistry file formats. Documentation is
here: https://iodata.readthedocs.io/en/latest/index.html


License
-------

IOData is distributed under GPL License version 3 (GPLv3).


Dependencies
------------

The following dependencies will be necessary for IOData to build properly,

* Python >= 3.5: http://www.python.org/
* SciPy >= 0.11.0: http://www.scipy.org/
* NumPy >= 1.9.1: http://www.numpy.org/
* Nosetests >= 1.1.2: http://readthedocs.org/docs/nose/en/latest/
* Cython


Installation
------------

To install IOData using a conda environment:

* Install miniconda: https://conda.io/miniconda.html
* OR anaconda: https://www.anaconda.com/download

.. code-block:: bash

    $ conda create -n iodata_env
    $ source activate iodata_env
    (iodata_env) $ conda install -c theochem iodata

See https://iodata.readthedocs.io/en/latest/install.html for full details.

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
