#!/usr/bin/env python
# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --


import os

import Cython.Build
import numpy as np
from setuptools import setup, Extension


def get_version():
    """Get the version string set by Travis, else default to version 0.0.0."""
    return os.environ.get("PROJECT_VERSION", "0.0.0")


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as fhandle:
        return fhandle.read()


setup(
    name='iodata',
    version=get_version(),
    description='Python Input and Output Library for Quantum Chemistry.',
    long_description=get_readme(),
    author='HORTON-ChemTools Dev Team',
    author_email='horton.chemtools@gmail.com',
    url='https://github.com/theochem/iodata',
    package_dir={'iodata': 'iodata'},
    packages=['iodata', 'iodata.test', 'iodata.test.data'],
    cmdclass={'build_ext': Cython.Build.build_ext},
    ext_modules=[Extension("iodata.overlap_accel",
                           sources=['iodata/overlap_accel.pyx'],
                           include_dirs=[np.get_include()])],
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
    zip_safe=False,
    setup_requires=['numpy>=1.0', 'cython>=0.24.1'],
    install_requires=['numpy>=1.0', 'cython>=0.24.1', 'scipy', 'pytest>=4.2.0',
                      'importlib_resources; python_version < "3.7"'],
)
