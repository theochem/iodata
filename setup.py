#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from setuptools import setup, Extension
import Cython.Build


def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable defining the
    version string with single quotes.

    """
    with open('iodata/version.py', 'r') as f:
        return f.read().split('=')[-1].replace('\'', '').strip()


setup(
    name='iodata',
    version=get_version(),
    description='',
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='https://github.com/theochem/iodata',
    package_dir={'iodata': 'iodata'},
    packages=['iodata', 'iodata.test'],
    cmdclass={'build_ext': Cython.Build.build_ext},
    ext_modules=[Extension(
        "iodata.overlap_accel",
        sources=['iodata/overlap_accel.pyx'],
        depends=['iodata/overlap_accel.pxd'],
        include_dirs=[np.get_include()],
    )],
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
    zip_safe=False,
    setup_requires=['numpy>=1.0', 'cython>=0.24.1'],
    install_requires=['numpy>=1.0', 'nose>=0.11', 'cython>=0.24.1'],    
)
