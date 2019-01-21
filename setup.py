#!/usr/bin/env python


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
    description='',
    long_description=get_readme(),
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='https://github.com/theochem/iodata',
    package_dir={'iodata': 'iodata'},
    packages=['iodata', 'iodata.test'],
    cmdclass={'build_ext': Cython.Build.build_ext},
    ext_modules=[Extension("iodata.overlap_accel",
                           sources=['iodata/overlap_accel.pyx'],
                           include_dirs=[np.get_include()])],
    scripts=['bin/horton-convert'],
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
    setup_requires=['numpy>=1.0', 'cython>=0.24.1', 'scipy'],
    install_requires=['numpy>=1.0', 'cython>=0.24.1', 'scipy', 'nose>=0.11'],
)
