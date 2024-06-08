#!/usr/bin/env python

from iodata import load_one

mol = load_one("water.xyz")  # XYZ files contain atomic coordinates in Angstrom
print(mol.atcoords)  # print coordinates in Bohr.
