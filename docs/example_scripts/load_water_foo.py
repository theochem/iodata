#!/usr/bin/env python

from iodata import load_one

mol = load_one("water.foo", fmt="xyz")  # XYZ file with unusual extension
print(mol.atcoords)
