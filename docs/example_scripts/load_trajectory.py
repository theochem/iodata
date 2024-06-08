#!/usr/bin/env python

from iodata import load_many

# print the title line from each frame in the trajectory.
for mol in load_many("trajectory.xyz"):
    print(mol.title)
    print(mol.atcoords)
