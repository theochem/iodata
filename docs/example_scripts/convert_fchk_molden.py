#!/usr/bin/env python

from iodata import dump_one, load_one

mol = load_one("water.fchk")
# Here you may put some code to manipulate mol before writing it the data
# to a different file.
dump_one(mol, "water.molden", allow_changes=True)
