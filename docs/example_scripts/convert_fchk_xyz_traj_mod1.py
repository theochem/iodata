#!/usr/bin/env python

from iodata import dump_many, load_many

# Read the trajectory
trj = list(load_many("peroxide_opt.fchk"))
# Manipulate if desired
for i, data in enumerate(trj):
    data.title = f"Frame {i}"
# Write the trajectory
dump_many(trj, "peroxide_opt.xyz")
