#!/usr/bin/env python

from iodata import dump_many, load_many

# Load all optimization steps and write as XYZ.
dump_many(load_many("peroxide_opt.fchk"), "peroxide_opt.xyz")
