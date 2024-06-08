#!/usr/bin/env python

from iodata import dump_many, load_many


def iter_data():
    """Read and modify the trajectory."""
    for i, data in enumerate(load_many("peroxide_opt.fchk")):
        data.title = f"Frame {i}"
        yield data


# Write the trajectory
dump_many(iter_data(), "peroxide_opt.xyz")
