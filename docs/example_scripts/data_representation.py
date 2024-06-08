#!/usr/bin/env python

import numpy as np

from iodata import IOData

mol = IOData(title="water")
mol.atnums = np.array([8, 1, 1])
mol.atcoords = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])  # in Bohr
