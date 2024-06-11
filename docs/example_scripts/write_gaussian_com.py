#!/usr/bin/env python

from iodata import load_one, write_input

write_input(load_one("water.pdb"), "water.com", fmt="gaussian")
