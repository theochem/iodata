#!/usr/bin/env python

from iodata import load_one, write_input

mol = load_one("water.pdb")
mol.lot = "B3LYP"
mol.obasis_name = "Def2QZVP"
mol.run_type = "opt"
custom_template = """\
%chk={chk_name}
#n {lot}/{obasis_name} {run_type}

{title}

{charge} {spinmult}
{geometry}

"""
# Custom keywords as arguments (best for few extra arguments)
write_input(
    mol, "water.com", fmt="gaussian", template=custom_template, chk_name="B3LYP_def2qzvp_water"
)

# Custom keywords from a dict (in cases with many extra arguments)
custom_keywords = {"chk_name": "B3LYP_def2qzvp_waters"}
write_input(mol, "water.com", fmt="gaussian", template=custom_template, **custom_keywords)
