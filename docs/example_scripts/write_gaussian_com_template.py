#!/usr/bin/env python

from iodata import load_one, write_input

mol = load_one("water.pdb")
mol.lot = "B3LYP"
mol.obasis_name = "Def2QZVP"
mol.run_type = "opt"
custom_template = """\
%NProcShared=4
%mem=16GB
%chk=B3LYP_def2qzvp_H2O
#n {lot}/{obasis_name} scf=(maxcycle=900,verytightlineq,xqc)
integral=(grid=ultrafinegrid) pop=(cm5, hlygat, mbs, npa, esp)

{title}

{charge} {spinmult}
{geometry}

"""
write_input(mol, "water.com", fmt="gaussian", template=custom_template)
