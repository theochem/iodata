from horton import *
from glob import glob

log.set_level(log.silent)

g = glob("/home/mattchan/Backup/pycharm/horton/data/test/*.wfn")
for i in g:
    a = IOData.from_file(i)
    print i

    title, numbers, coordinates, centers, type_assignment, exponents, \
    mo_count, mo_occ, mo_energy, coefficients, energy = load_wfn_low(i)

    print coefficients.shape[0] == a.obasis.nbasis