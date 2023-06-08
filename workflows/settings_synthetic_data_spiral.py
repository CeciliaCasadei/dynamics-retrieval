# -*- coding: utf-8 -*-
import math

import numpy

results_path = "../../synthetic_data_spiral/test1/ssa_2"
m = 2
S = 400
q = 40

# D_sq
paral_step = 20
n_workers = int(math.ceil(float(q) / paral_step))
print(n_workers)

data_file = "%s/x.jbl" % results_path
datatype = numpy.float64

b = 50

log10eps = 1.5
sigma_sq = 2 * 10 ** log10eps

l = 31
nmodes = l  # 7
toproject = list(range(nmodes))
paral_step_A = 20
n_workers_A = int(math.ceil(float(q) / paral_step_A))
print(n_workers_A)

ncopies = q
modes_to_reconstruct = list(range(3))

paral_step_reconstruction = 100
n_workers_reconstruction = int(
    math.ceil(float(S - q - ncopies + 1) / paral_step_reconstruction)
)
print(n_workers_reconstruction)
