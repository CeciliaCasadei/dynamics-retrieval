# -*- coding: utf-8 -*-
import math

import numpy

results_path = "out/synthetic_data_spiral_3D/test2"
m = 3
S = 800  # test1 400, from test2 800
q = 100  # test1 40,  from test2 100

data_file = "%s/x.jbl" % results_path
datatype = numpy.float64

# D_sq
paral_step = 20
n_workers = int(math.ceil(float(q) / paral_step))

b = 50
log10eps = 1.5
sigma_sq = 2 * 10 ** log10eps

l = 31
nmodes = l
toproject = list(range(nmodes))

paral_step_A = 20
n_workers_A = int(math.ceil(float(q) / paral_step_A))
print(n_workers_A)

ncopies = q
modes_to_reconstruct = list(range(10))

paral_step_reconstruction = 100
n_workers_reconstruction = int(
    math.ceil(float(S - q - ncopies + 1) / paral_step_reconstruction)
)
print(n_workers_reconstruction)
