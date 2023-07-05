# -*- coding: utf-8 -*-
import math

import numpy

m = 7000
S = 30000
q = 4000

results_path = "../../synthetic_data_5/test3/binning"

# # D_sq
# paral_step = 400
# n_workers = int(math.ceil(float(q)/paral_step))

# datatype = numpy.float64

# b = 1500


# # # # # # # # # # n_workers_aj = (2*f_max) + 1

# # # # # # # # # # paral_step_lp_filter_Dsq = 1000
# # # # # # # # # # n_workers_lp_filter_Dsq = int(math.ceil((S-q)/paral_step_lp_filter_Dsq)) #26

# log10eps = 1.0
# sigma_sq = 2*10**log10eps
# results_path = '../../synthetic_data_4/test5/nlsa/q_%d'%q#'distance_calculation_onlymeasured_normalised/b_%d_eu_nns/log10eps_1p0'%(q, b)#'b_%d_time_nns/log10eps_6p0'%(q, b)

# data_file = '%s/x.jbl'%results_path

# l = 50

# nmodes = l
# toproject = range(nmodes)
# paral_step_A = 400
# n_workers_A = int(math.ceil(float(q)/paral_step_A))

# ncopies = q
# modes_to_reconstruct = range(20)

# paral_step_reconstruction = 2000
# n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))

# # f_max = 100
# # f_max_considered = 100
