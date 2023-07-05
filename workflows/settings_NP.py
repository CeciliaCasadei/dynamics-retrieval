# -*- coding: utf-8 -*-
import numpy

# results_path = '../data_Giannakis_PNAS_2012/results_2_full_Ts'
results_path = "../data_Giannakis_PNAS_2012/results_3"

# results_path = '../data_bR'
# sparsity = 0.5

q = 24  # PNAS - North Pacific
# q = 240 # J. of Climate - Indo Pacific
# q = 16384 # bR
# q = 1000 # bR test

b = 1000  # PNAS
# b = 3500 # SADM
# b = 1000 # bR
# b = 100 # bR test
sigma_sq = 4.0
# epsilon = 4.0 # PNAS Use the square of the reported epsilon value

l = 23  # PNAS
# l = 27 # SADM
toproject = list(range(l))

datatype = numpy.float64  # PNAS
# datatype = numpy.float32

# paral_step = 1000 # bR
# paral_step = 5 # PNAS - North pacific
paral_step_reconstruction = 1000

ncopies = 10  # q
