# -*- coding: utf-8 -*-
import numpy
import joblib

label = 'light'

root_folder = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2'
data_path = '%s/converted_data_%s'%(root_folder, label)
results_path = '%s/results_NLSA/bR_%s'%(root_folder, label)
data_file = '%s/T_sparse_%s.jbl'%(data_path, label)

datatype = numpy.float64

q = 16384         # Concatenation n.
paral_step = 1000 # with 17 proc
n_workers = 17

n = 116732        # S-q+1 

# b = 5000

# ps = joblib.load('%s/epsilon_determination_b_nns/Dsq_ps_1_100_percent_b_nns.jbl'%results_path)
# Dsq_50th_p = ps[50-1]
# sigma_sq = 2*Dsq_50th_p

# # Dsq_p25 = joblib.load('%s/Dsq_p25.jbl'%results_path)
# # sigma_sq = Dsq_p25/3
# # # n = 4
# # # sigma_opt = 6574.4 
# # # sigma_sq = (n*sigma_opt)**2

# l = 20     # N. diffusion map eigenvectors
# eigenlabel = '_ARPACK'

# # nmodes = 6 # N. SVD modes   max nmodes = l
# # toproject = [0, 1, 2, 7, 8, 9]
# # paral_step_A = 1400 # with 12 proc

# # ncopies = 1000
# # paral_step_reconstruction = 10000 