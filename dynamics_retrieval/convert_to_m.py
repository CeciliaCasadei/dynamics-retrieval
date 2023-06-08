# -*- coding: utf-8 -*-
import scipy.io
import joblib
from scipy import sparse

# T = joblib.load('/das/work/p17/p17491/Cecilia_Casadei/NLSA/synthetic_data_jitter/test6/x.jbl')
# print T.dtype, T.shape, sparse.issparse(T)
# Ttr = T.transpose()
M = joblib.load(
    "/das/work/p17/p17491/Cecilia_Casadei/NLSA/synthetic_data_jitter/test6/mask.jbl"
)
print(M.dtype, M.shape, sparse.issparse(M))
Mtr = M.transpose()
mdic = {"M": Mtr}
scipy.io.savemat("model_sparse_partial_jitter_1p0_M.mat", mdic)