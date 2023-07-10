# -*- coding: utf-8 -*-
import joblib
from scipy import sparse


def main(settings):

    #label = settings.label
    folder = settings.results_path

    # LIGHT
    # T = joblib.load("%s/dT_sparse_LTD_%s.jbl" % (folder, label))
    # M = joblib.load("%s/M_sparse_%s.jbl" % (folder, label))

    # Syn data
    T = joblib.load( '%s/dT.jbl'%(folder))
    M = joblib.load( '%s/input_data_mask_sparsity_systematic.jbl'%(folder))
    

    print("T: is sparse: ", sparse.issparse(T), T.dtype, T.shape)
    print("M: is sparse: ", sparse.issparse(M), M.dtype, M.shape)

    if sparse.issparse(T) == False:
        print("T is not sparse.")
        T = sparse.csr_matrix(T)
        print("T, is sparse:", sparse.issparse(T), T.dtype, T.shape)
    if sparse.issparse(M) == False:
        print("M is not sparse.")
        M = sparse.csr_matrix(M)
        print("M, is sparse:", sparse.issparse(M), M.dtype, M.shape)

    n_obs = M.sum(axis=1)
    print("n_obs: is sparse: ", sparse.issparse(n_obs), n_obs.dtype, n_obs.shape)

    T_bst = T / n_obs
    T_bst = settings.S * T_bst

    print("T_bst: is sparse: ", sparse.issparse(T_bst), T_bst.dtype, T_bst.shape)

    # print(T[100, 250:350])
    # print(n_obs[100, 0])
    # print(T_bst[100, 250:350])

    T_bst_sparse = sparse.csr_matrix(T_bst)
    joblib.dump(T_bst_sparse, "%s/dT_bst.jbl" % (folder))

    print("dT_bst_sparse: is sparse: ", sparse.issparse(
        T_bst_sparse
    ), T_bst_sparse.dtype, T_bst_sparse.shape)