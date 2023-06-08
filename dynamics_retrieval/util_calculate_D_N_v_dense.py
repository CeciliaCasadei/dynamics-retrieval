# -*- coding: utf-8 -*-
import pickle

import numpy
import settings


def f():
    print("\n****** RUNNING calculate_D_N_v_dense.******")

    q = settings.q  # Concatenation number
    b = settings.b  # N. of nearest neighbors
    results_path = settings.results_path

    f = open("%s/X_backward_q_%d.pkl" % (results_path, q), "rb")
    X = pickle.load(f)
    f.close()

    s = X.shape[1] - 1  # s+1 is the n. of concatenated vectors

    Velocities = numpy.zeros((s, 1), dtype=numpy.float64)
    for i in range(1, s + 1):
        X1 = X[:, i - 1]
        X2 = X[:, i]
        d_X1_X2 = (X1 - X2) ** 2
        d_X1_X2 = d_X1_X2.sum()
        d_X1_X2 = numpy.sqrt(d_X1_X2)
        Velocities[i - 1] = d_X1_X2

    D = numpy.zeros((s, b), dtype=numpy.float64)
    N = numpy.zeros((s, b), dtype=numpy.uint)

    # Measure for each sample i distances to all neighbors j
    for i in range(1, s + 1):
        print(i)
        X_i = X[:, i]
        d_i = numpy.zeros((s), dtype=numpy.float64)
        for j in range(1, s + 1):
            X_j = X[:, j]
            d_Xi_Xj = (X_i - X_j) ** 2
            d_Xi_Xj = d_Xi_Xj.sum()
            d_Xi_Xj = numpy.sqrt(d_Xi_Xj)
            d_i[j - 1] = d_Xi_Xj
        sort_idxs = numpy.argsort(d_i)
        d_i_sorted = d_i[sort_idxs]
        D[i - 1, :] = d_i_sorted[0:b]
        N[i - 1, :] = sort_idxs[0:b]

    fn = open("%s/D_standard.pkl" % results_path, "wb")
    pickle.dump(D, fn)
    fn.close()

    fn = open("%s/N_standard.pkl" % results_path, "wb")
    pickle.dump(N, fn)
    fn.close()

    fn = open("%s/v_standard.pkl" % results_path, "wb")
    pickle.dump(Velocities, fn)
    fn.close()
