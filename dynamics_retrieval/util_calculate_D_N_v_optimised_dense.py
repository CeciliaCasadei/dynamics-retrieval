# -*- coding: utf-8 -*-
import pickle

import numpy
import settings_NP as settings


def f():
    print("\n****** RUNNING calculate_D_N_v_optimised_dense ******")
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

    print("Start")
    X_work = X[:, 1:]
    X_work = X_work.T

    X_work_2 = numpy.einsum("ij,ij->i", X_work, X_work)[:, numpy.newaxis]
    Y_work_2 = numpy.einsum("ij,ij->i", X_work, X_work)[numpy.newaxis, :]

    D_all_sq = X_work_2 + Y_work_2 - 2 * numpy.dot(X_work, X_work.T)

    print("D_all_sq:", D_all_sq.shape)
    test = numpy.argwhere(D_all_sq < 0)
    print(test.shape[0], "negative values in D_all_sq")
    test = numpy.argwhere(numpy.diag(D_all_sq) < 0)
    print(test.shape[0], "negative values in diag(D_all_sq)")

    print("D_all_sq min value: ", numpy.amin(D_all_sq))
    print("D_all_sq max value: ", numpy.amax(D_all_sq))

    print("\nSet negative values to zero.")
    D_all_sq[D_all_sq < 0] = 0
    test = numpy.argwhere(D_all_sq < 0)
    print(test.shape[0], "negative values in D_all_sq")
    print("Take sqrt(D_all_sq).")
    D_all = numpy.sqrt(D_all_sq)
    print("Done")

    diff = D_all - D_all.T
    print("D_all is symmetric:")
    print(numpy.amin(diff), numpy.amax(diff))

    print("Sorting: ")
    idxs = numpy.argsort(D_all, axis=1)
    D_all_sorted = numpy.sort(D_all, axis=1)
    print("idxs: ", idxs.shape)

    D = D_all_sorted[:, 0:b]
    N = idxs[:, 0:b]

    print("Saving")

    D[:, 0] = 0

    fn = open("%s/D_optimised.pkl" % results_path, "wb")
    pickle.dump(D, fn)
    fn.close()

    fn = open("%s/N_optimised.pkl" % results_path, "wb")
    pickle.dump(N, fn)
    fn.close()

    fn = open("%s/v_optimised.pkl" % results_path, "wb")
    pickle.dump(Velocities, fn)
    fn.close()
