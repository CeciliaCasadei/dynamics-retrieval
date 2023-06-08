# -*- coding: utf-8 -*-
import joblib
import numpy


def f(settings):
    print("\n****** RUNNING concatenate_backward. ******")
    datatype = settings.datatype
    results_path = settings.results_path
    q = settings.q
    data_file = settings.data_file  # Concatenation n.
    # sparsity = settings.sparsity

    # o = open('%s/T_anomaly_sparse_%.2f.pkl'%(results_path, sparsity), 'rb')
    # o = open('%s/T_anomaly.pkl'%(results_path), 'rb')
    # o = open('%s/T.pkl'%(results_path), 'rb')
    # x = pickle.load(o)
    x = joblib.load(data_file)

    m = x.shape[0]  # Size of physical space
    S = x.shape[1]  # N. ofsamples (original time points
    n = m * q  # Size of embedding space
    s = S - q  # N.of concatenated vectors

    X = numpy.zeros((n, s + 1), dtype=datatype)
    for j in range(s + 1):
        if j % 500 == 0:
            print(j, " / ", s + 1)
        # column j of X
        for i in range(q):
            # rows to fill in
            i_start_row = i * m
            i_end_row = i_start_row + m

            X[i_start_row:i_end_row, j] = x[:, j + q - i - 1]

    print("Concatenated data shape: ", X.shape)

    f = "%s/X_backward_q_%d.jbl" % (results_path, q)
    joblib.dump(X, f)
