# -*- coding: utf-8 -*-
import joblib
import matplotlib
import numpy
from scipy import sparse

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot


def calculate_W_localspeeds(D, N, v, results_path, datatype, sigma_sq):
    s = D.shape[0]
    b = D.shape[1]
    print(s, b)

    W = numpy.empty((s, s), dtype=datatype)
    W[:] = numpy.nan

    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        v_i = v[i]
        for j in range(b):
            d_i_neighbor = D[i, j]
            neighbor_idx = N[i, j]
            v_neighbor = v[neighbor_idx]
            W[i, neighbor_idx] = numpy.exp(
                -(d_i_neighbor * d_i_neighbor) / (sigma_sq * v_i * v_neighbor)
            )
    return W


def calculate_W(D, N, results_path, datatype, sigma_sq):
    s = D.shape[0]
    b = D.shape[1]
    print(s, b, sigma_sq)

    W = numpy.empty((s, s), dtype=datatype)
    W[:] = numpy.nan

    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        for j in range(b):
            d_i_neighbor = D[i, j]
            neighbor_idx = N[i, j]
            W[i, neighbor_idx] = numpy.exp(-(d_i_neighbor * d_i_neighbor) / (sigma_sq))

    return W


def symmetrise_W(W, datatype):
    s = W.shape[0]
    W_sym = numpy.empty((s, s), dtype=datatype)
    W_sym[:] = numpy.nan

    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        for j in range(s):
            if not numpy.isnan(W[i, j]):
                W_sym[i, j] = W[i, j]
            elif not numpy.isnan(W[j, i]):
                W_sym[i, j] = W[j, i]
    return W_sym


def symmetrise_W_optimised(W, datatype):
    s = W.shape[0]
    W_sym = numpy.empty((s, s), dtype=datatype)
    W_sym[:] = numpy.nan

    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        row_i = W[i, :].flatten()
        col_i = W[:, i].flatten()
        m_row_i = numpy.ones((s,))
        m_col_i = numpy.ones((s,))

        m_row_i[numpy.isnan(row_i)] = 0
        m_col_i[numpy.isnan(col_i)] = 0

        m_i = m_row_i + m_col_i

        row_i[numpy.isnan(row_i)] = 0
        col_i[numpy.isnan(col_i)] = 0

        sym_i = (row_i + col_i) / m_i  # Divisions by zero will be nan

        W_sym[i, :] = sym_i
        W_sym[:, i] = sym_i

    return W_sym


def check(settings):
    results_path = settings.results_path
    W_sym = joblib.load("%s/W_sym.jbl" % results_path)
    counts = []
    for i in range(W_sym.shape[0]):
        count = W_sym[i, :].count_nonzero()
        counts.append(count)
    matplotlib.pyplot.figure(figsize=(30, 10))
    matplotlib.pyplot.plot(list(range(W_sym.shape[0])), counts)
    matplotlib.pyplot.savefig("%s/W_sym_nns.png" % results_path)
    matplotlib.pyplot.close()


def main(settings):

    print("\n****** RUNNING tansition_matrix ****** ")

    sigma_sq = settings.sigma_sq
    print("Sigma_sq: ", sigma_sq)
    epsilon = sigma_sq / 2
    print("Epsilon: ", epsilon)
    log10_epsilon = numpy.log10(epsilon)
    print("Log10(epsilon): ", log10_epsilon)

    results_path = settings.results_path
    datatype = settings.datatype

    N = joblib.load("%s/N.jbl" % results_path)
    D = joblib.load("%s/D.jbl" % results_path)

    print("Calculate W")
    W = calculate_W(D, N, results_path, datatype, sigma_sq)
    joblib.dump(W, "%s/W.jbl" % results_path)

    print("Symmetrise W (optimised)")
    W_sym = symmetrise_W_optimised(W, datatype)

    print("W_sym: Set NaN values to zero")
    W_sym[numpy.isnan(W_sym)] = 0

    print("W_sym", W_sym.shape, W_sym.dtype)

    print("Check that W_sym is symmetric:")
    diff = W_sym - W_sym.T
    print(numpy.amax(diff), numpy.amin(diff))

    joblib.dump(W_sym, "%s/W_sym.jbl" % results_path)
