# -*- coding: utf-8 -*-
import joblib
import matplotlib
import numpy

matplotlib.use("Agg")
import matplotlib.pyplot


def plot(k_miller, T, data_path, label, corr_label):
    Is_avg = []
    for i in range(40):
        idxs = numpy.argwhere(k_miller == i)[:, 0]
        T_temp = T[idxs, :]
        avg = T_temp.sum() / T_temp.nnz
        Is_avg.append(avg)

    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.plot(list(range(len(Is_avg))), Is_avg, "-o")
    matplotlib.pyplot.xlabel(r"$k$", fontsize=36)
    matplotlib.pyplot.ylabel(r"$<I>_k$", fontsize=36)
    matplotlib.pyplot.xticks(fontsize=36)
    matplotlib.pyplot.yticks(fontsize=36)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig("%s/%s_Is%s_vs_k.png" % (data_path, label, corr_label))
    matplotlib.pyplot.close()


def main(settings):
    label = settings.label
    data_path = settings.data_path
    datatype = settings.datatype
    t_d_y = settings.t_d_y

    print("label: ", label)
    print("Data path: ", data_path)
    print("Datatype: ", datatype)
    print("t_d_y:", t_d_y)

    f_alpha = settings.f_alpha
    f_beta = settings.f_beta
    f_gamma = settings.f_gamma

    k_miller = joblib.load("%s/miller_k_%s.jbl" % (data_path, label))
    print("N. Bragg pts:", k_miller.shape[0])

    T_sparse = joblib.load(
        "%s/T_sparse_%s.jbl" % (data_path, label)
    )  # sparse.csr.csr_matrix

    print("T_sparse: ", T_sparse.shape, T_sparse.dtype)
    m = T_sparse.shape[0]
    S = T_sparse.shape[1]
    print(m, " Bragg pts, ", S, " samples.")

    sum_of_sq = f_alpha * f_alpha + f_beta * f_beta + f_gamma * f_gamma
    factors = 1.0 / (
        sum_of_sq
        + 2
        * (f_alpha * f_beta + f_beta * f_gamma)
        * numpy.cos(2 * numpy.pi * k_miller * t_d_y)
        + 2 * f_alpha * f_gamma * numpy.cos(2 * numpy.pi * k_miller * 2 * t_d_y)
    )

    print("Correction factors: ", factors.shape)
    factors_rep = numpy.repeat(factors, S, axis=1)
    factors_rep = numpy.asarray(factors_rep)
    print("Correction factors: ", factors_rep.shape)
    T_corr = T_sparse.multiply(factors)  # sparse.coo.coo_matrix
    T_corr = T_corr.tocsr()
    T_corr = T_corr.astype(datatype)
    print("T_corr: ", T_corr.shape, T_corr.dtype)  # sparse.csr.csr_matrix
    print("nnz before correction: ", T_sparse.nnz)
    print("nnz after correction: ", T_corr.nnz)
    joblib.dump(T_corr, "%s/T_sparse_LTD_%s.jbl" % (data_path, label))

    plot(k_miller, T_sparse, data_path, label, "")
    plot(k_miller, T_corr, data_path, label, "_corr")
