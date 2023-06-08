# -*- coding: utf-8 -*-
import os

import joblib
import numpy
from scipy import sparse


def extract(settings, x_r, nm):
    results_path = settings.results_path
    label = settings.label
    p = settings.p

    miller_h = joblib.load("%s/miller_h_%s.jbl" % (results_path, label))
    miller_k = joblib.load("%s/miller_k_%s.jbl" % (results_path, label))
    miller_l = joblib.load("%s/miller_l_%s.jbl" % (results_path, label))

    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()

    fpath = "%s/reconstruction_p_%d/extracted_Is" % (results_path, p)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    for i in range(0, x_r.shape[1], 100):
        print(i)
        f_out = "%s/extracted_Is_p_%d_%d_modes_timestep_%0.6d.txt" % (fpath, p, nm, i)

        I = numpy.squeeze(x_r[:, i])
        # I = x_r[:, i]
        print("I: ", I.shape)

        I[numpy.isnan(I)] = 0
        I[I < 0] = 0
        sigIs = numpy.sqrt(I)

        out[:, 3] = I
        out[:, 4] = sigIs

        numpy.savetxt(f_out, out, fmt="%6d%6d%6d%20.2f%16.2f")


def f(settings, nmodes):
    p = settings.p
    results_path = "%s/reconstruction_p_%d"(settings.results_path, p)

    x_r_tot = 0
    for mode in range(0, nmodes):
        print("Mode: ", mode)
        x_r = joblib.load("%s/movie_p_%d_mode_%d.jbl" % (results_path, p, mode))
        x_r_tot += x_r

    joblib.dump(x_r_tot, "%s/x_r_p_%d_sum_%d_modes.jbl" % (results_path, p, nmodes))
    extract(settings, x_r_tot, nmodes)


def f_static_plus_dynamics(settings, nmodes):
    results_path = settings.results_path
    label = settings.label
    p = settings.p
    q = settings.q
    S = settings.S

    ncopies = 2 * p + 1

    T = joblib.load("%s/T_sparse_LTD_%s.jbl" % (results_path, label))
    M = joblib.load("%s/M_sparse_%s.jbl" % (results_path, label))
    print("T: is sparse: ", sparse.issparse(T), T.shape, T.dtype)
    print("M: is sparse: ", sparse.issparse(M), M.shape, M.dtype)

    ns = numpy.sum(M, axis=1)
    avgs = numpy.sum(T, axis=1) / ns
    print("avgs: ", avgs.shape)
    print(avgs[100, 0])

    avgs_matrix = numpy.tile(avgs, (1, S - q + 1 - ncopies + 1))
    print("avgs_matrix: ", avgs_matrix.shape)
    print(avgs_matrix[100, 0:5])

    x_r_tot = 0
    for mode in range(0, nmodes):
        print("Mode: ", mode)
        x_r = joblib.load(
            "%s/reconstruction_p_%d/movie_p_%d_mode_%d.jbl" % (results_path, p, p, mode)
        )
        x_r_tot += x_r
    print("x_r_tot: ", x_r_tot.shape)
    print(x_r_tot[100, 9])

    x_r_tot_plus_avg = x_r_tot + avgs_matrix
    print("x_r_tot_plus_avg: ", x_r_tot_plus_avg.shape)
    print(x_r_tot_plus_avg[100, 9])
    joblib.dump(
        x_r_tot_plus_avg,
        "%s/reconstruction_p_%d/x_r_tot_plus_avg_p_%d_%d_modes.jbl"
        % (results_path, p, p, nmodes),
    )
    extract(settings, x_r_tot_plus_avg, nmodes)
