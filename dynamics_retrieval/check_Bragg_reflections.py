#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:27:25 2023

@author: casadei_c
"""
import joblib
import matplotlib.pyplot
import numpy


def f(path, Bragg_i, q, p, f_max_s, nmodes, bin_size):

    fig, axes = matplotlib.pyplot.subplots(
        len(f_max_s) + 1, 1, sharex=True, figsize=(10, 18)
    )

    # LPSA RESULT
    for i, f_max in enumerate(f_max_s):

        results_path = "%s/f_max_%d_q_%d" % (path, f_max, q)
        x_r_tot = joblib.load(
            "%s/reconstruction_p_%d/movie_p_%d_sum_%d_modes.jbl"
            % (results_path, p, p, nmodes)
        )
        print("x_r_tot: ", x_r_tot.shape)
        t_r = joblib.load("%s/reconstruction_p_%d/t_r_p_%d.jbl" % (results_path, p, p))
        print("t_r: ", t_r.shape)
        axes[i].set_title("$j_{\mathrm{max}}=$%d, n.modes=%d" % (f_max, nmodes))
        axes[i].plot(t_r, x_r_tot[Bragg_i, :], c="b")

    # BINNING RESULT
    dT = joblib.load("%s/dT_sparse_light.jbl" % path)
    M = joblib.load("%s/M_sel_sparse_light.jbl" % path)
    ts = joblib.load("%s/ts_sel_light.jbl" % path)
    ts = ts.flatten()
    print(dT.shape, M.shape, ts.shape)

    ts_plot = []
    Is_plot = []
    dT_i = dT[Bragg_i, :].todense().tolist()[0]
    dT_i = numpy.asarray(dT_i)
    print(dT_i.shape)
    M_i = M[Bragg_i, :].todense().tolist()[0]
    M_i = numpy.asarray(M_i)
    print(M_i.shape)
    for j in range(0, (dT_i.shape[0]) - bin_size):
        M_sum = M_i[j : j + bin_size].sum()
        if M_sum > 0:
            I_sum = dT_i[j : j + bin_size].sum()
            I_avg = I_sum / M_sum
            t_avg = numpy.average(ts[j : j + bin_size])
            ts_plot.append(t_avg)
            Is_plot.append(I_avg)

    axes[-1].plot(ts_plot, Is_plot, c="b")
    axes[-1].set_title("bin size: %d" % bin_size)

    matplotlib.pyplot.savefig(
        "%s/Bragg_pt_%d_%d_modes.png" % (path, Bragg_i, nmodes), dpi=96 * 4
    )
    matplotlib.pyplot.close()


def f_ps(path, Bragg_i, q, ps, f_max, nmodes, bin_size):

    fig, axes = matplotlib.pyplot.subplots(2, 1, sharex=True, figsize=(15, 18))
    # LPSA RESULT
    cs = ["m", "b", "c", "k"]
    for i, p in enumerate(ps):

        results_path = "%s/f_max_%d_q_%d" % (path, f_max, q)
        if p == 0:
            x_r_tot = joblib.load(
                "%s/reconstruction_p_%d/movie_p_%d_sum_%d_modes.jbl"
                % (results_path, p, p, nmodes)
            )
        elif p == 10000:
            x_r_tot = joblib.load(
                "%s/reconstruction_p_%d/movie_sum_%d_modes.jbl"
                % (results_path, p, nmodes)
            )

        print(x_r_tot.shape)
        t_r = joblib.load("%s/reconstruction_p_%d/t_r_p_%d.jbl" % (results_path, p, p))
        print(t_r.shape)
        axes[0].plot(t_r, x_r_tot[Bragg_i, :], c=cs[i], label="p=%d" % (p))

    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].legend(frameon=False, loc="upper right", fontsize=20)
    axes[0].set_title(
        "q=%d, $j_{\mathrm{max}}=$%d, n.modes=%d" % (q, f_max, nmodes), fontsize=20
    )

    # BINNING RESULTS
    dT = joblib.load("%s/dT_sparse_light.jbl" % path)
    M = joblib.load("%s/M_sel_sparse_light.jbl" % path)
    ts = joblib.load("%s/ts_sel_light.jbl" % path)
    ts = ts.flatten()
    print(dT.shape, M.shape, ts.shape)

    ts_plot = []
    Is_plot = []
    dT_i = dT[Bragg_i, :].todense().tolist()[0]
    dT_i = numpy.asarray(dT_i)
    print(dT_i.shape)
    M_i = M[Bragg_i, :].todense().tolist()[0]
    M_i = numpy.asarray(M_i)
    print(M_i.shape)
    for j in range(0, (dT_i.shape[0]) - bin_size):
        M_sum = M_i[j : j + bin_size].sum()
        if M_sum > 0:
            I_sum = dT_i[j : j + bin_size].sum()
            I_avg = I_sum / M_sum
            t_avg = numpy.average(ts[j : j + bin_size])
            ts_plot.append(t_avg)
            Is_plot.append(I_avg)
            # print t_avg, I_avg
    axes[-1].plot(ts_plot, Is_plot, c="b", label="bin size %d" % bin_size)
    axes[-1].set_title("bin size: %d" % bin_size, fontsize=20)
    axes[-1].tick_params(axis="both", which="major", labelsize=18)

    # BR RESO
    hs = joblib.load("%s/miller_h_light.jbl" % path)
    ks = joblib.load("%s/miller_k_light.jbl" % path)
    ls = joblib.load("%s/miller_l_light.jbl" % path)

    a = 62.36
    b = 62.39
    c = 111.18
    # q-vector in reciprocal space:
    q_x = hs / a
    q_y = hs / (a * numpy.sqrt(3)) + 2 * ks / (b * numpy.sqrt(3))
    q_z = ls / c
    q_sq = q_x * q_x + q_y * q_y + q_z * q_z
    q = numpy.sqrt(q_sq)
    d = 1.0 / q  # A
    d_i = d[Bragg_i]

    fig.suptitle("Bragg point %d, %.2f A resolution" % (Bragg_i, d_i), fontsize=20)
    matplotlib.pyplot.savefig(
        "%s/Bragg_pt_%d_%d_modes_ps.png" % (path, Bragg_i, nmodes), dpi=96 * 4
    )
    matplotlib.pyplot.close()
