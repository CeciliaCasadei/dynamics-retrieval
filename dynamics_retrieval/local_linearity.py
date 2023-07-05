#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:45:10 2022

@author: casadei_c
"""
import joblib
import matplotlib.pyplot
import numpy


def local_linearity_measure(x):
    approx_l = x[:, 0:-3] - (x[:, 1:-2] - 1 * (x[:, 2:-1] - x[:, 1:-2]))
    # print 'approx_l:', approx_l.shape
    approx_r = x[:, 3:] - (x[:, 1:-2] + 2 * (x[:, 2:-1] - x[:, 1:-2]))
    # print 'approx_r:', approx_r.shape
    L = 0.5 * (abs(approx_l) + abs(approx_r))
    # print L.shape
    L = numpy.average(L)
    print(L)
    return L


def local_linearity_measure_jitter(x, ts_r):
    print(x.shape, ts_r.shape)
    approx_l = x[:, 0:-3] - (
        x[:, 1:-2]
        - numpy.multiply(
            (ts_r[1:-2] - ts_r[0:-3]),
            ((x[:, 2:-1] - x[:, 1:-2]) / (ts_r[2:-1] - ts_r[1:-2])),
        )
    )
    # print 'approx_l:', approx_l.shape
    approx_r = x[:, 3:] - (
        x[:, 2:-1]
        + numpy.multiply(
            (ts_r[3:] - ts_r[2:-1]),
            ((x[:, 2:-1] - x[:, 1:-2]) / (ts_r[2:-1] - ts_r[1:-2])),
        )
    )
    # print 'approx_r:', approx_r.shape
    L = 0.5 * (abs(approx_l) + abs(approx_r))
    # print L.shape
    L = numpy.average(L)
    print(L)
    return L


def get_L(settings):
    p = settings.p
    results_path = "%s/reconstruction_p_%d" % (settings.results_path, p)
    t_r = joblib.load("%s/t_r_p_%d.jbl" % (results_path, p))

    Ls = []
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print("Mode: ", mode)
        x_r = joblib.load("%s/movie_p_%d_mode_%d.jbl" % (results_path, p, mode))
        x_r_tot += x_r
        L = local_linearity_measure_jitter(x_r_tot, t_r)
        Ls.append(L)

    joblib.dump(Ls, "%s/p_%d_local_linearity_vs_nmodes.jbl" % (results_path, p))

    matplotlib.pyplot.scatter(list(range(1, len(Ls) + 1)), numpy.log10(Ls), c="b")
    matplotlib.pyplot.xticks(list(range(1, len(Ls) + 1, 2)))
    matplotlib.pyplot.savefig(
        "%s/p_%d_log10_L_vs_nmodes_q_%d_fmax_%d.png"
        % (results_path, p, settings.q, settings.f_max_considered)
    )
    matplotlib.pyplot.close()
