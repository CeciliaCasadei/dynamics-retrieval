#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:58:30 2023

@author: casadei_c
"""
import joblib
import matplotlib
import numpy

import dynamics_retrieval.correlate as correlate

#####################
#####  q-scan  ######
#####################


def get_x_r_large_q_scan(q, nmodes):
    modulename = "settings_q_%d" % q
    settings_scan = __import__(modulename)

    results_path = settings_scan.results_path
    p = settings_scan.p
    q = settings_scan.q
    S = settings_scan.S
    m = settings_scan.m

    print("p: ", p)
    print("q: ", q)
    print("S: ", S)
    print("m: ", m)

    x_r_tot = 0
    for mode in range(0, nmodes):
        print("Mode: ", mode)
        x_r = joblib.load(
            "%s/reconstruction_p_%d/movie_p_%d_mode_%d.jbl" % (results_path, p, p, mode)
        )
        x_r_tot += x_r

    x_r_tot_large = numpy.zeros((m, S))

    if p == 0:
        x_r_tot_large[:, (q - 1) / 2 : (q - 1) / 2 + (S - q + 1)] = x_r_tot
        return x_r_tot_large
    else:
        print("Case p>0 to be implemented")


def get_CCs_q_scan(settings, qs, nmodes):

    f_max = settings.f_max_q_scan
    S = settings.S
    results_path = settings.results_path
    p = settings.p

    if p == 0:
        start_large = get_x_r_large_q_scan(qs[0], nmodes)

        q_max = qs[-1]
        CCs = []
        for q in qs[1:]:
            x_r_tot_large = get_x_r_large_q_scan(q, nmodes)
            start_flat = start_large[
                :, (q_max - 1) / 2 : (q_max - 1) / 2 + (S - q_max + 1)
            ].flatten()
            x_r_tot_flat = x_r_tot_large[
                :, (q_max - 1) / 2 : (q_max - 1) / 2 + (S - q_max + 1)
            ].flatten()
            print(start_flat.shape, x_r_tot_flat.shape)
            CC = correlate.Correlate(start_flat, x_r_tot_flat)
            CCs.append(CC)
            print(CC)
            start_large = x_r_tot_large
        joblib.dump(
            CCs,
            "%s/LPSA_f_max_%d_q_scan_p_%s_%d_modes_successive_CCs.jbl"
            % (results_path, f_max, p, nmodes),
        )

    else:
        print("Case p>0 to be implemented")


def plot_CCs_q_scan(settings, qs, nmodes):

    f_max = settings.f_max_q_scan
    results_path = settings.results_path
    p = settings.p

    CCs = joblib.load(
        "%s/LPSA_f_max_%d_q_scan_p_%d_%d_modes_successive_CCs.jbl"
        % (results_path, f_max, p, nmodes)
    )

    matplotlib.rcParams["axes.formatter.useoffset"] = False
    matplotlib.pyplot.figure(figsize=(15, 7))
    matplotlib.pyplot.xticks(qs[1:])
    labs = []
    for i, CC in enumerate(CCs):
        matplotlib.pyplot.scatter(qs[i + 1], CCs[i], c="b")
        labs.append("$q=$(%d,%d)" % (qs[i], qs[i + 1]))
    matplotlib.pyplot.axhline(y=1.0, c="k")
    matplotlib.pyplot.gca().set_xticklabels(labs, rotation=45, fontsize=18)
    matplotlib.pyplot.ylim(top=1.002)
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=18)
    matplotlib.pyplot.locator_params(axis="y", nbins=6)
    matplotlib.pyplot.ylabel("correlation coefficient", fontsize=20)
    matplotlib.pyplot.savefig(
        "%s/LPSA_f_max_%d_q_scan_p_%d_%d_modes_successive_CCs.png"
        % (results_path, f_max, p, nmodes),
        bbox_inches="tight",
        dpi=96 * 2,
    )
    matplotlib.pyplot.close()


########################
#####  jmax-scan  ######
########################


def get_x_r(fmax, nmodes):
    modulename = "settings_f_max_%d" % fmax
    settings_scan = __import__(modulename)
    print("jmax: ", settings_scan.f_max)
    results_path = settings_scan.results_path
    p = settings_scan.p
    print(results_path)
    x_r_tot = 0
    for mode in range(0, min(nmodes, 2 * fmax + 1)):
        print("Mode: ", mode)
        x_r = joblib.load(
            "%s/reconstruction_p_%d/movie_p_%d_mode_%d.jbl" % (results_path, p, p, mode)
        )
        x_r_tot += x_r
    return x_r_tot


def get_CCs_jmax_scan(settings, f_max_s, nmodes):
    q = settings.q_f_max_scan
    results_path = settings.results_path
    p = settings.p

    start = get_x_r(f_max_s[0], nmodes)
    print(start.shape)

    CCs = []
    for f_max in f_max_s[1:]:
        x_r_tot = get_x_r(f_max, nmodes)
        print(x_r_tot.shape)
        start_flat = start.flatten()
        x_r_tot_flat = x_r_tot.flatten()
        CC = correlate.Correlate(start_flat, x_r_tot_flat)
        CCs.append(CC)
        print(CC)
        start = x_r_tot
    joblib.dump(
        CCs,
        "%s/LPSA_f_max_scan_q_%d_p_%d_%d_modes_successive_CCs.jbl"
        % (results_path, q, p, nmodes),
    )


def plot_CCs_jmax_scan(settings, f_max_s, nmodes):

    q = settings.q_f_max_scan
    results_path = settings.results_path
    p = settings.p

    CCs = joblib.load(
        "%s/LPSA_f_max_scan_q_%d_p_%d_%d_modes_successive_CCs.jbl"
        % (results_path, q, p, nmodes)
    )

    matplotlib.rcParams["axes.formatter.useoffset"] = False
    matplotlib.pyplot.figure(figsize=(15, 7))
    matplotlib.pyplot.xticks(f_max_s[1:])
    labs = []
    for i, CC in enumerate(CCs):
        matplotlib.pyplot.scatter(f_max_s[i + 1], CCs[i], c="b")
        labs.append("$j_{\mathrm{max}}=$(%d, %d)" % (f_max_s[i], f_max_s[i + 1]))
    matplotlib.pyplot.gca().set_xticklabels(labs, rotation=45, fontsize=14)
    matplotlib.pyplot.axhline(y=1.0, c="k")
    matplotlib.pyplot.gca().set_xlim([0, 35])
    matplotlib.pyplot.ylim(top=1.01)
    matplotlib.pyplot.locator_params(axis="y", nbins=6)
    matplotlib.pyplot.ylabel("correlation coefficient", fontsize=20)
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=15)

    matplotlib.pyplot.savefig(
        "%s/LPSA_f_max_scan_q_%d_p_%d_%d_modes_successive_CCs.png"
        % (results_path, q, p, nmodes),
        bbox_inches="tight",
        dpi=96 * 4,
    )
    matplotlib.pyplot.close()
