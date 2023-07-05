#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:21:08 2023

@author: casadei_c
"""

# CMP RECONSTRUCTION TIMES
import joblib
import matplotlib
import numpy


def get_t_r_large(q):
    modulename = "settings_q_%d" % q
    settings_scan = __import__(modulename)
    print("q: ", settings_scan.q)
    S = settings_scan.S
    results_path = settings_scan.results_path

    p = settings_scan.p
    t_r = joblib.load("%s/reconstruction_p_%d/t_r_p_%d.jbl" % (results_path, p, p))

    if p == 0:
        t_r_large = numpy.zeros((S,))
        print(t_r_large.shape)
        t_r_large[
            (q - 1) / 2 : (q - 1) / 2 + (S - q + 1),
        ] = t_r
        return t_r_large
    else:
        print("Case p>0 to be implemented")


def main(settings, qs):

    results_path = settings.results_path
    f_max = settings.f_max_q_scan
    S = settings.S
    p = settings.p
    start_large = get_t_r_large(qs[0])

    q_max = qs[-1]
    avgs = []
    for q in qs[1:]:
        t_r_large = get_t_r_large(q)
        start = start_large[
            (q_max - 1) / 2 : (q_max - 1) / 2 + (S - q_max + 1),
        ]
        t_r = t_r_large[
            (q_max - 1) / 2 : (q_max - 1) / 2 + (S - q_max + 1),
        ]
        print(start.shape, t_r.shape)
        diff = abs(start - t_r)
        avg = numpy.average(diff)
        print(avg)
        avgs.append(avg)
        start_large = t_r_large
    joblib.dump(
        avgs,
        "%s/LPSA_f_max_%d_q_scan_p_%d_successive_abs_dt_avg.jbl"
        % (results_path, f_max, p),
    )
    n_curves = len(avgs)
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15, 1, n_curves))
    matplotlib.pyplot.xticks(list(range(1, len(avgs) + 1, 1)))
    for i, avg in enumerate(avgs):
        matplotlib.pyplot.scatter(
            i + 1, avgs[i], c=colors[i], label="q=%d,q=%d" % (qs[i], qs[i + 1])
        )
    matplotlib.pyplot.legend(
        frameon=True, scatterpoints=1, loc="upper right", fontsize=6
    )
    matplotlib.pyplot.savefig(
        "%s/LPSA_f_max_%d_q_scan_p_%d_successive_abs_dt_avg.png"
        % (results_path, f_max, p),
        dpi=96 * 4,
    )
    matplotlib.pyplot.close()


import settings_bR_light as settings

qs = [1, 1001, 2501, 5001, 7501, 10001, 12501, 15001, 17501, 20001, 22501, 25001]
main(settings, qs)
