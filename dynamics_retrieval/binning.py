#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:58:01 2023

@author: casadei_c
"""
import joblib
import matplotlib.pyplot
import numpy

import dynamics_retrieval.correlate
import dynamics_retrieval.plot_syn_data


def binning_f(settings):

    results_path = settings.results_path

    x = joblib.load("%s/x.jbl" % results_path)
    print("x: ", x.shape)

    benchmark = joblib.load("../../synthetic_data_5/test1/x.jbl")
    print(benchmark.shape)

    bin_sizes = []
    CCs = []

    bin_size = 1
    bin_sizes.append(bin_size)
    CC = dynamics_retrieval.correlate.Correlate(benchmark.flatten(), x.flatten())
    CCs.append(CC)
    print(CC)

    for n in range(200, 2200, 200):
        bin_size = 2 * n + 1
        bin_sizes.append(bin_size)

        x_binned = numpy.zeros((settings.m, settings.S - bin_size + 1))
        for i in range(x_binned.shape[1]):
            x_avg = numpy.average(x[:, i : i + bin_size], axis=1)
            x_binned[:, i] = x_avg

        CC = dynamics_retrieval.correlate.Correlate(
            benchmark[:, n:-n].flatten(), x_binned.flatten()
        )
        CCs.append(CC)
        print("Bin size: ", bin_size, "CC: ", CC)

        x_large = numpy.zeros((settings.m, settings.S))
        x_large[:] = numpy.nan
        x_large[:, n:-n] = x_binned

        dynamics_retrieval.plot_syn_data.f(
            x_large,
            "%s/x_r_binsize_%d.png" % (results_path, bin_size),
            "Bin size: %d CC: %.4f" % (bin_size, CC),
        )

    joblib.dump(bin_sizes, "%s/binsizes.jbl" % results_path)
    joblib.dump(CCs, "%s/reconstruction_CC_vs_binsize.jbl" % results_path)


def plot(settings):
    results_path = settings.results_path

    bin_sizes = joblib.load("%s/binsizes.jbl" % results_path)
    CCs = joblib.load("%s/reconstruction_CC_vs_binsize.jbl" % results_path)

    matplotlib.pyplot.plot(bin_sizes, CCs, "o-", c="b")
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c="k", linewidth=1)
    matplotlib.pyplot.xlabel("bin size", fontsize=14)
    matplotlib.pyplot.ylabel("CC", fontsize=14)
    matplotlib.pyplot.xlim(left=bin_sizes[0] - 100, right=bin_sizes[-1] + 100)
    matplotlib.pyplot.savefig(
        "%s/reconstruction_CC_vs_binsize.pdf" % (results_path), dpi=4 * 96
    )
    matplotlib.pyplot.close()
