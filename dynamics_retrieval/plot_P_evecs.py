# -*- coding: utf-8 -*-
import joblib
import matplotlib

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams["agg.path.chunksize"] = 10000
import os

import matplotlib.pyplot


def main(settings):

    print("****** RUNNING plot_P_evecs ******")
    results_path = settings.results_path
    l = settings.l

    evals_sorted = joblib.load("%s/evals_sorted.jbl" % (results_path))
    evecs_sorted = joblib.load("%s/evecs_sorted.jbl" % (results_path))

    figure_path = "%s/evecs" % (results_path)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    s = evecs_sorted.shape[0]
    for i in range(l):
        print(i)
        phi = evecs_sorted[:, i]
        matplotlib.pyplot.figure(figsize=(30, 10))
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        matplotlib.pyplot.plot(list(range(s)), phi, "o-", markersize=2)
        matplotlib.pyplot.savefig("%s/evec_%d.png" % (figure_path, i), dpi=2 * 96)
        matplotlib.pyplot.close()

    matplotlib.pyplot.scatter(list(range(l)), evals_sorted[0:l])
    matplotlib.pyplot.savefig("%s/eigenvalues.png" % (figure_path))
    matplotlib.pyplot.close()
