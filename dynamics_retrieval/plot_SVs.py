# -*- coding: utf-8 -*-
import os

import joblib
import matplotlib.pyplot
import numpy


def f(xs, ys, xticks, fn):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xticks(xticks)
    matplotlib.pyplot.scatter(xs, ys, s=5, c="b")
    matplotlib.pyplot.savefig(fn, dpi=96 * 2)
    matplotlib.pyplot.close()


def main(settings):

    results_path = settings.results_path

    S = joblib.load("%s/S.jbl" % results_path)
    print(S.shape)
    if S.shape[0] >= 20:
        nmodes = 20
    else:
        nmodes = S.shape[0]
    print("nmodes: ", nmodes)
    for i in range(nmodes):
        print("mode", i, "S", S[i])

    out_folder = "%s/chronos" % results_path
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    s0 = S[0]
    S_norm = S / s0

    my_xs = list(range(1, nmodes + 1))
    my_xticks = list(range(1, nmodes + 1, 2))

    f(my_xs, S[0:nmodes], my_xticks, "%s/SVs.png" % out_folder)
    f(my_xs, S_norm[0:nmodes], my_xticks, "%s/SVs_norm.png" % out_folder)
    f(
        my_xs,
        numpy.log10(S_norm[0:nmodes]),
        my_xticks,
        "%s/SVs_norm_log.png" % out_folder,
    )
