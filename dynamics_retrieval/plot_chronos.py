# -*- coding: utf-8 -*-
import os

import joblib
import matplotlib.pyplot
import numpy


def f(xs, ys, fn):
    matplotlib.pyplot.figure(figsize=(30, 10))
    matplotlib.pyplot.plot(xs, ys, c="b", markeredgewidth=0)
    ax = matplotlib.pyplot.gca()
    ax.tick_params(axis="x", labelsize=55)
    ax.tick_params(axis="y", labelsize=55)
    matplotlib.pyplot.locator_params(axis="y", nbins=3)
    # ax.axes.get_xaxis().set_ticks([])
    matplotlib.pyplot.savefig(fn, dpi=2 * 96)
    matplotlib.pyplot.close()


def plot(settings):
    results_path = settings.results_path

    VT_final = joblib.load("%s/VT_final.jbl" % (results_path))

    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]

    print("nmodes: ", nmodes)
    print("s: ", s)

    out_folder = "%s/chronos" % (results_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for i in range(0, min(20, nmodes)):
        print(i)
        chrono = VT_final[i, :]
        f(list(range(s)), chrono, "%s/chrono_%d.png" % (out_folder, i))


def plot_ts(settings):
    results_path = settings.results_path

    VT_final = joblib.load("%s/VT_final.jbl" % results_path)
    ts = joblib.load("%s/ts_svs.jbl" % results_path)

    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]

    print("nmodes: ", nmodes)
    print("s: ", s)

    out_folder = "%s/chronos" % (results_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for i in range(0, min(20, nmodes)):
        print(i)
        chrono = VT_final[i, :]

        f(ts, chrono, "%s/chrono_ts_%d.png" % (out_folder, i))
        f(ts, -chrono, "%s/chrono_ts_%d_minussign.png" % (out_folder, i))
        f(
            ts[0 : int(float(2*settings.S - settings.q + 1)/2)],# settings.S - settings.q - 1],
            chrono[0 : int(float(2*settings.S - settings.q + 1)/2)], #settings.S - settings.q - 1],
            "%s/chrono_ts_%d_firsthalf.png" % (out_folder, i),
        )
        f(
            ts[0 : int(float(2*settings.S - settings.q + 1)/2)],
            -chrono[0 : int(float(2*settings.S - settings.q + 1)/2)],
            "%s/chrono_ts_%d_firsthalf_minussign.png" % (out_folder, i),
        )


def main(settings):
    plot_ts(settings)
