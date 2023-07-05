# -*- coding: utf-8 -*-

import joblib
import matplotlib.pyplot
import numpy
import scipy.linalg

numpy.set_printoptions(precision=2)


def plot_f(settings, M, label, ts):

    if M.shape[1] >= 11:
        fig = matplotlib.pyplot.figure(figsize=(35, 40))
        ax = fig.add_subplot(6, 1, 1)
        ax.plot(ts, M[:, 0], color="m")
        ax.tick_params(axis="both", labelsize=26)
        for j in range(1, 6):
            ax = fig.add_subplot(6, 1, j + 1)
            ax.plot(ts, M[:, 2 * j - 1], color="m")
            ax.plot(ts, M[:, 2 * j], color="b")
            ax.tick_params(axis="both", labelsize=34)
            ax.text(
                0.01,
                0.1,
                "j=%d" % j,
                fontsize=32,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        matplotlib.pyplot.savefig(
            "%s/lp_filter_functions%s_0to5.png" % (settings.results_path, label)
        )
        matplotlib.pyplot.close()

    if M.shape[1] >= 41:
        fig = matplotlib.pyplot.figure(figsize=(35, 40))
        ax = fig.add_subplot(5, 1, 1)
        ax.plot(ts, M[:, 0], color="m")
        ax.tick_params(axis="both", labelsize=26)
        for j in range(5, 21, 5):
            ax = fig.add_subplot(5, 1, j / 5 + 1)
            ax.plot(ts, M[:, 2 * j - 1], color="m")
            ax.plot(ts, M[:, 2 * j], color="b")
            ax.tick_params(axis="both", labelsize=34)
            ax.text(
                0.01,
                0.1,
                "j=%d" % j,
                fontsize=32,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        matplotlib.pyplot.savefig(
            "%s/lp_filter_functions%s_0to20.png" % (settings.results_path, label)
        )
        matplotlib.pyplot.close()

    if M.shape[1] >= 101:
        fig = matplotlib.pyplot.figure(figsize=(35, 40))
        ax = fig.add_subplot(6, 1, 1)
        ax.plot(ts, M[:, 0], color="m")
        ax.tick_params(axis="both", labelsize=26)
        for j in range(10, 51, 10):
            ax = fig.add_subplot(6, 1, j / 10 + 1)
            ax.plot(ts, M[:, 2 * j - 1], color="m")
            ax.plot(ts, M[:, 2 * j], color="b")
            ax.tick_params(axis="both", labelsize=34)
            ax.text(
                0.01,
                0.1,
                "j=%d" % j,
                fontsize=32,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        matplotlib.pyplot.savefig(
            "%s/lp_filter_functions%s_0to50.png" % (settings.results_path, label)
        )
        matplotlib.pyplot.close()


def plot(settings, M, label):
    ts = list(range(M.shape[0]))
    plot_f(settings, M, label, ts)


def plot_t_sv_range(settings, M, label):
    ts = joblib.load("%s/ts_svs.jbl" % settings.results_path)
    plot_f(settings, M, label, ts)


def get_F(settings):
    S = settings.S
    q = settings.q
    s = S - q

    t = numpy.asarray(list(range(s)))
    T = 1 * (s)
    omega = 2 * numpy.pi / T

    # Make filter matrix F
    f_max = settings.f_max
    F = numpy.zeros((s, (2 * f_max) + 1))

    for i in range(0, f_max + 1):
        if i == 0:
            lp_filter_cos_i = numpy.cos(i * omega * t)
            F[:, i] = lp_filter_cos_i
        else:
            lp_filter_cos_i = numpy.cos(i * omega * t)
            lp_filter_sin_i = numpy.sin(i * omega * t)
            F[:, 2 * i - 1] = lp_filter_cos_i
            F[:, 2 * i] = lp_filter_sin_i
    plot(settings, F, "")
    return F


def get_F_sv_t_range(settings):

    fn = "%s/ts_mirrored.jbl" % settings.results_path  # ts_meas,.jbl
    ts_meas = joblib.load(fn)

    S = ts_meas.shape[0]
    q = settings.q
    s = S - q + 1

    ts_svs = []
    for i in range(s):
        t_sv = numpy.average(ts_meas[i : i + q])
        ts_svs.append(t_sv)

    ts_svs = numpy.asarray(ts_svs)
    joblib.dump(ts_svs, "%s/ts_svs.jbl" % settings.results_path)

    print("ts_meas:", ts_meas.shape)
    print("start: ", ts_meas[0])
    print("end: ", ts_meas[-1])
    print("ts_svs:", ts_svs.shape)
    print("start: ", ts_svs[0])
    print("end: ", ts_svs[-1])

    T = ts_svs[-1] - ts_svs[0]
    omega = 2 * numpy.pi / T
    print("T:", T)

    # Make filter matrix F
    f_max = settings.f_max
    F = numpy.zeros((s, (2 * f_max) + 1))

    for i in range(0, f_max + 1):
        if i == 0:
            lp_filter_cos_i = numpy.cos(i * omega * ts_svs)
            F[:, i] = lp_filter_cos_i
        else:
            lp_filter_cos_i = numpy.cos(i * omega * ts_svs)
            lp_filter_sin_i = numpy.sin(i * omega * ts_svs)
            F[:, 2 * i - 1] = lp_filter_cos_i
            F[:, 2 * i] = lp_filter_sin_i
    plot_t_sv_range(settings, F, "")
    return F


def normalise(v):
    mod_sq = numpy.inner(v, v)
    v = v / numpy.sqrt(mod_sq)
    return v


def on(settings, Z):
    Z[:, 0] = normalise(Z[:, 0])
    # print 'Normalisation: %0.2f'%numpy.inner(Z[:,0], Z[:,0])

    for i in range(1, Z.shape[1]):
        # print 'i =', i
        for j in range(0, i):
            Z[:, i] = Z[:, i] - numpy.inner(Z[:, i], Z[:, j]) * Z[:, j]
        Z[:, i] = normalise(Z[:, i])

    plot_t_sv_range(settings, Z, "_o_n")
    return Z


def on_modified(settings, Z):
    Z[:, 0] = normalise(Z[:, 0])
    print("Normalisation: %0.2f" % numpy.inner(Z[:, 0], Z[:, 0]))

    for start in range(1, Z.shape[1]):
        print("start =", start)

        for j in range(start, Z.shape[1]):
            Z[:, j] = Z[:, j] - numpy.inner(Z[:, j], Z[:, start - 1]) * Z[:, start - 1]
        Z[:, start] = normalise(Z[:, start])
        print("Normalisation: %0.2f" % numpy.inner(Z[:, start], Z[:, start]))
        for j in range(0, start):
            print("Orthogonality: %0.2f" % numpy.inner(Z[:, start], Z[:, j]))

    plot_t_sv_range(settings, Z, "_o_n_modified")
    return Z


def on_qr(settings, Z):
    q_ortho, r = scipy.linalg.qr(Z, mode="economic")
    plot_t_sv_range(settings, q_ortho, "_qr")
    return q_ortho, r


def check_on(Z):
    diff = numpy.matmul(Z.T, Z) - numpy.eye(Z.shape[1])
    return diff


def main(settings):

    results_path = settings.results_path

    F = get_F(settings)
    Q, R = on_qr(settings, F)

    d = check_on(Q)
    print(numpy.amax(abs(d)))

    joblib.dump(Q, "%s/F_on_qr.jbl" % results_path)


def check(settings):
    results_path = settings.results_path
    Q = joblib.load("%s/F_on_qr.jbl" % results_path)
    d = check_on(Q)
    print(numpy.amax(abs(d)))
    plot(settings, Q, "_qr_check")
