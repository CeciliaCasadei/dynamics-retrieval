# -*- coding: utf-8 -*-
import joblib
import matplotlib
import numpy

matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot
from scipy import sparse


def main(settings):
    results_path = settings.results_path

    D = joblib.load("%s/D.jbl" % results_path)
    print("D", D.shape, D.dtype)
    D_sq = D * D
    print("D_sq", D_sq.shape, D_sq.dtype)
    D_sq = D_sq.flatten()
    print("D_sq", D_sq.shape, D_sq.dtype)
    D_sq = list(D_sq)
    print("D_sq", len(D_sq))
    D_sq = [D_sq[i] for i in range(len(D_sq)) if not numpy.isnan(D_sq[i])]
    D_sq = numpy.asarray(D_sq)
    print("D_sq", D_sq.shape, D_sq.dtype)

    log_epsilon_list = numpy.linspace(-3, 6, num=100)

    epsilons = []
    sigmas = []
    values = []
    for log_epsilon in log_epsilon_list:
        print(log_epsilon)
        epsilon = 10 ** log_epsilon
        sigma = numpy.sqrt(2 * epsilon)

        exponent = D_sq / (2 * epsilon)
        W = numpy.exp(-exponent)
        value = W.sum()
        print(value)

        epsilons.append(epsilon)
        sigmas.append(sigma)
        values.append(value)

    values = numpy.asarray(values)
    log_values = numpy.log10(values)

    max_val = numpy.amax(log_values)
    min_val = numpy.amin(log_values)
    opt_val = (max_val + min_val) / 2

    diff = abs(log_values - opt_val)
    idx = numpy.argmin(diff)
    print(idx)

    epsilon_opt = epsilons[idx]
    sigma_opt = sigmas[idx]
    log_epsilon_opt = log_epsilon_list[idx]
    print("Epsilon_opt = (sigma_opt**2) / 2: ", epsilon_opt)
    print("Log(epsilon_opt): ", log_epsilon_opt)
    print("Sigma_opt: ", sigma_opt)

    matplotlib.pyplot.plot(log_epsilon_list, log_values)
    matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
    matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
    # matplotlib.pyplot.axvline(x=log_epsilon_opt, ymin=0, ymax=1)
    matplotlib.pyplot.savefig("%s/bilog_2.png" % (results_path))
    matplotlib.pyplot.close()
