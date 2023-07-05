#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:37:03 2023

@author: casadei_c
"""
import joblib
import matplotlib
import numpy
import settings_bR_light as settings

path = settings.results_path
n_obs = joblib.load("%s/boost_factors_light.jbl" % path)
ts = joblib.load("%s/ts_sel_light.jbl" % path)
hs = joblib.load("%s/miller_h_light.jbl" % path)
ks = joblib.load("%s/miller_k_light.jbl" % path)
ls = joblib.load("%s/miller_l_light.jbl" % path)

T_range = ts[-1] - ts[0]
print("T range [fs]:", T_range)

# Approx. max noise period
period_max = T_range / n_obs

# bR: q-vector in reciprocal space:
a = 62.36
b = 62.39
c = 111.18
q_x = hs / a
q_y = hs / (a * numpy.sqrt(3)) + 2 * ks / (b * numpy.sqrt(3))
q_z = ls / c
q_sq = q_x * q_x + q_y * q_y + q_z * q_z
q = numpy.sqrt(q_sq)
d = 1.0 / q  # A

matplotlib.pyplot.figure(figsize=(10, 10))
matplotlib.pyplot.scatter(d, n_obs, c="b", s=3, alpha=0.3)
matplotlib.pyplot.gca().set_xlabel("d [A]")
matplotlib.pyplot.gca().set_ylabel("n. of observations")
matplotlib.pyplot.gca().set_ylim([0, 1000])
matplotlib.pyplot.savefig("%s/n_obs_vs_res_zoom.png" % path, dpi=4 * 96)
matplotlib.pyplot.close()

matplotlib.pyplot.figure(figsize=(10, 10))
matplotlib.pyplot.scatter(d, numpy.log10(n_obs), c="b", s=3, alpha=0.3)
matplotlib.pyplot.gca().set_xlabel("d [A]")
matplotlib.pyplot.gca().set_ylabel("log(n. of observations)")
matplotlib.pyplot.savefig("%s/n_obs_log_vs_res.png" % path, dpi=4 * 96)
matplotlib.pyplot.close()

print(numpy.amin(d))
matplotlib.pyplot.figure(figsize=(20, 10))
matplotlib.pyplot.hist(n_obs, bins=70, color="b")
matplotlib.pyplot.gca().set_xlabel("n. of observations")
matplotlib.pyplot.gca().set_ylabel("n. of reflections")
matplotlib.pyplot.savefig("%s/n_obs_hist.png" % path)
matplotlib.pyplot.close()

matplotlib.pyplot.figure(figsize=(30, 10))
matplotlib.pyplot.hist(period_max, bins=70, color="b")
matplotlib.pyplot.gca().set_xlabel("Max sparsity-related period")
matplotlib.pyplot.gca().set_ylabel("n. of reflections")
matplotlib.pyplot.savefig("%s/T_max_hist.png" % path)
matplotlib.pyplot.close()


idx = numpy.argmax(n_obs)

M = joblib.load("%s/M_sel_sparse_light.jbl" % path)
print("M: ", M.shape)
T = joblib.load("%s/T_bst_sparse_light.jbl" % path)
print("T: ", T.shape)

bragg_i = T[idx, :]
idxs = bragg_i.nonzero()
bragg_i = bragg_i[idxs]
bragg_i = bragg_i.tolist()[0]

matplotlib.pyplot.figure(figsize=(20, 10))
matplotlib.pyplot.hist(bragg_i, bins=50, color="b")
matplotlib.pyplot.gca().set_xlabel("I")
matplotlib.pyplot.gca().set_ylabel("n. of obs")
matplotlib.pyplot.title(
    "(%d, %d, %d), %0.2fA, %d observations"
    % (hs[idx], ks[idx], ls[idx], d[idx], M.sum(axis=1)[idx])
)
matplotlib.pyplot.savefig("%s/bragg_%d_hist.png" % (path, idx))
matplotlib.pyplot.close()
