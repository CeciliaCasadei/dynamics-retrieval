#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:57:09 2022

@author: casadei_c
"""
import random

import matplotlib.pyplot
import numpy
import scipy.linalg


def get_F_sv_t_range(ts):

    s = ts.shape[0]
    T = ts[-1] - ts[0]
    omega = 2 * numpy.pi / T
    print("T:", T)

    # Make filter matrix F
    f_max = 100
    F = numpy.zeros((s, (2 * f_max) + 1))

    for i in range(0, f_max + 1):
        if i == 0:
            lp_filter_cos_i = numpy.cos(i * omega * ts)
            F[:, i] = lp_filter_cos_i
        else:
            lp_filter_cos_i = numpy.cos(i * omega * ts)
            lp_filter_sin_i = numpy.sin(i * omega * ts)
            F[:, 2 * i - 1] = lp_filter_cos_i
            F[:, 2 * i] = lp_filter_sin_i
    # plot_t_sv_range(settings, F, '')
    return F


def on_qr(Z):
    q_ortho, r = scipy.linalg.qr(Z, mode="economic")
    # plot_t_sv_range(settings, q_ortho, '_qr')
    return q_ortho, r


ts = numpy.asarray(list(range(0, 1000)), dtype=numpy.float64)
funcs = get_F_sv_t_range(ts)
funcs_on, r = on_qr(funcs)

T = ts[-1] - ts[0]
omega = 2 * numpy.pi / T

# a_1 = numpy.sin(100*omega*ts)
# a_2 = numpy.sin(150*omega*ts)
# a_3 = numpy.sin(70*omega*ts)

# x = 3*(1-numpy.exp(-(ts-700)/200)) #+ 2*(numpy.exp(-ts/600))#
x_dyn = 2 * numpy.sin(11 * omega * ts) + 1 * numpy.cos(56 * omega * ts)

N_observed = 50
N_unobserved = 1000 - N_observed
idxs = random.sample(list(range(0, 1000)), N_unobserved)
idxs = sorted(idxs)
x_dyn[idxs] = 0

x = 3 + x_dyn
# matplotlib.pyplot.plot(ts, x)
res_lst = []
for i in range(funcs_on.shape[1]):
    func_on = funcs_on[:, i]
    res = (x * func_on).sum()
    res_lst.append(res)

matplotlib.pyplot.plot(list(range(funcs_on.shape[1])), res_lst)

print(res_lst[0], res_lst[22], res_lst[111])
# res = numpy.inner(x, funcs_on)

# res = a_3 * x_1
# res = res[0:1000].sum()
# print res
