#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:40:20 2023

@author: casadei_c
"""
import math

import joblib
import numpy


def make_Ui_subf(settings, i, U_temp):
    results_path = settings.results_path
    f_max = settings.f_max
    A_i = joblib.load("%s/A_%d.jbl" % (results_path, i))[:, 0 : 2 * f_max + 1]
    print("A_i:", A_i.shape)
    U_i = numpy.matmul(A_i, U_temp)
    # joblib.dump(U_i[:,0:20], '%s/U_%d.jbl'%(results_path, i))
    for j in range(0, 1):  # 2*f_max+1):
        joblib.dump(U_i[:, j], "%s/u_%d_chunck_%d.jbl" % (results_path, j, i))


def make_Ui_f(settings, n_chuncks):
    results_path = settings.results_path
    S = joblib.load("%s/S.jbl" % results_path)
    V = joblib.load("%s/V.jbl" % results_path)
    U_temp = numpy.matmul(V, numpy.diag(1.0 / S))[:, 0:20]

    for i in range(0, n_chuncks):
        make_Ui_subf(settings, i, U_temp)


def make_uj(settings, n_chuncks):
    results_path = settings.results_path
    datatype = settings.datatype
    m = settings.m
    q = settings.q
    f_max = settings.f_max

    step = int(math.ceil(float(m * q) / n_chuncks))

    for j in range(0, 2 * f_max + 1):
        u_j = numpy.zeros(shape=(m * q,), dtype=datatype)
        for i in range(0, n_chuncks):
            start = i * step
            end = min([(i + 1) * step, m * q])
            print(start, end)
            u_j_chunck_i = joblib.load("%s/u_%d_chunck_%d.jbl" % (results_path, j, i))
            u_j[
                start:end,
            ] = u_j_chunck_i

        print("Saving.")
        joblib.dump(u_j, "%s/u_%d.jbl" % (results_path, j))


def make_U(settings, n_chuncks):
    results_path = settings.results_path
    datatype = settings.datatype
    m = settings.m
    q = settings.q

    U = numpy.zeros(shape=(m * q, 20), dtype=datatype)
    step = int(math.ceil(float(m * q) / n_chuncks))
    for i in range(0, n_chuncks):
        start = i * step
        end = min([(i + 1) * step, m * q])
        U_i = joblib.load("%s/U_%d.jbl" % (results_path, i))
        U[start:end, :] = U_i

    print("Saving.")
    joblib.dump(U, "%s/U.jbl" % results_path)
