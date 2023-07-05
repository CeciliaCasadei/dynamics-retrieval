#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:20:50 2023

@author: casadei_c
"""
import math

import joblib
import numpy


def make_Ai_f(settings, n_chuncks):
    results_path = settings.results_path
    datatype = settings.datatype
    m = settings.m
    q = settings.q
    f_max = settings.f_max

    step = int(math.ceil(float(m * q) / n_chuncks))
    for i in range(0, n_chuncks):
        start = i * step
        end = min([(i + 1) * step, m * q])
        print(start, end)
        A_i = numpy.zeros(shape=(end - start, 2 * f_max + 1), dtype=datatype)
        for j in range(0, 2 * f_max + 1):
            print(j)
            fn = "%s/aj/a_%d.jbl" % (results_path, j)
            aj = joblib.load(fn)
            A_i[:, j] = aj[start:end]
        print("i: ", A_i.shape, A_i.dtype)
        joblib.dump(A_i, "%s/A_%i.jbl" % (results_path, i))


def get_Ai_subf(settings, i, n_chuncks):
    results_path = settings.results_path
    f_max = settings.f_max
    m = settings.m
    q = settings.q

    step = int(math.ceil(float(m * q) / n_chuncks))
    start = i * step
    end = min([(i + 1) * step, m * q])
    print(start, end)
    A = joblib.load("%s/A_parallel.jbl" % results_path)[:, 0 : 2 * f_max + 1]
    print("A:", A.shape)
    joblib.dump(A[start:end, :], "%s/A_%d.jbl" % (results_path, i))


def get_Ai_f(settings, n_chuncks):
    for i in range(0, n_chuncks):
        get_Ai_subf(settings, i, n_chuncks)
