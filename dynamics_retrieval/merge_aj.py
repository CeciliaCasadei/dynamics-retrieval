#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:50:41 2023

@author: casadei_c
"""
import joblib
import numpy


def main(settings):
    results_path = settings.results_path
    datatype = settings.datatype
    m = settings.m
    q = settings.q
    f_max = settings.f_max

    A = numpy.zeros(shape=(m * q, 2 * f_max + 1), dtype=datatype)
    for i in range(0, 2 * f_max + 1):
        print(i)
        fn = "%s/aj/a_%d.jbl" % (results_path, i)
        aj = joblib.load(fn)
        A[:, i] = aj

    print("A: ", A.shape, A.dtype)
    joblib.dump(A, "%s/A_parallel.jbl" % results_path)
