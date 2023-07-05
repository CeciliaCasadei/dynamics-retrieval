#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 15:43:46 2022

@author: casadei_c
"""
import time

import joblib
import numpy


def main_euclidean_nn(settings):
    from . import calculate_distances_utilities

    calculate_distances_utilities.sort_D_sq(settings)


def main_time_nn_1(settings):

    print("\n****** RUNNING ******")

    starttime = time.time()
    b = settings.b
    S = settings.S
    q = settings.q
    datatype = settings.datatype

    D = joblib.load("%s/D_sq_normalised.jbl" % settings.results_path)
    print("D_sq: ", D.shape, D.dtype)

    if D.shape[0] == S - q + 1:
        print("Remove first sample.")
        D = D[1:, 1:]
    print("D_sq: ", D.shape, D.dtype)

    b_over_two = int(b / 2)
    print(b_over_two)

    print(numpy.amax(D), numpy.amin(D))
    D[D < 0] = 0
    print("Sqrt of D_sq.")
    D = numpy.sqrt(D)

    D_nn = numpy.zeros((D.shape[0], b), dtype=datatype)
    N_nn = numpy.zeros((D.shape[0], b), dtype=numpy.uint32)

    print(D_nn.shape, N_nn.shape)
    for i in range(S - q):
        # print i
        if i < b_over_two:
            left_idx = 0
            right_idx = b

        elif i > S - q - b_over_two:
            left_idx = S - q - b
            right_idx = S - q

        else:
            left_idx = i - b_over_two
            right_idx = i + b_over_two
        idxs = list(range(left_idx, right_idx))
        if len(idxs) != b:
            print("Issue!")
        N_nn[i, :] = idxs
        D_nn[i, :] = D[i, idxs]
    print("Time: ", time.time() - starttime, "s")
    print("Saving")
    joblib.dump(D_nn, "%s/D.jbl" % settings.results_path)
    joblib.dump(N_nn, "%s/N.jbl" % settings.results_path)


def main_time_nn_2(settings):

    print("\n****** RUNNING ******")

    starttime = time.time()
    b = settings.b
    S = settings.S
    q = settings.q
    datatype = settings.datatype

    D = joblib.load("%s/D_sq_normalised.jbl" % settings.results_path)
    print("D_sq: ", D.shape, D.dtype)

    if D.shape[0] == S - q + 1:
        print("Remove first sample.")
        D = D[1:, 1:]
    print("D_sq: ", D.shape, D.dtype)

    b_over_two = int(b / 2)
    print(b_over_two)

    print(numpy.amax(D), numpy.amin(D))
    D[D < 0] = 0
    print("Sqrt of D_sq.")
    D = numpy.sqrt(D)

    D_nn = numpy.zeros((D.shape[0], b), dtype=datatype)
    N_nn = numpy.zeros((D.shape[0], b), dtype=numpy.uint32)
    D_nn[:] = numpy.nan
    N_nn[:] = numpy.nan

    print(D_nn.shape, N_nn.shape)
    for i in range(S - q):

        left_idx = max([0, i - b_over_two])
        right_idx = min([i + b_over_two, S - q])

        idxs = list(range(left_idx, right_idx))
        print(len(idxs))
        # if len(idxs)!=b:
        #     print 'Issue!'
        N_nn[i, 0 : len(idxs)] = idxs
        D_nn[i, 0 : len(idxs)] = D[i, idxs]
    print("Time: ", time.time() - starttime, "s")
    print("Saving")
    joblib.dump(D_nn, "%s/D.jbl" % settings.results_path)
    joblib.dump(N_nn, "%s/N.jbl" % settings.results_path)
