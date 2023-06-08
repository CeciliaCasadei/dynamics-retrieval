# -*- coding: utf-8 -*-
import math
import time

import numpy


def f_unoptimised(D_sq, settings):
    print("\n****** RUNNING sort_D_sq ******")
    b = settings.b
    S = settings.S
    q = settings.q

    starttime = time.time()

    print("D_sq: ", D_sq.shape, D_sq.dtype)

    diff = D_sq - D_sq.T
    print("D_sq is symmetric:")
    print(numpy.amin(diff), numpy.amax(diff))

    print("Extract velocities.")
    v = numpy.sqrt(numpy.diag(D_sq, 1))

    # if D_sq.shape[0] == S - q + 1:
    #     print 'Remove first sample.'
    #     D_sq = D_sq[1:, 1:]
    print("D_sq: ", D_sq.shape, D_sq.dtype)

    print("Sqrt of D_sq.")
    D_all = numpy.sqrt(D_sq)

    print("Sorting: ")
    idxs = numpy.argsort(D_all, axis=1)
    D_all_sorted = numpy.sort(D_all, axis=1)

    D = D_all_sorted[:, 0:b]
    N = idxs[:, 0:b]

    D[:, 0] = 0

    print("It took: ", time.time() - starttime)

    return D, N, v


def f(D, b):  # It is D_sq
    print("\n****** RUNNING sort_D_sq ******")

    starttime = time.time()

    print("D_sq: ", D.shape, D.dtype)

    #    diff = D_sq - D_sq.T
    #    print 'D_sq is symmetric:'
    #    print numpy.amin(diff), numpy.amax(diff)
    #
    #    print 'Extract velocities.'
    #    v = numpy.sqrt(numpy.diag(D_sq, 1))

    # print 'Remove first sample.'
    # D = D[1:, 1:]
    print("D_sq: ", D.shape, D.dtype)

    print("Sqrt of D_sq.")
    D = numpy.sqrt(D)

    print("Sorting: ")
    N = numpy.argsort(D, axis=1)
    N = N[:, 0:b]

    D = numpy.sort(D, axis=1)
    D = D[:, 0:b]

    D[:, 0] = 0

    print("It took: ", time.time() - starttime)

    return D, N  # , v


def f_opt(D, settings):  # It is D_sq
    print("\n****** RUNNING sort_D_sq ******")

    starttime = time.time()
    b = settings.b
    S = settings.S
    q = settings.q
    datatype = settings.datatype

    print("D_sq: ", D.shape, D.dtype)

    # if D.shape[0] == S - q + 1:
    #     print 'Remove first sample.'
    #     D = D[1:, 1:]

    print("D_sq: ", D.shape, D.dtype)

    print(numpy.amax(D), numpy.amin(D))
    D[D < 0] = 0

    print("Sqrt of D_sq.")
    D = numpy.sqrt(D)

    D_sorted = numpy.zeros((D.shape[0], b), dtype=datatype)
    N_sorted = numpy.zeros((D.shape[0], b), dtype=numpy.uint32)

    print("Sorting: ")
    n_chuncks = 100
    step = int(math.floor(D.shape[0] / n_chuncks))
    print("Step: ", step)
    for i in range(n_chuncks + 1):
        start = i * step
        stop = (i + 1) * step
        if stop > D.shape[0]:
            stop = D.shape[0]
        print("Iter: ", i, start, stop)

        D_chunck = D[start:stop, :]

        N_chunck = numpy.argsort(D_chunck, axis=1)
        N_chunck = N_chunck[:, 0:b]

        D_chunck = numpy.sort(D_chunck, axis=1)
        D_chunck = D_chunck[:, 0:b]

        N_sorted[start:stop, :] = N_chunck
        D_sorted[start:stop, :] = D_chunck

    D_sorted[:, 0] = 0

    print("It took: ", time.time() - starttime)

    return D_sorted, N_sorted
