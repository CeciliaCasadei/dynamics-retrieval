# -*- coding: utf-8 -*-
"""Reconstruct vectors from the filtered supervector matrix

This module is optimized for small p (eg p=0). Use reconstruct_parallel when averaging
all copies of a vector (p = (q - 1)/2)
"""

import time

import joblib
import numpy


def f(settings):
    results_path = settings.results_path
    q = settings.q
    p = settings.p  # Number of copies to be averaged.
    print(results_path)

    datatype = settings.datatype

    # TODO Assumes full U was saved. Comment out for column-wise U
    U = joblib.load('%s/U.jbl'%results_path)
    print('U:', U.shape)
    S = joblib.load("%s/S.jbl" % results_path)
    print("S: ", S.shape)
    VT_final = joblib.load("%s/VT_final.jbl" % results_path)
    print("VT_final: ", VT_final.shape)

    # VT_final[k,:] is the chronogram for k
    # TODO Assumes that time was mirrored, so it needs to be cropped 
    #VT_final = VT_final[:, 0 : settings.S - settings.q + 1]
    VT_final = VT_final[:, 0 : int(float(2*settings.S - settings.q + 1)/2)]
    print("VT_final: ", VT_final.shape)
    print(settings.modes_to_reconstruct)

    s = VT_final.shape[1]
    print("s: ", s)
    m = settings.m
    print("m: ", m)

    for k in settings.modes_to_reconstruct:
        print("Mode: ", k)

        # TODO Choice here depends on whether U was created chunk-wise or column-wise
        u_k = U[:,k]
        #u_k = joblib.load("%s/uj/u_%d.jbl" % (results_path, k))  # Load single column
        s_k = S[k]
        v_k = VT_final[k, :]
        print(u_k.shape, v_k.shape)

        block_idx_center = (q - 1) / 2
        print("block_idx_center:", block_idx_center)
        block_idxs = list(range(int(block_idx_center) - p, int(block_idx_center) + p + 1))
        print("block idxs: ", block_idxs)

        starttime = time.time()

        ncopies = 2 * p + 1
        x_r = numpy.zeros((m, s - ncopies + 1), dtype=datatype)
        print("x_r:", x_r.shape)

        for i in range(-p, p + 1):

            block_idx = block_idx_center + i
            print(i, block_idx)
            i1 = int(block_idx * m)
            i2 = int((block_idx + 1) * m)
            print(i1, i2)
            term = s_k * numpy.outer(u_k[i1:i2], v_k)
            print("term: ", term.shape)
            print("starting column:", i + p)
            print("ending column (included):", i + p + (s - ncopies + 1) - 1)

            x_r += term[:, i + p : i + p + (s - ncopies + 1)]
        x_r = x_r / ncopies

        print("Time: ", time.time() - starttime)
        joblib.dump(x_r, "%s/movie_p_%d_mode_%d.jbl" % (results_path, p, k))


def f_ts(settings):
    S = settings.S
    q = settings.q
    #s = S - q + 1
    s = int(float(2*S - q + 1)/2)

    ts_svs = joblib.load("%s/ts_svs.jbl" % settings.results_path)
    print("ts_svs:", ts_svs.shape, ts_svs[0:5], "..", ts_svs[-2:])

    p = settings.p

    block_idx_center = (q - 1) / 2
    print("block_idx_center:", block_idx_center)
    block_idxs = list(range(int(block_idx_center) - p, int(block_idx_center) + p + 1))
    print(block_idxs)

    ncopies = 2 * p + 1
    t_r = numpy.zeros((s - ncopies + 1,), dtype=settings.datatype)
    print("t_r:", t_r.shape)

    for i in range(-p, p + 1):
        print(i)

        print("starting column:", i + p)
        print("ending column (included):", i + p + (s - ncopies + 1) - 1)

        t_r += ts_svs[i + p : i + p + (s - ncopies + 1)]
    t_r = t_r / ncopies
    joblib.dump(t_r, "%s/t_r_p_%d.jbl" % (settings.results_path, p))

    print("t_r:", t_r.shape, t_r[0:5], "...", t_r[-2:])
