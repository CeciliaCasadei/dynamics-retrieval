# -*- coding: utf-8 -*-
import argparse
import importlib
import time

import joblib
import numpy


def f(loop_idx, settings):
    results_path = settings.results_path
    # q = settings.q
    ncopies = settings.ncopies
    step = settings.paral_step_reconstruction
    datatype = settings.datatype
    modes = settings.modes_to_reconstruct
    U = joblib.load("%s/U.jbl" % results_path)
    U = U[:, 0:20]
    S = joblib.load("%s/S.jbl" % results_path)
    VT_final = joblib.load("%s/VT_final.jbl" % results_path)

    s = VT_final.shape[1]
    VT_final = VT_final.T
    m = settings.m

    for k in modes:
        print("Mode: ", k)
        print("loop_idx: ", loop_idx)
        # u_k = joblib.load('%s/uj/u_%d.jbl'%(results_path, k))

        start_j = loop_idx * step
        end_j = (loop_idx + 1) * step

        if end_j > s - ncopies + 1:
            end_j = s - ncopies + 1
        print("Loop: ", start_j, end_j)

        starttime = time.time()
        x_r = numpy.zeros((m, s - ncopies + 1), dtype=datatype)
        for j in range(start_j, end_j):
            if j % 1000 == 0:
                print(j)
            for i in range(ncopies):
                i1 = i * m
                i2 = (i + 1) * m
                x_r[:, j] = x_r[:, j] + U[i1:i2, k] * S[k] * VT_final[j + i, k]
                # x_r[:,j] = x_r[:,j] + u_k[i1:i2] * S[k] * VT_final[j+i,k]
            x_r[:, j] = x_r[:, j] / ncopies
        joblib.dump(x_r, "%s/movie_mode_%d_chunck_%d.jbl" % (results_path, k, loop_idx))
        print("x_r:", x_r.shape)
        print("Time: ", time.time() - starttime)


def main(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("worker_ID", help="Worker ID")
    parser.add_argument("settings", help="settings file")
    args = parser.parse_args(args)

    print("Worker ID: ", args.worker_ID)

    # Dynamic import based on the command line argument
    settings = importlib.import_module(args.settings)

    # print("Label: %s"%settings.label)

    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()
