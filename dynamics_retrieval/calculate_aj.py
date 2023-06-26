# -*- coding: utf-8 -*-
import argparse
import importlib
import time

import joblib
import numpy


def f(loop_idx, settings):
    "Calculate part of matrix A = X@phi"

    print("node_idx: ", loop_idx)

    q = settings.q
    datatype = settings.datatype
    results_path = settings.results_path
    data_file = settings.data_file

    Q = joblib.load("%s/F_on.jbl" % (results_path))
    print("Q: ", Q.shape, Q.dtype)
    Qj = Q[:, loop_idx]
    print("Qj: ", Qj.shape, Qj.dtype)

    try:
        T_sparse = joblib.load(data_file)
        x = T_sparse[:, :].todense()
    except:
        x = joblib.load(data_file)
    print("x: ", x.shape, x.dtype)
    x = numpy.asarray(x, dtype=datatype)
    print("x (m, S): ", x.shape, x.dtype)

    m = x.shape[0]
    S = x.shape[1]
    s = S - q + 1
    n = m * q

    print("Calculate aj, index: ", loop_idx)

    q_start = 0
    q_end = q

    print("q_start: ", q_start, "q_end: ", q_end)

    aj = numpy.zeros((n,), dtype=datatype)

    print("Qj (s, 1): ", Qj.shape)
    starttime = time.time()
    for i in range(q_start, q_end):
        if i % 100 == 0:
            print(i)
        aj[i * m : (i + 1) * m] = numpy.matmul(x[:, q - 1 - i : q - 1 - i + s], Qj)

    print("Time: ", time.time() - starttime)

    print("aj: ", aj.shape)
    joblib.dump(aj, "%s/aj/a_%d.jbl" % (results_path, loop_idx))
    print("Time: ", time.time() - starttime)


def main(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("worker_ID", help="Worker ID")
    parser.add_argument("settings", help="settings file")
    args = parser.parse_args(args)

    print("Worker ID: ", args.worker_ID)

    # Dynamic import based on the command line argument
    settings = importlib.import_module(args.settings)

    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()
