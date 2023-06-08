# -*- coding: utf-8 -*-
import argparse
import importlib
import time

import joblib
import numpy


def f(loop_idx, settings):

    print("loop_idx: ", loop_idx)

    q = settings.q
    datatype = settings.datatype
    results_path = settings.results_path
    step = settings.paral_step_A

    data_file = settings.data_file

    evecs = joblib.load("%s/F_on_qr.jbl" % (results_path))
    print("evecs (s, s): ", evecs.shape)
    f_max = settings.f_max_considered
    Phi = evecs[:, 0 : 2 * f_max + 1]

    try:
        T_sparse = joblib.load(data_file)
        x = T_sparse[:, :].todense()
        print("x (sparse -> dense): ", x.shape, x.dtype)
        x = numpy.asarray(x, dtype=datatype)
        print("x (m, S): ", x.shape, x.dtype)
    except:
        x = joblib.load(data_file)
        print("x (dense): ", x.shape, x.dtype)

    m = x.shape[0]
    S = x.shape[1]
    # s = S-q # paper version
    s = S - q + 1
    n = m * q

    print("Calculate A, index: ", loop_idx)

    q_start = loop_idx * step
    q_end = q_start + step
    if q_end > q:
        q_end = q
    print("q_start: ", q_start, "q_end: ", q_end)

    A = numpy.zeros((n, 2 * f_max + 1), dtype=datatype)

    starttime = time.time()
    for i in range(q_start, q_end):
        if i % 100 == 0:
            print(i)
        # A[i*m : (i+1)*m, :] = numpy.matmul(x[:, q-i : q-i+s], Phi) # paper version
        A[i * m : (i + 1) * m, :] = numpy.matmul(x[:, q - 1 - i : q - 1 - i + s], Phi)

    print("Time: ", time.time() - starttime)

    print("A: ", A.shape)
    joblib.dump(A, "%s/A_chunck_idx_%d.jbl" % (results_path, loop_idx))
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