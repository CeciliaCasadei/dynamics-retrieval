# -*- coding: utf-8 -*-
import argparse
import importlib
import time

import joblib
import numpy


def f(loop_idx, settings):

    print("PARALLEL JOB loop_idx: ", loop_idx)

    q = settings.q
    datatype = settings.datatype
    results_path = settings.results_path
    print("q: ", q)

    step = settings.paral_step

    n_d_sq = joblib.load("%s/n_dsq_elements.jbl" % results_path)
    print("n_d_sq: ", n_d_sq.shape, n_d_sq.dtype)

    S = n_d_sq.shape[0]
    s = S - q

    print(S, " samples")

    print("Calculate N_D_sq, index: ", loop_idx)

    q_start = loop_idx * step
    q_end = q_start + step
    if q_end > q:
        q_end = q
    print("q_start: ", q_start, "q_end: ", q_end)

    N_D_sq = numpy.zeros((s + 1, s + 1), dtype=datatype)  # will need to be s,s
    starttime = time.time()

    for i in range(q_start, q_end):
        if i % 100 == 0:
            print(i, "/", q)
        N_D_sq += n_d_sq[i : i + s + 1, i : i + s + 1]

    print("Time: ", time.time() - starttime)

    print("N_D_sq: ", N_D_sq.shape)
    joblib.dump(N_D_sq, "%s/N_D_sq_loop_idx_%d.jbl" % (results_path, loop_idx))

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
    print(("Concatenation n: %d" % settings.q))

    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()
