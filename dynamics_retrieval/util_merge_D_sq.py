# -*- coding: utf-8 -*-
import time

import joblib
import numpy


def f(settings):
    print("\n****** RUNNING merge_D_sq ******")
    results_path = settings.results_path
    datatype = settings.datatype
    nproc = settings.n_workers
    n = settings.S - settings.q + 1

    starttime = time.time()

    print("Merge D_sq")
    D_sq = numpy.zeros((n, n), dtype=datatype)
    for i in range(nproc):
        print(i)
        fn = "%s/D_sq_loop_idx_%d.jbl" % (results_path, i)
        print(fn)
        temp = joblib.load(fn)
        D_sq += temp
    print("Done.")

    print("Saving.")
    joblib.dump(D_sq, "%s/D_sq_parallel.jbl" % results_path)
    print("Done.")
    print("It took: ", time.time() - starttime)


def f_lp_filter(settings):
    print("\n****** RUNNING merge_D_sq ******")
    results_path = settings.results_path
    datatype = settings.datatype
    nproc = settings.n_workers_lp_filter_Dsq
    n = settings.S - settings.q
    f_max = settings.f_max_considered

    starttime = time.time()

    print("Merge D_sq")
    D_sq = numpy.zeros((n, n), dtype=datatype)
    for i in range(nproc):
        print(i)
        fn = "%s/D_sq_lp_filtered_fmax_%d_chunck_%d.jbl" % (results_path, f_max, i)
        print(fn)
        temp = joblib.load(fn)
        D_sq += temp
    print("Done.")

    print("Saving.")
    joblib.dump(D_sq, "%s/D_sq_lp_filtered_fmax_%d.jbl" % (results_path, f_max))
    print("Done.")
    print("It took: ", time.time() - starttime)


def f_N_D_sq_elements(settings):
    print("\n****** RUNNING merge_N_D_sq_elements ******")
    results_path = settings.results_path
    datatype = settings.datatype
    nproc = settings.n_workers
    n = settings.S - settings.q + 1

    starttime = time.time()

    print("Merge N_D_sq_elements")
    N_D_sq = numpy.zeros((n, n), dtype=datatype)
    for i in range(nproc):
        print(i)
        fn = "%s/N_D_sq_loop_idx_%d.jbl" % (results_path, i)
        print(fn)
        temp = joblib.load(fn)
        N_D_sq += temp
    print("Done.")

    print("Saving.")
    joblib.dump(N_D_sq, "%s/N_D_sq_parallel.jbl" % results_path)
    print("Done.")
    print("It took: ", time.time() - starttime)
