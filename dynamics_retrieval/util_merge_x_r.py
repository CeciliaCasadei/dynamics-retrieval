# -*- coding: utf-8 -*-
import time

import joblib


def f(settings, mode):
    print("\n****** RUNNING merge_x_r ******")
    results_path = settings.results_path
    nproc = settings.n_workers_reconstruction

    print("Mode: ", mode)
    starttime = time.time()

    fn = "%s/movie_mode_%d_chunck_0.jbl" % (results_path, mode)
    x_r = joblib.load(fn)
    print(x_r.shape, x_r.dtype)

    print("Merge x_r")
    for i in range(1, nproc):
        print(i)
        fn = "%s/movie_mode_%d_chunck_%d.jbl" % (results_path, mode, i)
        print(fn)
        temp = joblib.load(fn)
        x_r = x_r + temp
    print("Done.")

    print("Saving.")
    joblib.dump(x_r, "%s/movie_mode_%d.jbl" % (results_path, mode))
    print("Done.")
    print("It took: ", time.time() - starttime)
