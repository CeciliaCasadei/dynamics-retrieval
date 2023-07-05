# -*- coding: utf-8 -*-
import time

import joblib
import numpy


def main(settings):

    print("\n****** RUNNING merge_A ******")
    results_path = settings.results_path
    datatype = settings.datatype
    n_workers_A = settings.n_workers_A

    starttime = time.time()

    fn = "%s/A_chunck_idx_0.jbl" % (results_path)
    A = joblib.load(fn)
    
    for i in range(1, n_workers_A):
        print(i)
        fn = "%s/A_chunck_idx_%d.jbl" % (results_path, i)
        print(fn)
        temp = joblib.load(fn)
        A += temp
    print('Done.')
     
    print('A: ', A.shape, A.dtype)
    print('Saving.')
    joblib.dump(A, '%s/A_parallel.jbl'%results_path)
    print('Done.')
    print('It took: ', time.time() - starttime)