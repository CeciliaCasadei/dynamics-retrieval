# -*- coding: utf-8 -*-
import time

import joblib
import numpy


def main(settings):
    results_path = settings.results_path
    f_max = settings.f_max
    n_vectors = 2 * f_max + 1
    ATA = numpy.zeros((n_vectors, n_vectors))
    print(ATA.shape, ATA.dtype)
    start = time.time()
    for i in range(0, n_vectors):
        print(i)
        chunck = joblib.load("%s/ATA/chunck_%d.jbl" % (results_path, i))
        ATA += chunck
    print("Time:", time.time() - start)
    joblib.dump(ATA, "%s/ATA.jbl" % (results_path))
