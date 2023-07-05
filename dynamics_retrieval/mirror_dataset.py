#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:42:07 2023

@author: casadei_c
"""
import joblib
import matplotlib.pyplot
import numpy
import numpy.random
from scipy import sparse


def main(settings):
    print(settings.results_path)
    T = joblib.load(
        "%s/dT_bst.jbl" % (settings.results_path)
    )

    x = T[:, :].todense()
    print("x (sparse -> dense): ", x.shape, x.dtype)
    x = numpy.asarray(x, dtype=settings.datatype)
    print("x (m, S): ", x.shape, x.dtype)
    print(x.shape)

    T_mirrored = numpy.zeros(shape=(T.shape[0], 2 * settings.S))
    print(T_mirrored.shape)

    T_mirrored[:, 0 : settings.S] = x
    T_mirrored[:, settings.S :] = numpy.fliplr(x)

    print(numpy.amax(abs(T_mirrored[:, 0] - T_mirrored[:, -1])))
    print("T_mirrored, is sparse? ", sparse.issparse(T_mirrored))
    T_mirrored_sparse = sparse.csr_matrix(T_mirrored)
    print("T_mirrored, is sparse? ", sparse.issparse(T_mirrored_sparse))

    joblib.dump(
        T_mirrored_sparse,
        "%s/dT_bst_mirrored.jbl"
        % (settings.results_path),
    )


def make_virtual_ts(settings):
    ts = joblib.load("%s/t_%s.jbl" % (settings.results_path, settings.label))
    print(ts.shape)
    ts_mirrored = numpy.zeros(shape=(2 * settings.S, 1))
    ts_mirrored[0 : settings.S, 0] = ts[:, 0]

    T = ts[-1, 0] - ts[0, 0]
    print(T)
    virtual_ts = ts[-1, 0] + T * numpy.sort(numpy.random.rand(settings.S))

    print(virtual_ts.shape)

    ts_mirrored[settings.S :, 0] = virtual_ts

    matplotlib.pyplot.scatter(list(range(2 * settings.S)), ts_mirrored)

    joblib.dump(
        ts_mirrored, "%s/ts_%s_mirrored.jbl" % (settings.results_path, settings.label)
    )
