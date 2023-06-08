# -*- coding: utf-8 -*-
import h5py
import joblib
import matplotlib.pyplot
import numpy
import scipy.io
from scipy import sparse


def main(settings):

    datatype = settings.datatype
    label = settings.label
    folder = settings.results_path
    print(datatype, label)
    print(folder)

    T = joblib.load("%s/T_sparse_%s.jbl" % (folder, label))
    M = joblib.load("%s/M_sparse_%s.jbl" % (folder, label))
    print(sparse.issparse(T), T.dtype, T.shape)
    print(sparse.issparse(M), M.dtype, M.shape)

    ts = joblib.load("%s/t_%s.jbl" % (folder, label))
    print(ts.shape)

    matplotlib.pyplot.hist(ts, bins=40, histtype="step", color="c")
    matplotlib.pyplot.savefig("%s/hist.png" % folder)
    matplotlib.pyplot.close()

    idxs = numpy.argwhere(ts > -150)
    idxs = idxs[:, 0]
    print(idxs.shape, "elements above -150fs")
    # print idxs[0:10], idxs[-5:]

    ts_tmp = ts[idxs, :]
    print(ts_tmp.shape)

    T_tmp = T[:, idxs]
    M_tmp = M[:, idxs]
    print(sparse.issparse(T_tmp), T_tmp.dtype, T_tmp.shape)
    print(sparse.issparse(M_tmp), M_tmp.dtype, M_tmp.shape)

    idxs = numpy.argwhere(ts_tmp < 1100)
    idxs = idxs[:, 0]
    print(idxs.shape, "elements below 1100fs")
    # print idxs[0:10], idxs[-5:]

    ts_sel = ts_tmp[idxs, :]
    print(ts_sel.shape)

    T_sel = T_tmp[:, idxs]
    M_sel = M_tmp[:, idxs]
    print(sparse.issparse(T_sel), T_sel.dtype, T_sel.shape)
    print(sparse.issparse(M_sel), M_sel.dtype, M_sel.shape)

    matplotlib.pyplot.hist(ts_sel, bins=40, histtype="step", color="c")
    matplotlib.pyplot.savefig("%s/hist_selected.png" % folder)

    joblib.dump(T_sel, "%s/T_sel_sparse_%s.jbl" % (folder, label))
    joblib.dump(M_sel, "%s/M_sel_sparse_%s.jbl" % (folder, label))
    joblib.dump(ts_sel, "%s/ts_sel_%s.jbl" % (folder, label))
