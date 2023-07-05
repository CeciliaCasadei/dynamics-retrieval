# -*- coding: utf-8 -*-
import joblib
import numpy
import numpy.linalg
import scipy.sparse.linalg
from scipy import sparse


def eigendecompose_P(P):
    evals, evecs = numpy.linalg.eig(P)
    return evals, evecs


def eigendecompose_P_sym(P_sym):
    evals, evecs = numpy.linalg.eigh(P_sym)
    return evals, evecs


def eigendecompose_P_sym_ARPACK(P_sym, l):

    ### NEW!!! ###
    P_sym = sparse.csr_matrix(P_sym)
    print("P_sym", P_sym.shape, P_sym.dtype)
    print("N. non-zero:", P_sym.count_nonzero())
    ###

    # evals, evecs = scipy.sparse.linalg.eigsh(P_sym_sparse, k=s-1)
    evals, evecs = scipy.sparse.linalg.eigsh(P_sym, k=l, which="LM")
    return evals, evecs


def eigendecompose_P_ARPACK(P, l):

    ### NEW!!! ###
    P = sparse.csr_matrix(P)
    print("P", P.shape, P.dtype)
    print("N. non-zero:", P.count_nonzero())
    ###

    # evals, evecs = scipy.sparse.linalg.eigsh(P_sym_sparse, k=s-1)
    evals, evecs = scipy.sparse.linalg.eigs(P, k=l, which="LM")
    return evals, evecs


def check_ev(P, evecs, evals):
    for i in range(20):
        v = evecs[:, i]
        dot = numpy.dot(P, v)
        diff = dot - evals[i] * v
        print("%.4f  %.4f" % (numpy.amax(diff), numpy.amin(diff)))


def sort(evecs, evals):
    sort_idxs = numpy.argsort(evals)[::-1]
    evals_sorted = evals[sort_idxs]
    evecs_sorted = evecs[:, sort_idxs]
    return evecs_sorted, evals_sorted


def main(settings):

    # label = settings.eigenlabel
    results_path = settings.results_path
    l = settings.l

    P = joblib.load("%s/W_tilde.jbl" % results_path)
    # P = joblib.load('%s/P_sym.jbl'%results_path)

    print("NaN values: ", numpy.isnan(P).any())
    # s = P.shape[0]

    # print 'Verify P row normalization'
    # row_sum = numpy.sum(P, axis = 1)
    # diff = row_sum - numpy.ones((s,))
    # print numpy.amax(diff), numpy.amin(diff), '\n'

    print("Eigendecompose")
    # evals, evecs = eigendecompose_P(P)
    # evals, evecs = eigendecompose_P_sym(P)
    evals, evecs = eigendecompose_P_sym_ARPACK(P, l)
    # evals, evecs = eigendecompose_P_ARPACK(P, l)
    print("Done")

    #    print 'Saving'
    #    joblib.dump(evals, '%s/P%s_evals.jbl'%(results_path, label))
    #    joblib.dump(evecs, '%s/P%s_evecs.jbl'%(results_path, label))

    # print 'Check eigenvalue problem'
    # check_ev(P, evecs, evals)

    # Sorting!
    print("Sorting")
    evecs_sorted, evals_sorted = sort(evecs, evals)

    print("Saving")
    joblib.dump(evals_sorted, "%s/evals_sorted.jbl" % (results_path))
    joblib.dump(evecs_sorted, "%s/evecs_sorted.jbl" % (results_path))