# -*- coding: utf-8 -*-
import joblib
import numpy
import numpy.linalg
import scipy.sparse.linalg
from scipy import sparse

import settings_rho_light as settings


def eigendecompose_P(P):
    evals, evecs = numpy.linalg.eig(P)
    return evals, evecs


def eigendecompose_P_sym(P_sym):
    evals, evecs = numpy.linalg.eigh(P_sym)
    return evals, evecs


def eigendecompose_P_sym_ARPACK(P_sym, l):
    
    ### NEW!!! ###
    P_sym_sparse = sparse.csr_matrix(P_sym)
    ###
    
    #evals, evecs = scipy.sparse.linalg.eigsh(P_sym_sparse, k=s-1)
    evals, evecs = scipy.sparse.linalg.eigsh(P_sym_sparse, k=l, which='LM')
    return evals, evecs


def check_ev(P, evecs, evals):
    for i in range(20):
        v = evecs[:,i]
        dot = numpy.dot(P,v)
        diff = dot - evals[i]*v
        print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))


def sort(evecs, evals):
    sort_idxs = numpy.argsort(evals)[::-1]
    evals_sorted = evals[sort_idxs]
    evecs_sorted = evecs[:,sort_idxs]
    return evecs_sorted, evals_sorted
    
    
if __name__ == '__main__':

    label = '_sym_ARPACK'
    results_path = settings.results_path
    l = settings.l

    #P = joblib.load('%s/P.jbl'%results_path)
    P = joblib.load('%s/P_sym.jbl'%results_path)
    
    print 'NaN values: ', numpy.isnan(P).any()
    s = P.shape[0]

#    print 'Verify P row normalization'
#    row_sum = numpy.sum(P, axis = 1)
#    diff = row_sum - numpy.ones((s,))
#    print numpy.amax(diff), numpy.amin(diff), '\n'

    print 'Eigendecompose'    
    #evals, evecs = eigendecompose_P(P)
    #evals, evecs = eigendecompose_P_sym(P)
    evals, evecs = eigendecompose_P_sym_ARPACK(P, l)
    print 'Done'

#    print 'Saving'
#    joblib.dump(evals, '%s/P%s_evals.jbl'%(results_path, label))
#    joblib.dump(evecs, '%s/P%s_evecs.jbl'%(results_path, label))

    #print 'Check eigenvalue problem'
    #check_ev(P, evecs, evals)
    
    # Sorting!
    print 'Sorting'
    evecs_sorted, evals_sorted = sort(evecs, evals)

    print 'Saving'
    joblib.dump(evals_sorted, '%s/P%s_evals_sorted.jbl'%(results_path, label))
    joblib.dump(evecs_sorted, '%s/P%s_evecs_sorted.jbl'%(results_path, label))