# -*- coding: utf-8 -*-
import pickle
import time

import joblib
import numpy
import settings_NP as settings


def get_A(X, mu, evecs_norm, l):

    Phi = evecs_norm[:, 0:l]
    mu = numpy.diag(mu)
    X = X[:, 1:]
    print("X (mq=n, s): ", X.shape)
    A_temp = numpy.dot(mu, Phi)
    print("A_temp  (s, l): ", A_temp.shape)
    A = numpy.dot(X, A_temp)
    print("A (mq=n, l): ", A.shape)
    return A


def get_A_loop(X, mu, evecs_norm, l):

    Phi = evecs_norm[:, 0:l]
    print("mu: ", mu.shape)
    print("X (mq=n, s+1): ", X.shape)
    n = X.shape[0]
    A = numpy.zeros((n, l))
    for j in range(l):
        print(j)
        for i in range(n):
            temp = numpy.multiply(X[i, 1:], mu[:])
            temp = numpy.multiply(temp, Phi[:, j])
            A[i, j] = numpy.sum(temp)
    return A


def get_A_chuncks(x, mu, Phi, q, datatype):
    m = x.shape[0]
    S = x.shape[1]
    s = S - q
    n = m * q
    l = Phi.shape[1]
    print(s, Phi.shape[0], mu.shape)
    A = numpy.zeros((n, l), dtype=datatype)
    mu_Phi = numpy.matmul(numpy.diag(mu), Phi)
    print("mu_Phi (s, l): ", mu_Phi.shape)
    starttime = time.time()
    for i in range(q):
        print(i)
        A[i * m : (i + 1) * m, :] = numpy.matmul(x[:, q - i : q - i + s], mu_Phi)
    print("Time: ", time.time() - starttime)
    return A


if __name__ == "__main__":

    results_path = settings.results_path
    l = settings.l
    q = settings.q
    datatype = settings.datatype
    label = "_sym_ARPACK"

    f = open("%s/X_backward_q_24.pkl" % results_path, "rb")
    X = pickle.load(f)
    f.close()
    print("X (mq=n, s+1): ", X.shape)

    # For sparse data
    if numpy.any(numpy.isnan(X)):
        print("NaN values in X")
        X[numpy.isnan(X)] = 0
        print("Set X NaNs to zero")

    #    if label == '_sym_ARPACK':
    #        mu_label = '_sym'
    #    else:
    #        mu_label = label
    # mu = joblib.load('%s/mu_P%s.jbl'%(results_path, mu_label))
    mu = joblib.load("%s/mu_rightEV.jbl" % (results_path))
    print("mu (s, ): ", mu.shape)

    evecs_norm = joblib.load(
        "%s/P%s_evecs_normalised_rightEV.jbl" % (results_path, label)
    )
    print("evecs_norm (s, s): ", evecs_norm.shape)

    #    print 'Test: diag(mu) * Evecs_norm * Evecs_norm.T'
    #    test = numpy.matmul(numpy.diag(mu),
    #                        numpy.matmul(evecs_norm,
    #                                     evecs_norm.T))
    #    diff = test - numpy.eye(s)
    #    print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
    #
    A = get_A(X, mu, evecs_norm, l)
    joblib.dump(A, "%s/A_norm_rightEV.jbl" % (results_path))

#    A_loop = get_A_loop(X, mu, evecs_norm, l)
#    joblib.dump(A_loop, '%s/A_loop.jbl'%(results_path))
#
#    diff = A - A_loop
#    print 'A-A_loop: ', numpy.amax(diff), numpy.amin(diff)
#
#    print  '\nGet A in chuncks'
#
#    f = open('%s/T_anomaly.pkl'%results_path,'rb')
#    x = pickle.load(f)
#    f.close()
#    # For sparse data
#    if numpy.any(numpy.isnan(x)):
#        print 'NaN values in x'
#        x[numpy.isnan(x)] = 0
#        print 'Set x NaNs to zero'

#    T_sparse = joblib.load('%s/converted_data/T_sparse_light.jbl'%results_path)
#    x = T_sparse[:,:].todense()
#    print 'x: ', x.shape, x.dtype
#    x = numpy.asarray(x, dtype=datatype)
#    print 'x (m, S): ', x.shape, x.dtype

#    Phi = evecs_norm[:,0:l]
#
#    A_chuncks = get_A_chuncks(x, mu, Phi, q, datatype)
#    joblib.dump(A_chuncks, '%s/A_chuncks.jbl'%(results_path))
#
#    diff = A - A_chuncks
#    print 'A-A_chuncks: ', numpy.amax(diff), numpy.amin(diff)
