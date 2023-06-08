# -*- coding: utf-8 -*-
import time

import joblib
import numpy
import scipy.linalg


def density_normalisation_memopt(W_sym, datatype):
    s = W_sym.shape[0]
    # Q = numpy.sum(W_sym, axis=0)
    # CHANGE 07/10/2021
    Q = numpy.sum(W_sym, axis=1)
    print("Q", Q.shape)
    W_norm = numpy.zeros((s, s), dtype=datatype)
    start = time.time()
    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        Q_i = Q[i]
        # print 'Q_i', Q_i
        for j in range(s):
            if abs(W_sym[i, j]) > 0:
                Q_j = Q[j]
                # print 'Q_i, Q_j, Q_i*Q_j', Q_i, Q_j, Q_i*Q_j
                W_norm[i, j] = W_sym[i, j] / (Q_i * Q_j)
    print("It took: ", time.time() - start)
    return W_norm
 
       

def density_normalisation_timeopt(W_sym):
    s = W_sym.shape[0]
    Q = numpy.sum(W_sym, axis=0)
    Q = Q[:, numpy.newaxis]
    Q_rep = numpy.repeat(Q, s, axis=1)

    norm_factor = numpy.multiply(Q_rep, Q_rep.T)
    W_norm_opt = W_sym / norm_factor

    return W_norm_opt
    



def get_W_tilde_memopt(W_norm, datatype):
    s = W_norm.shape[0]
    sqrt_Q = numpy.sqrt(numpy.sum(W_norm, axis=1))
    W_tilde = numpy.zeros((s, s), dtype=datatype)
    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        sqrt_Q_i = sqrt_Q[i]
        W_tilde[i, :] = W_norm[i, :] / sqrt_Q_i
    for i in range(s):
        if i % 1000 == 0:
            print(i, "/", s)
        sqrt_Q_i = sqrt_Q[i]
        W_tilde[:, i] = W_tilde[:, i] / sqrt_Q_i
    return W_tilde



def main(settings):

    results_path = settings.results_path
    datatype = settings.datatype

    flag_W_norm = 1
    if flag_W_norm == 1:
        print("Load W_sym")
        W_sym = joblib.load("%s/W_sym.jbl" % results_path)
        print("Density normalization (low memory use, long running time)")
        W_norm = density_normalisation_memopt(W_sym, datatype)
        joblib.dump(W_norm, "%s/W_norm.jbl" % results_path)

    flag_check_W_norm = 1
    if flag_check_W_norm == 1:
        W_norm = joblib.load("%s/W_norm.jbl" % results_path)
        print("Check that W_norm is symmetric")
        diff = W_norm - W_norm.T
        print(numpy.amax(diff), numpy.amin(diff))

    flag_W_tilde = 1
    if flag_W_tilde == 1:
        W_norm = joblib.load("%s/W_norm.jbl" % results_path)
        print("Get W_tilde")
        W_tilde = get_W_tilde_memopt(W_norm, datatype)
        joblib.dump(W_tilde, "%s/W_tilde.jbl" % results_path)

    flag_check_W_tilde = 1
    if flag_check_W_tilde == 1:
        W_tilde = joblib.load("%s/W_tilde.jbl" % results_path)
        print("Check that W_tilde is symmetric")
        diff = W_tilde - W_tilde.T
        print(numpy.amax(diff), numpy.amin(diff))     