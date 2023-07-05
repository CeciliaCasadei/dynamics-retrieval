#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:52:38 2023

@author: casadei_c
"""
import os

import joblib
import numpy


def f(T_matrix, M_matrix):
    I = numpy.sum(T_matrix, axis=1)
    n = numpy.sum(M_matrix, axis=1)
    I = I / n

    I[numpy.isnan(I)] = 0
    I[I < 0] = 0
    sigI = numpy.sqrt(I)

    return I, sigI


def main():

    # data_path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA'
    data_path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA/scan_19"

    fpath = "%s/binning_LTD" % (data_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    # LIGHT

    miller_h = joblib.load("%s/converted_data_light/miller_h_light.jbl" % (data_path))
    miller_k = joblib.load("%s/converted_data_light/miller_k_light.jbl" % (data_path))
    miller_l = joblib.load("%s/converted_data_light/miller_l_light.jbl" % (data_path))

    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    print("out: ", out.shape)

    T = joblib.load("%s/converted_data_light/T_sparse_LTD_light.jbl" % (data_path))
    M = joblib.load("%s/converted_data_light/M_sparse_light.jbl" % (data_path))
    ts = joblib.load("%s/converted_data_light/t_light.jbl" % (data_path))
    print("T: ", T.shape)
    print("M: ", M.shape, end=' ')
    print("ts: ", ts.shape)

    N = 20000

    T_early = T[:, 0:N]
    M_early = M[:, 0:N]
    print("Early times", ts[0, 0], ts[N, 0])
    print(T_early.shape)

    T_late = T[:, T.shape[1] - N :]
    M_late = M[:, M.shape[1] - N :]
    print("Late times", ts[T.shape[1] - N, 0], ts[-1, 0])
    print(T_late.shape)

    # LIGHT - early
    I_early, sigI_early = f(T_early, M_early)

    out[:, 3] = I_early.flatten()
    out[:, 4] = sigI_early.flatten()

    f_out = "%s/I_early_avg.txt" % (fpath)
    numpy.savetxt(f_out, out, fmt="%6d%6d%6d%17.2f%17.2f")

    # LIGHT - late
    I_late, sigI_late = f(T_late, M_late)

    out[:, 3] = I_late.flatten()
    out[:, 4] = sigI_late.flatten()

    f_out = "%s/I_late_avg.txt" % (fpath)
    numpy.savetxt(f_out, out, fmt="%6d%6d%6d%17.2f%17.2f")
    """
    # DARK    
    miller_h = joblib.load('%s/converted_data_alldark/miller_h_alldark.jbl'%(data_path))
    miller_k = joblib.load('%s/converted_data_alldark/miller_k_alldark.jbl'%(data_path))
    miller_l = joblib.load('%s/converted_data_alldark/miller_l_alldark.jbl'%(data_path))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    print 'out: ', out.shape
    
    T_dark  = joblib.load('%s/converted_data_alldark/T_sparse_LTD_alldark.jbl'%(data_path))
    M_dark  = joblib.load('%s/converted_data_alldark/M_sparse_alldark.jbl'%(data_path))
    print T_dark.shape
    
    I_dark, sigI_dark = f(T_dark, M_dark)
        
    out[:, 3] = I_dark.flatten()
    out[:, 4] = sigI_dark.flatten()
    
    f_out = '%s/I_dark_avg.txt'%(fpath)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
    """


def merge_all():

    # data_path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA'
    data_path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA"

    fpath = "%s/translation_correction_light" % (data_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    # LIGHT

    miller_h = joblib.load("%s/converted_data_light/miller_h_light.jbl" % (data_path))
    miller_k = joblib.load("%s/converted_data_light/miller_k_light.jbl" % (data_path))
    miller_l = joblib.load("%s/converted_data_light/miller_l_light.jbl" % (data_path))

    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    print("out: ", out.shape)

    T = joblib.load("%s/converted_data_light/T_sparse_light.jbl" % (data_path))
    M = joblib.load("%s/converted_data_light/M_sparse_light.jbl" % (data_path))
    ts = joblib.load("%s/converted_data_light/t_light.jbl" % (data_path))
    print("T: ", T.shape)
    print("M: ", M.shape, end=' ')
    print("ts: ", ts.shape)

    I, sigI = f(T, M)

    out[:, 3] = I.flatten()
    out[:, 4] = sigI.flatten()

    f_out = "%s/I_light_avg.txt" % (fpath)
    numpy.savetxt(f_out, out, fmt="%6d%6d%6d%17.2f%17.2f")
    
def main_sliding_window():

    # data_path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA'
    data_path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA"

    fpath = "%s/binning_LTD_sliding_window" % (data_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    # LIGHT

    miller_h = joblib.load("%s/converted_data_light/miller_h_light.jbl" % (data_path))
    miller_k = joblib.load("%s/converted_data_light/miller_k_light.jbl" % (data_path))
    miller_l = joblib.load("%s/converted_data_light/miller_l_light.jbl" % (data_path))

    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    print("out: ", out.shape)

    T = joblib.load("%s/converted_data_light/T_sparse_LTD_light.jbl" % (data_path))
    M = joblib.load("%s/converted_data_light/M_sparse_light.jbl" % (data_path))
    ts = joblib.load("%s/converted_data_light/t_light.jbl" % (data_path))
    print("T: ", T.shape)
    print("M: ", M.shape, end=' ')
    print("ts: ", ts.shape)

    stepsize = 20000
    binsize = 2*stepsize
    
    start = 0
    for i in range(0, 1+int(float(T.shape[1]-binsize)/stepsize)):
        start = i*stepsize
        end = start+binsize
        
        t_start = ts[start]
        t_end = ts[end]
        
        print(start, end, t_start, t_end)


        T_bin = T[:, start:end]
        M_bin = M[:, start:end]
    
        I_bin, sigI_bin = f(T_bin, M_bin)

        out[:, 3] = I_bin.flatten()
        out[:, 4] = sigI_bin.flatten()

        f_out = "%s/I_bin_%d_%d_avg_%0.1ffs_%0.1ffs.txt" % (fpath, start, end, t_start, t_end)
        print(f_out)
        numpy.savetxt(f_out, out, fmt="%6d%6d%6d%17.2f%17.2f")