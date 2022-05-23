#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:35:30 2022

@author: casadei_c
"""

import matplotlib.pyplot
import matplotlib.pylab
import joblib
import numpy

def makelabel(test_number):
    if test_number == 5:
        l = 'benchmark'
    elif test_number == 6:
        l = 'sparsepartial'
    else:
        l = 'undefined'
    return l

root_f = '../../synthetic_data_4'



flag = 0
if flag == 1:
    for test_n in [5, 6]:
        results_path = '%s/test%d'%(root_f, test_n)
        label = makelabel(test_n)
        
        q = 4000
        b = 3000
    
        CCs_SVD = joblib.load('%s/ssa/q_1/reconstruction_CC_vs_nmodes.jbl'%results_path)
        CCs_SSA = joblib.load('%s/ssa/q_%d/reconstruction_CC_vs_nmodes.jbl'%(results_path, q))
        CCs_fourier = joblib.load('%s/fourier_para_search/f_max_100_q_%d/reconstruction_CC_vs_nmodes.jbl'%(results_path, q))
        if test_n == 5:
            CCs_nlsa_e = joblib.load('%s/nlsa/q_%d/b_%d_eu_nns/log10eps_6p0/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, b))
        if test_n == 6:
            CCs_nlsa_e = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_%d_eu_nns/log10eps_1p0/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, b))
        
        Nmodes = len(CCs_SVD)+1
        
        matplotlib.pyplot.ylim(bottom=0.65, top=1.02)
        matplotlib.pyplot.xlim(left=0, right=Nmodes)
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_SVD, 'o-', c='b', markersize=3, markeredgewidth=0.0, label='SVD')
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_SSA, 'o-', color='indigo', markersize=10, markeredgewidth=0.0, label='SSA')#' (q=%d)'%q)
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_fourier, 'o-', c='c', markersize=7, markeredgewidth=0.0, label='LPSA')#' (q=%d, j$_{\mathrm{max}}$=100)'%q)
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_e, 'o-', c='m', markersize=3, markeredgewidth=0.0, label='E-NLSA')#' (q=%d, b=%d, l=50)'%(q, b))
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.xticks(range(1,Nmodes,2))
        matplotlib.pyplot.xlabel('number of modes', fontsize=18)
        matplotlib.pyplot.ylabel('correlation coefficient', fontsize=18)
        matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes_%s_SVD_SSA_LPSA_ENLSA.pdf'%(root_f, label), dpi=96*4)
        matplotlib.pyplot.close()
    


flag = 0
if flag == 1:
    test_n = 6
    results_path = '%s/test%d'%(root_f, test_n)
    label = makelabel(test_n)
    
    for b in [1500, 3000]:
        q=4000
        log10eps = '1p0'
        
        CCs_enlsa = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_%d_eu_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, b, log10eps))
        CCs_tnlsa = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_%d_time_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, b, log10eps))
        
        Nmodes = len(CCs_enlsa)+1
        matplotlib.pyplot.figure(figsize=(8,15))
        #matplotlib.pyplot.title('q = %d, b = %d, l = 50, log10eps = %s'%(q, b, log10eps), fontsize=20)
        matplotlib.pyplot.ylim(bottom=0.65, top=1.02)
        matplotlib.pyplot.xlim(left=0, right=Nmodes)
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_enlsa, 'o-', c='b', markersize=8, markeredgewidth=0.0, label='E-NLSA')
        matplotlib.pyplot.plot(range(1,Nmodes), CCs_tnlsa, 'o-', c='m', markersize=4, markeredgewidth=0.0, label='T-NLSA')
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=22)
        matplotlib.pyplot.xlabel('number of modes', fontsize=24)
        matplotlib.pyplot.ylabel('correlation coefficient', fontsize=24)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=22)
        matplotlib.pyplot.xticks(range(1,Nmodes,2))
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig('%s/reconstruction_ENLSA_TNLSA_%s_CC_vs_nmodes_q_%d_b_%d_l_50_minlegend.pdf'%(root_f, label, q, b), dpi=96*4)
        matplotlib.pyplot.close()


# flag = 0
# if flag == 1:
#     test_n = 6
#     results_path = '%s/test%d'%(root_f, test_n)
#     label = makelabel(test_n)
    
#     q=4000    
#     log10eps = '1p0'
    
#     CCs_enlsa_b_3000 = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_3000_eu_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, log10eps))
#     CCs_tnlsa_b_3000 = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_3000_time_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, log10eps))
#     CCs_enlsa_b_1500 = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_1500_eu_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, log10eps))
#     CCs_tnlsa_b_1500 = joblib.load('%s/nlsa/q_%d/distance_calculation_onlymeasured_normalised/b_1500_time_nns/log10eps_%s/reconstruction_CC_vs_nmodes.jbl'%(results_path, q, log10eps))
    
#     Nmodes = len(CCs_enlsa_b_3000)+1
#     #matplotlib.pyplot.figure(figsize=(15,10))
#     matplotlib.pyplot.title('q = %d, l = 50, log10eps = %s'%(q, log10eps))
#     matplotlib.pyplot.ylim(bottom=0.92, top=1.01)
#     matplotlib.pyplot.xlim(left=0, right=Nmodes)
#     matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
#     matplotlib.pyplot.plot(range(1,Nmodes), CCs_enlsa_b_3000, '-', c='c', markersize=8, markeredgewidth=0.0, label='E-NLSA, b=3000')
#     matplotlib.pyplot.plot(range(1,Nmodes), CCs_tnlsa_b_3000, '-', c='m', markersize=4, markeredgewidth=0.0, label='T-NLSA, b=3000')
#     matplotlib.pyplot.plot(range(1,Nmodes), CCs_enlsa_b_1500, '-', c='b', markersize=8, markeredgewidth=0.0, label='E-NLSA, b=1500')
#     matplotlib.pyplot.plot(range(1,Nmodes), CCs_tnlsa_b_1500, '-', c='k', markersize=4, markeredgewidth=0.0, label='T-NLSA, b=1500')
    
#     matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
#     matplotlib.pyplot.xlabel('n. modes', fontsize=14)
#     matplotlib.pyplot.ylabel('CC', fontsize=14)
#     matplotlib.pyplot.gca().tick_params(axis='both', labelsize=12)
#     matplotlib.pyplot.xticks(range(1,Nmodes,2))
#     matplotlib.pyplot.savefig('%s/reconstruction_ENLSA_TNLSA_%s_CC_vs_nmodes_b.png'%(results_path, label), dpi=96*3)
#     matplotlib.pyplot.close()



##############
#### LPSA ####
##############


#### q-scan, CC ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves)) 
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))           
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.ylim(bottom=0.64, top=1.02)
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('correlation coefficient', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='$q$=%d'%q)  
            
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_CC_vs_nmodes_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()

#### jmax-scan, CC ####
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic')  
    for test_n in [5,6]:
    
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))      
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.ylim(bottom=0.60, top=1.02)
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('correlation coefficient', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_CC_vs_nmodes_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()

#### q-scan, L ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='$q$=%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_L_vs_nmodes_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()
        
#### jmax-scan, L ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5,6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))  
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)       
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_L_vs_nmodes_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()


#### q-scan, SVs ####        
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_1)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%folder)
            matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$q$=%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/SVs_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()
        

#### jmax-scan, SVs ####        
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5,6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_1)$', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%folder)
            if S.shape[0] < 20:
                n_m = S.shape[0]
            else:
                n_m = 20
            matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/SVs_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()
        
        
################################################       
#########       Jitter study:       ############
################################################

#### q-scan, L of central block ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [6]:
        root_f = '../../synthetic_data_jitter'
        test_f = '%s/test%d/LPSA_para_search'%(root_f, test_n)
        
        if test_n == 5:
            label = 'jitter_factor_0p3'
        if test_n == 6:
            label = 'jitter_factor_1p0'
              
        n_m = 20
        f_max = 100
        p = 0
        qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            lls = joblib.load('%s/p_%d_local_linearity_vs_nmodes.jbl'%(folder,p))
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='$q$=%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/LPSA_p_%d_reconstruct_log10_L_vs_nmodes_%s_q_scan_fmax_%d.png'%(test_f, p, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()
        
#### q-scan, SVD of A ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [6]:
        root_f = '../../synthetic_data_jitter'
        test_f = '%s/test%d/LPSA_para_search'%(root_f, test_n)
        
        if test_n == 5:
            label = 'jitter_factor_0p3'
        if test_n == 6:
            label = 'jitter_factor_1p0'   
            
        n_m = 20
        f_max = 100
        
        qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_1)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%(folder))
            matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$q=$%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/LPSA_A_SVs_vs_modes_%s_q_scan_fmax_%d.png'%(test_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()


#### jmax-scan, L of central block ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [6]:
        root_f = '../../synthetic_data_jitter'
        test_f = '%s/test%d/LPSA_para_search'%(root_f, test_n)
        
        if test_n == 5:
            label = 'jitter_factor_0p3'
        if test_n == 6:
            label = 'jitter_factor_1p0'   
              
        n_m = 20
        q = 4001
        p = 0
        f_max_s = [1, 5, 10, 50, 100, 150]#, 300] 
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('number of modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, fmax in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, fmax, q)
            lls = joblib.load('%s/p_%d_local_linearity_vs_nmodes.jbl'%(folder,p))
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%fmax)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/LPSA_p_%d_reconstruct_log10_L_vs_nmodes_%s_fmax_scan_q_%d.png'%(test_f, p, label, q), dpi=96*4)
        matplotlib.pyplot.close()
        
        
#### jmax-scan, SVD of A ####    
flag = 0
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [6]:
        root_f = '../../synthetic_data_jitter'
        test_f = '%s/test%d/LPSA_para_search'%(root_f, test_n)
        
        if test_n == 5:
            label = 'jitter_factor_0p3'
        if test_n == 6:
            label = 'jitter_factor_1p0'   
            
        n_m = 20
        q = 4001
        p = 0
        f_max_s = [1, 5, 10, 50, 100, 150]#, 300] 
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_1)$', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%(folder))
            if S.shape[0] < 20:
                n_m = S.shape[0]
            else:
                n_m = 20
            matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/LPSA_A_SVs_vs_modes_%s_fmax_scan_q_%d.png'%(test_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()


# SVD of reconstructed signal        
flag = 1
if flag ==1:
    n_m = 20
    q = 4001
    fmax = 100
    test_n = 6
    if test_n == 5:
        label = 'jitter_factor_0p3'
    if test_n == 6:
        label = 'jitter_factor_1p0'
          
    root_path = '../../synthetic_data_jitter'
    results_path = '%s/test%d/LPSA_para_search/f_max_%d_q_%d'%(root_path, test_n, fmax, q)
    
    ps = [0, 9, 100, 2000]
    
    n_curves = len(ps)        
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))      
    matplotlib.pyplot.figure(figsize=(10,10))  
    matplotlib.pyplot.xticks(range(1,n_m+1,2))   
    
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
    matplotlib.pyplot.xlim(left=0, right=n_m+1)
    
    matplotlib.pyplot.xlabel('number of modes', fontsize=20)
    
    matplotlib.pyplot.ylabel('Correlation coefficient to benchmark', fontsize=20)
    
    for i, p in enumerate(ps):
        folder = '%s/reconstruction_p_%d/x_r_SVD'%(results_path, p)
        
        CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%(folder))
        
        print CCs[9]
        matplotlib.pyplot.plot(range(1, n_m+1), CCs, '-o', c=colors[i], label='$p$=%d'%p)  
    
    matplotlib.pyplot.ylim(min(CCs)-0.01, top=1.02)
    matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
    matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
    matplotlib.pyplot.savefig('%s/test%d/LPSA_para_search/%s_q_%d_fmax_%d_x_r_4modes_reconstruction_CC_vs_nmodes_p.png'%(root_path, test_n, label, q, fmax), dpi=96*4)
    matplotlib.pyplot.close() 
    
flag = 1
if flag ==1:
    n_m = 20
    q = 4001
    fmax = 100
    test_n = 6
    
    if test_n == 5:
        label = 'jitter_factor_0p3'
    if test_n == 6:
        label = 'jitter_factor_1p0'
        
    root_path = '../../synthetic_data_jitter'
    results_path = '%s/test%d/LPSA_para_search/f_max_%d_q_%d'%(root_path, test_n, fmax, q)
    
    ps = [0, 9, 100, 2000]
    
    n_curves = len(ps)        
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))      
    matplotlib.pyplot.figure(figsize=(10,10))  
    matplotlib.pyplot.xticks(range(1,n_m+1,2))   
    matplotlib.pyplot.xlim(left=0, right=n_m+1)
    matplotlib.pyplot.xlabel('number of modes', fontsize=20)
    matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
    
    for i, p in enumerate(ps):
        folder = '%s/reconstruction_p_%d/x_r_SVD'%(results_path, p)
        
        Ls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%(folder))
            
        matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(Ls), '-o', c=colors[i], label='$p$=%d'%p)  
    
    matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=20)
    matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
    matplotlib.pyplot.savefig('%s/test%d/LPSA_para_search/%s_q_%d_fmax_%d_x_r_4modes_reconstruction_log10L_vs_nmodes_p.png'%(root_path, test_n, label, q, fmax), dpi=96*4)
    matplotlib.pyplot.close() 
    

flag = 1
if flag ==1:
    n_m = 20
    q = 4001
    fmax = 100
    test_n = 6
    if test_n == 5:
        label = 'jitter_factor_0p3'
    if test_n == 6:
        label = 'jitter_factor_1p0'
        
    root_path = '../../synthetic_data_jitter'
    results_path = '%s/test%d/LPSA_para_search/f_max_%d_q_%d'%(root_path, test_n, fmax, q)
    
    ps = [0, 9, 100, 2000]
    
    n_curves = len(ps)        
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))      
    matplotlib.pyplot.figure(figsize=(10,10))  
    matplotlib.pyplot.xticks(range(1,n_m+1,2))   
    matplotlib.pyplot.xlim(left=0, right=n_m+1)
    matplotlib.pyplot.xlabel('mode', fontsize=20)
    matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_1)$', fontsize=20)
    
    for i, p in enumerate(ps):
        folder = '%s/reconstruction_p_%d/x_r_SVD'%(results_path, p)
        S = joblib.load('%s/S.jbl'%folder)
        matplotlib.pyplot.plot(range(1, n_m+1), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$p$=%d'%p)  
        
    matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=20)
    matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
    matplotlib.pyplot.savefig('%s/test%d/LPSA_para_search/%s_q_%d_fmax_%d_x_r_4modes_SVD_S_vs_nmodes_p.png'%(root_path, test_n, label, q, fmax), dpi=96*4)
    matplotlib.pyplot.close() 
