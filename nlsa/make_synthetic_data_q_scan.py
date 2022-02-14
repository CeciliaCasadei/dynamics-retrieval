#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:49:34 2022

@author: casadei_c
"""
import os
import numpy
import math
import joblib
import matplotlib.pyplot
import correlate

def make_lp_filter_functions(settings):
    import make_lp_filter
    
    F = make_lp_filter.get_F(settings)
    Q, R = make_lp_filter.on_qr(settings, F)
    
    d = make_lp_filter.check_on(Q)
    print numpy.amax(abs(d))
    
    joblib.dump(Q, '%s/F_on_qr.jbl'%settings.results_path)


root_path = '../../synthetic_data_4/test6/fourier_para_search' 
m = 7000
S = 30000
paral_step_A = 400
paral_step_reconstruction = 1000
f_max = 100
f_max_considered = f_max


# Fourier filtering para search

x = joblib.load('%s/x.jbl'%root_path)
print x.shape

qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]

flag = 0
if flag == 1:
    for q in qs:
        q_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)
            
flag = 0
if flag == 1:
    for q in qs:
        q_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        fopen = open('/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/nlsa/settings_q_%d.py'%q, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('S = %d\n'%S)
        fopen.write('m = %d\n'%m)
        fopen.write('q = %d\n'%q)    
        fopen.write('f_max = %d\n'%f_max)
        fopen.write('f_max_considered = f_max\n')
        fopen.write('results_path = "%s"\n'%q_path)
        fopen.write('paral_step_A = %d\n'%paral_step_A)
        fopen.write('datatype = numpy.float64\n')
        data_file = '%s/x.jbl'%q_path
        fopen.write('data_file = "%s"\n'%data_file)

        n_workers_A = int(math.ceil(float(q)/paral_step_A))
        fopen.write('n_workers_A = %d\n'%n_workers_A)
        ncopies = q
        fopen.write('ncopies = %d\n'%ncopies)
        fopen.write('modes_to_reconstruct = range(12)\n')
        fopen.write('paral_step_reconstruction = %d\n'%paral_step_reconstruction)


        n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))
        fopen.write('n_workers_reconstruction = %d\n'%n_workers_reconstruction)
        fopen.close()

flag = 0
if flag == 1:    
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        
        print settings.q
        print settings.results_path
        
        make_lp_filter_functions(settings)
       
    
flag = 0
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        
        print settings.q
        end_worker = settings.n_workers_A - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s'
                  %(end_worker, settings.__name__)) 


flag = 0
if flag == 1:
    import nlsa.util_merge_A
    for q in qs:
        print q
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        
        print settings.q
        nlsa.util_merge_A.main(settings)

flag = 0
if flag == 1: 
    import nlsa.SVD
    print '\n****** RUNNING SVD ******'
    for q in qs:
        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        
        print 'q: ', settings.q
    
        results_path = settings.results_path
        datatype = settings.datatype
    
        A = joblib.load('%s/A_parallel.jbl'%results_path)
        print 'Loaded'
        U, S, VH = nlsa.SVD.SVD_f_manual(A)
        U, S, VH = nlsa.SVD.sorting(U, S, VH)
        
        print 'Done'
        print 'U: ', U.shape
        print 'S: ', S.shape
        print 'VH: ', VH.shape
    
        joblib.dump(U, '%s/U.jbl'%results_path)
        joblib.dump(S, '%s/S.jbl'%results_path)
        joblib.dump(VH, '%s/VH.jbl'%results_path)
        
        evecs = joblib.load('%s/F_on_qr.jbl'%(results_path))
        Phi = evecs[:,0:2*settings.f_max_considered+1]
        
        VT_final = nlsa.SVD.project_chronos(VH, Phi)    
        print 'VT_final: ', VT_final.shape
        joblib.dump(VT_final, '%s/VT_final.jbl'%results_path)
  
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    import nlsa.plot_chronos
    
    for q in qs:
        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        nlsa.plot_SVs.main(settings)       
        nlsa.plot_chronos.main(settings)
        

flag = 0
if flag == 1:
    for q in qs:
        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        
        end_worker = settings.n_workers_reconstruction - 1
        os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
                  %(end_worker, settings.__name__))    
        


flag = 0
if flag == 1:
    import nlsa.util_merge_x_r  
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        for mode in settings.modes_to_reconstruct:
            nlsa.util_merge_x_r.f(settings, mode) 
 
qs = [2000]
flag = 1
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        
        benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
        print benchmark.shape
        benchmark = benchmark[:, settings.q:settings.q+(settings.S-settings.q-settings.ncopies+1)]
        print benchmark.shape
        benchmark = benchmark.flatten()
    
        CCs = []
        
        x_r_tot = 0
        for mode in range(12):#settings.modes_to_reconstruct:
            print mode
            x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
            matplotlib.pyplot.imshow(x_r, cmap='jet')
            matplotlib.pyplot.colorbar()
            matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
            matplotlib.pyplot.close()  
            x_r_tot += x_r
        
            matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
            matplotlib.pyplot.colorbar()
            
            print x_r_tot.shape
                   
            x_r_tot_flat = x_r_tot.flatten()
            
            CC = correlate.Correlate(benchmark, x_r_tot_flat)
            print CC
            
            matplotlib.pyplot.title('%.4f'%CC)
            matplotlib.pyplot.savefig('%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), dpi=96*3)
            matplotlib.pyplot.close() 
            
            CCs.append(CC)
        joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
        
        
 
flag = 0
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
        matplotlib.pyplot.scatter(range(1, len(CCs)+1), CCs, c='b')
        matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
        matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes_q_%d.png'%(settings.results_path, q))
        matplotlib.pyplot.close()

flag = 0
if flag == 1:
    #matplotlib.pyplot.style.use('classic') 
    n_qs = len(qs)
    import matplotlib.pylab
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0,1,n_qs))   
    matplotlib.pyplot.figure(figsize=(10,10))  
    n = 12   
    matplotlib.pyplot.xticks(range(1,n+1,2))   
    for i, q in enumerate(qs):
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
        matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='q=%d'%q)  
    matplotlib.pyplot.legend(frameon=False)
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes_q_scan.png'%(root_path), dpi=96*3)
    matplotlib.pyplot.close()
