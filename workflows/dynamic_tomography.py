#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:03:08 2023

@author: casadei_c
"""
import os
import numpy
import joblib
import scipy.io
import matplotlib.pyplot
import random

def plot_real_space_tomogram():
    path_name = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_tomography/tomo_all_imgs'
    
    for i in range(1,801):
        fn = '%s/tomo_all_img_%0.3d.mat'%(path_name, i)
        print(fn)
    
        matfile = scipy.io.loadmat(fn)
    
        print(list(matfile.keys()))
    
        tomogram_real   = matfile['tomo_all_img']
        print(tomogram_real.shape)
        
        matplotlib.pyplot.imshow(tomogram_real, vmin=0, vmax=8)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.savefig('%s/tomo_real_time_%0.6d.png'%(path_name, i))
        matplotlib.pyplot.close()
        
def get_syn_projections():
    path_name = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_tomography/projections_real_space'
    
    n_proj_pxls = 200
    n_ts = 800
    n_projections = 180
    
    pxl_n = n_proj_pxls*10+190
    trace = []
    for i in range(1,n_ts+1):
        fn = '%s/proj_real_t_%0.3d.mat'%(path_name, i)
        print(fn)
        
        matfile = scipy.io.loadmat(fn)
        print(list(matfile.keys()))   
        proj_real   = matfile['proj_real']
        print(proj_real.shape[0], 'pxls', proj_real.shape[1], 'directions')     # 200 pxls x 180 theta values
        proj_real_flat = proj_real.flatten('F') # column-major
        print(proj_real_flat.shape, n_proj_pxls*n_projections)
        trace.append(proj_real_flat[pxl_n,])
        joblib.dump(proj_real_flat, 
                    "%s/combined_projections_t_%0.3d.jbl"%(path_name, i))
        
    matplotlib.pyplot.scatter(list(range(1,n_ts+1)), trace)
    matplotlib.pyplot.savefig("%s/projections_alltheta_pxl_%d_vs_t.png"
                              %(path_name, pxl_n))
    matplotlib.pyplot.close()
    
    deg = 10
    matplotlib.pyplot.scatter(list(range(0,n_proj_pxls)), 
                              proj_real_flat[n_proj_pxls*deg:n_proj_pxls*(deg+1),])
    matplotlib.pyplot.savefig("%s/projection_theta_%d_degrees_t_%0.3d.png"
                              %(path_name, deg, i))
    matplotlib.pyplot.close()

def merge_projections():
    path_name = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_tomography/projections_real_space'
    n_proj_pxls = 200
    n_projections = 180
    m = n_proj_pxls * n_projections
    S = 500
    x = numpy.zeros(shape=(m,S))
    
    for t in range(S):
        print(t)
        x_column = joblib.load("%s/combined_projections_t_%0.3d.jbl"
                               %(path_name, t+1))
        x[:,t] = x_column
    joblib.dump(x, 
                "%s/benchmark.jbl"%(path_name))

def make_sparse_data():
    path_name = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_tomography/projections_real_space'
    bm = joblib.load("%s/benchmark.jbl"%(path_name))  
    
    sparsity_level = 0.5 #0.9  # 90% present #0.1 # 10% present
    
    n_proj_pxls = 200
    n_projections = 180
    m = n_proj_pxls * n_projections
    S = 500
    
    n_proj_per_tstep = int(sparsity_level * n_projections)
    print('N. measured projections per timestep: ', n_proj_per_tstep)
    
    x_input = numpy.zeros(shape=(m,S))
    mask = numpy.zeros(shape=(m,S))
    for i in range(S):
        idxs = random.sample(list(range(0, n_projections)), n_proj_per_tstep)
        
        for idx in idxs:
            x_input[idx*n_proj_pxls:(idx+1)*n_proj_pxls, i] = bm[idx*n_proj_pxls:(idx+1)*n_proj_pxls, i]
            mask[idx*n_proj_pxls:(idx+1)*n_proj_pxls, i] = 1
            
    joblib.dump(x_input, 
                "%s/input_data_sparsity_%0.2f.jbl"
                %(path_name, sparsity_level))
    joblib.dump(mask, 
                "%s/input_data_mask_sparsity_%0.2f.jbl"
                %(path_name, sparsity_level))

# MAKE DATA
flag = 0
if flag == 1:
    get_syn_projections()
flag = 0
if flag == 1:
    merge_projections()
flag = 0
if flag == 1:
    make_sparse_data()

flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings

    import dynamics_retrieval.calculate_dI

    dynamics_retrieval.calculate_dI.main(settings)
    
flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings

    import dynamics_retrieval.boost

    dynamics_retrieval.boost.main(settings)
    
flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings

    import dynamics_retrieval.mirror_dataset

    dynamics_retrieval.mirror_dataset.main(settings)
    
    ts_mirrored = numpy.asarray(list(range(0, 2*settings.S)))
    joblib.dump(
        ts_mirrored, "%s/ts_mirrored.jbl" % (settings.results_path)
    )


flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings
    from scipy import sparse

    x_sp = joblib.load(
        "%s/dT_bst_mirrored.jbl" % settings.results_path
    )
    print(x_sp[1000, :])
    x = x_sp[:, :].todense()
    if numpy.isnan(x).any():
        print ("x contain NaN values")
        N = numpy.count_nonzero(numpy.isnan(x))
        print("N nans: ", N, "out of ", x.shape)
        x[numpy.isnan(x)] = 0
    else:
        print ("x does not contain NaN values")
        
        
qs = [1, 11, 51, 81, 101, 121, 151]

flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings

    import dynamics_retrieval.make_settings

    for q in qs:

        # MAKE OUTPUT FOLDER
        q_path = "%s/f_max_%d_q_%d" % (settings.results_path, settings.f_max_q_scan, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)

        # MAKE SETTINGS FILE
        data_file = "%s/dT_bst_mirrored.jbl" % (
            q_path,
        )
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_q_%d.py"
            % q
        )
        dynamics_retrieval.make_settings.main(
            settings,
            fn,
            q,
            settings.f_max_q_scan,
            q_path,
            data_file,
            min(20, 2 * settings.f_max_q_scan + 1),
        )
        
        
f_max_s = [2, 4, 6, 10, 30, 70, 100, 120, 150]#50

flag = 0
if flag == 1:
    import settings_dynamic_tomography as settings

    import dynamics_retrieval.make_settings

    for f_max in f_max_s:

        # MAKE OUTPUT FOLDER
        f_max_path = "%s/f_max_%d_q_%d" % (
            settings.results_path,
            f_max,
            settings.q_f_max_scan,
        )
        if not os.path.exists(f_max_path):
            os.mkdir(f_max_path)

        # MAKE SETTINGS FILE
        data_file = "%s/dT_bst_mirrored.jbl" % (
            f_max_path,
        )
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_f_max_%d.py"
            % f_max
        )
        dynamics_retrieval.make_settings.main(
            settings,
            fn,
            settings.q_f_max_scan,
            f_max,
            f_max_path,
            data_file,
            min(20, 2 * f_max + 1),
        )

############# START ##################
flag = 0
if flag == 1:
    import dynamics_retrieval.make_lp_filter

    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)

        F = dynamics_retrieval.make_lp_filter.get_F_sv_t_range(settings)
        Q, R = dynamics_retrieval.make_lp_filter.on_qr(settings, F)
        d = dynamics_retrieval.make_lp_filter.check_on(Q)
        print("Normalisation: ", numpy.amax(abs(d)))
        joblib.dump(Q, "%s/F_on.jbl" % settings.results_path)


flag = 0
if flag == 1:
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        end_worker = (2 * settings.f_max + 1) - 1
        os.system(
            "sbatch -p hour --mem=100G --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s"
            % (end_worker, settings.__name__)
        )
        
flag = 0
if flag == 1:
    import dynamics_retrieval.merge_aj

    # for f_max in f_max_s:
    #     modulename = "settings_f_max_%d" % f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        dynamics_retrieval.merge_aj.main(settings)
        
flag = 0
if flag == 1:
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)

        end_worker = (2 * settings.f_max + 1) - 1
        os.system(
            "sbatch -p hour --mem=40G --array=0-%d ../scripts_parallel_submission/run_parallel_ATA.sh %s"
            % (end_worker, settings.__name__)
        )
        
flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_ATA_merge

    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        dynamics_retrieval.calculate_ATA_merge.main(settings)
        
flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        dynamics_retrieval.SVD.get_chronos(settings)


flag = 0
if flag == 1:
    import dynamics_retrieval.plot_chronos
    import dynamics_retrieval.plot_SVs

    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        dynamics_retrieval.plot_SVs.main(settings)
        dynamics_retrieval.plot_chronos.main(settings)
        

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)
        dynamics_retrieval.SVD.get_topos(settings)

        
flag = 0
if flag == 1:
    import dynamics_retrieval.reconstruct_p

    for q in qs:
        modulename = 'settings_q_%d'%q
    # for f_max in f_max_s:
    #     modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)

        dynamics_retrieval.reconstruct_p.f(settings)
        dynamics_retrieval.reconstruct_p.f_ts(settings)      
        
flag = 1
if flag == 1:
    import dynamics_retrieval.plot_syn_data
    import dynamics_retrieval.correlate
    from scipy import sparse
    import settings_dynamic_tomography as settings
    
    
    T = joblib.load("%s/input_data_sparsity_0.50.jbl" % (settings.results_path))
    M = joblib.load("%s/input_data_mask_sparsity_0.50.jbl" % (settings.results_path))
    print("T, is sparse: ", sparse.issparse(T), T.shape, T.dtype)
    print("M, is sparse: ", sparse.issparse(M), M.shape, M.dtype)

    if sparse.issparse(T) == False:
        T = sparse.csr_matrix(T)
    print("T, is sparse:", sparse.issparse(T), T.dtype, T.shape)
    if sparse.issparse(M) == False:
        M = sparse.csr_matrix(M)
    print("M, is sparse:", sparse.issparse(M), M.dtype, M.shape)

    ns = numpy.sum(M, axis=1)
    print("ns: ", ns.shape, ns.dtype)

    avgs = numpy.sum(T, axis=1) / ns
    print("avgs: ", avgs.shape, avgs.dtype)

    avgs_matrix = numpy.repeat(avgs, settings.S, axis=1)
    print("avgs: ", avgs_matrix.shape, avgs_matrix.dtype)
    print('element 150: ', avgs_matrix[150,0])
    
    # f_max = 150
    # modulename = "settings_f_max_%d" % f_max
    
    q = 1
    modulename = 'settings_q_%d'%q

    settings = __import__(modulename)

    m = settings.m
    S = settings.S
    q = settings.q
    p = settings.p
    results_path = "%s/reconstruction_p_%d" % (settings.results_path, p)

    print("jmax: ", settings.f_max)
    print("q: ", settings.q)
    print("p: ", settings.p)
    print(results_path)

    t_r = joblib.load("%s/t_r_p_%d.jbl" % (results_path, p))
    print('Reconstructed times: ', t_r.shape, t_r[0], t_r[-1])
    benchmark = joblib.load("%s/benchmark.jbl" % (results_path))
    print('Benchmark: ', benchmark.shape)

    start_idx = int(t_r[0])
    end_idx = start_idx + t_r.shape[0]
    print(start_idx, end_idx)
    bm = benchmark[:, start_idx:end_idx]
    print('Benchmark: ', bm.shape)

    benchmark_flat = bm.flatten()

    CCs = []
    nmodes = 20
    x_r_tot = avgs_matrix[:,start_idx:end_idx]
    
    
    x_r_tot_flat = x_r_tot.flatten()
    CC = dynamics_retrieval.correlate.Correlate(benchmark_flat, x_r_tot_flat)
    CCs.append(CC)
    print('CC: ', CC)
        
    pxl_ns = [1870, 2005, 2190, 10150, 19887]
    for pxl_n in pxl_ns:
        y = x_r_tot[pxl_n,:]
        y_T = y.T
        matplotlib.pyplot.plot(t_r, y_T, 'bo')
        matplotlib.pyplot.plot(list(range(start_idx,end_idx)), bm[pxl_n,:], 'm')
        matplotlib.pyplot.savefig("%s/projections_alltheta_pxl_%d_vs_t_0_modes.png"
                          %(results_path, pxl_n))
        matplotlib.pyplot.close()
    
    for mode in range(0, min(nmodes, 2 * settings.f_max + 1)):
        print("Mode: ", mode)

        x_r = joblib.load("%s/movie_p_%d_mode_%d.jbl" % (results_path, p, mode))
        print('x_r:', x_r.shape)
        
        print(x_r_tot[150, 30], 'plus ', x_r[150, 30])
        x_r_tot += x_r
        print('result ', x_r_tot[150, 30])
        
        x_r_tot_flat = x_r_tot.flatten()
        CC = dynamics_retrieval.correlate.Correlate(benchmark_flat, x_r_tot_flat)
        CCs.append(CC)
        print('CC: ', CC)
        
        pxl_ns = [1870, 2005, 2190, 10150, 19887]
        for pxl_n in pxl_ns:
            y = x_r_tot[pxl_n,:]
            y_T = y.T
            matplotlib.pyplot.plot(t_r, y_T, 'bo')
            matplotlib.pyplot.plot(list(range(start_idx,end_idx)), bm[pxl_n,:], 'm')
            matplotlib.pyplot.savefig("%s/projections_alltheta_pxl_%d_vs_t_%d_modes.png"
                              %(results_path, pxl_n, mode+1))
            matplotlib.pyplot.close()
    

    joblib.dump(CCs, "%s/CCs_to_benchmark.jbl" % (results_path))

    matplotlib.pyplot.scatter(list(range(0, len(CCs))), CCs)
    matplotlib.pyplot.axhline(y=1.0)
    matplotlib.pyplot.xticks(list(range(0, len(CCs), 1)))
    matplotlib.pyplot.savefig("%s/CCs_to_benchmark.png" % (results_path))
    matplotlib.pyplot.close()
 