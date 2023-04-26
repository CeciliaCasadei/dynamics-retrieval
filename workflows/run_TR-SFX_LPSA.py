# -*- coding: utf-8 -*-
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import joblib
import os
import matplotlib.pylab


### Prepare dataset ###

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.convert
    nlsa.convert.main(settings)

flag = 0 # Rho data
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.t_disorder_correct
    nlsa.t_disorder_correct.main(settings)

##### MERGE TEST #######
flag = 0
if flag == 1:
    import dynamics_retrieval.merge_test
    nlsa.merge_test.main()
########################

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.get_t_distribution
    nlsa.get_t_distribution.main(settings)

flag = 0 # bR data
if flag == 1:
    import settings_bR_light as settings
    import dynamics_retrieval.select_frames
    nlsa.select_frames.main(settings)

# Calculate intensity deviations from the mean
flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.calculate_dI
    nlsa.calculate_dI.main(settings)

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.boost
    nlsa.boost.main(settings)

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.mirror_dataset
    nlsa.mirror_dataset.main(settings)
    nlsa.mirror_dataset.make_virtual_ts(settings)

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    from scipy import sparse


    x_sp = joblib.load('%s/dT_bst_sparse_LTD_light_mirrored.jbl'%settings.results_path)
    print x_sp[1000,:]
    x = x_sp[:,:].todense()
    if(numpy.isnan(x).any()):
        print("x contain NaN values")
        N = numpy.count_nonzero(numpy.isnan(x))
        print 'N nans: ', N, 'out of ', x.shape
        x[numpy.isnan(x)] = 0
    else:
        print("x does not contain NaN values")

    M = joblib.load('%s/M_sparse_light.jbl'%(settings.results_path))
    M = M[:,:].todense()
    mysum = M.sum(axis=1)
    print 'M sum: ', mysum.shape
    idxs = numpy.argwhere(mysum==0)
    print 'N. zeros: ', idxs.shape[0]

    x_sparse = sparse.csr_matrix(x)
    joblib.dump(x_sparse, '%s/dT_bst_sparse_LTD_light_mirrored_nonans.jbl'%settings.results_path)



#################################
### LPSA PARA SEARCH : q-scan ###
#################################

qs = [12501, 17501]

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.make_settings
    for q in qs:

        # MAKE OUTPUT FOLDER
        q_path = '%s/f_max_%d_q_%d'%(settings.results_path, settings.f_max_q_scan, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)

        # MAKE SETTINGS FILE
        data_file = '%s/dT_bst_sparse_LTD_%s_mirrored_nonans.jbl'%(q_path, settings.label)
        fn = '/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_q_%d.py'%q
        nlsa.make_settings.main(settings, fn, q, settings.f_max_q_scan, q_path, data_file, min(20,2*settings.f_max_q_scan+1))

####################################
### LPSA PARA SEARCH : jmax-scan ###
####################################

f_max_s = [10, 15]

flag = 0
if flag == 1:
    import settings_rho_light_LPSA as settings
    import dynamics_retrieval.make_settings
    for f_max in f_max_s:

        # MAKE OUTPUT FOLDER
        f_max_path = '%s/f_max_%d_q_%d'%(settings.results_path, f_max, settings.q_f_max_scan)
        if not os.path.exists(f_max_path):
            os.mkdir(f_max_path)

        # MAKE SETTINGS FILE
        data_file = '%s/dT_bst_sparse_LTD_%s_mirrored_nonans.jbl'%(f_max_path, settings.label)
        fn = '/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_f_max_%d.py'%f_max
        nlsa.make_settings.main(settings, fn, settings.q_f_max_scan, f_max, f_max_path, data_file, min(20, 2*f_max+1))

############# START ##################

flag = 0
if flag == 1:
    import dynamics_retrieval.make_lp_filter
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        F = nlsa.make_lp_filter.get_F_sv_t_range(settings)
        Q, R = nlsa.make_lp_filter.on_qr(settings, F)
        d = nlsa.make_lp_filter.check_on(Q)
        print 'Normalisation: ', numpy.amax(abs(d))
        joblib.dump(Q, '%s/F_on.jbl'%settings.results_path)

flag = 0
if flag == 1:
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=100G --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s'
                  %(end_worker, settings.__name__))

# q = 10001
# modulename = 'settings_q_%d'%q
# settings = __import__(modulename)
# for i in range(40):
#     aj = joblib.load('%s/aj/a_%d.jbl'%(settings.results_path, i))
#     print aj.shape
#     print aj[40:50,]
#     if(numpy.isnan(aj).any()):
#         print("aj contain NaN values")
#         N = numpy.count_nonzero(numpy.isnan(aj))
#         print 'N nans: ', N, 'out of ', aj.shape
#     else:
#         print("aj does not contain NaN values")

flag = 0
if flag == 1:
    import dynamics_retrieval.merge_aj
    for f_max in [30]:#f_max_s:
        modulename = 'settings_f_max_%d'%f_max
    # for q in qs:
    #     modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.merge_aj.main(settings)

flag = 0
if flag == 1:
    # for f_max in [30]:#f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max

        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p hour --mem=40G --array=0-%d ../scripts_parallel_submission/run_parallel_ATA.sh %s'
                  %(end_worker, settings.__name__))

flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_ATA_merge
    # for f_max in [30]:#f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.calculate_ATA_merge.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD
    # for f_max in [25]:#f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.SVD.get_chronos(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs
    import dynamics_retrieval.plot_chronos
    # for f_max in [25]:#f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.plot_SVs.main(settings)
        nlsa.plot_chronos.main(settings)


flag = 0
if flag == 1:
    import dynamics_retrieval.SVD
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.SVD.get_topos(settings)

######################
# IF DATA ARE TOO BIG:
n_chuncks = 4

flag = 0 # Make Ai's from aj's
if flag == 1:
    import dynamics_retrieval.make_Ai
    # for f_max in [30]:#f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.make_Ai.make_Ai_f(settings, n_chuncks)

flag = 0 # Make Ai's from A
if flag == 1:
    import dynamics_retrieval.make_Ai
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
    # for q in qs:
    #     modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.make_Ai.get_Ai_f(settings, n_chuncks)

flag = 0 # Make Ui's (or uij's) from Ai's
if flag == 1:
    import dynamics_retrieval.make_Ui
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.make_Ui.make_Ui_f(settings, n_chuncks)

flag = 0 # Make uj's from uij's
if flag == 1:
    import dynamics_retrieval.make_Ui
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.make_Ui.make_uj(settings, n_chuncks)

flag = 0#TO DO # Make U from Ui's
if flag == 1:
    import dynamics_retrieval.make_Ui
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.make_Ui.make_U(settings, n_chuncks)
###################

# RECONSTRUCTION

# p-dependent reconstruction
flag = 0
if flag == 1:
    import dynamics_retrieval.reconstruct_p
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.reconstruct_p.f(settings)
        nlsa.reconstruct_p.f_ts(settings)

# Calculate L of p=0 reconstructed signal
# (ie L of central block of reconstructed supervectors)
flag = 0
if flag == 1:
    import dynamics_retrieval.local_linearity
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nlsa.local_linearity.get_L(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.add_modes
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
    # for q in qs:
    #     modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nmodes = 1
        nlsa.add_modes.f(settings, nmodes)

flag = 1
if flag == 1:
    import dynamics_retrieval.add_modes
    # for f_max in f_max_s:
    #     modulename = 'settings_f_max_%d'%f_max
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print 'jmax: ', settings.f_max
        nmodes = 1
        nlsa.add_modes.f_static_plus_dynamics(settings, nmodes)

flag = 0 # CMP SINGLE PIXEL RESULTS FROM LPSA AND BINNING
if flag == 1:
    import dynamics_retrieval.check_Bragg_reflections

    path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA/bR_light_dI'
    Bragg_i = 20000
    q = 20001
    p = 0
    f_max_s = [10, 20, 30, 40, 50]
    nmodes = 5
    bin_size = 10000

    nlsa.check_Bragg_reflections.f(path, Bragg_i, q, p, f_max_s, nmodes, bin_size)

# Calculate step-wise CC
# in the sequence of reconstructed signal (p=0)
# with progressively larger q.

flag = 0
if flag == 1:
    import dynamics_retrieval.get_successive_CCs
    import settings_bR_light as settings
    qs = [1, 1001, 2501, 5001, 7501, 10001, 12501, 15001, 17501, 20001, 22501, 25001]
    nmodes = 1
    nlsa.get_successive_CCs.get_CCs_q_scan(settings, qs, nmodes)

flag = 0
if flag == 1:
    import dynamics_retrieval.get_successive_CCs
    import settings_bR_light as settings
    qs = [1, 1001, 2501, 5001, 7501, 10001, 12501, 15001, 17501, 20001, 22501, 25001]
    nmodes = 1
    nlsa.get_successive_CCs.plot_CCs_q_scan(settings, qs, nmodes)

# USE test_cmp_reconstruction_times.py TO CMP RECONSTRUCTION TIMES FOR VARIOUS qs

# Calculate step-wise CC
# in the sequence of reconstructed signal
# with progressively larger fmax.

flag = 0
if flag == 1:
    import dynamics_retrieval.get_successive_CCs
    import settings_bR_light as settings
    f_max_s = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    nmodes = 1
    nlsa.get_successive_CCs.get_CCs_jmax_scan(settings, f_max_s, nmodes)

flag = 0
if flag == 1:
    import dynamics_retrieval.get_successive_CCs
    import settings_bR_light as settings
    f_max_s = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    nmodes = 1
    nlsa.get_successive_CCs.plot_CCs_jmax_scan(settings, f_max_s, nmodes)

###############################
### Standard reconstruction ###
###############################
flag = 0
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        end_worker = settings.n_workers_reconstruction - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=150G --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
                  %(end_worker, settings.__name__))

# TIME ASSIGNMENTwith p=(q-1)/2
flag = 0
if flag == 1:
    import dynamics_retrieval.reconstruct_p
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        nlsa.reconstruct_p.f_ts(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        for mode in settings.modes_to_reconstruct:
            nlsa.util_merge_x_r.f(settings, mode)

flag = 0 # CMP SINGLE PXL RESULTS VS p AND BINNING
if flag == 1:
    import dynamics_retrieval.check_Bragg_reflections

    path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA/bR_light_dI'
    Bragg_i = 21110
    q = 20001
    ps = [0, 10000]
    f_max_ = 50
    nmodes = 5
    bin_size = 10000

    nlsa.check_Bragg_reflections.f_ps(path, Bragg_i, q, ps, f_max, nmodes, bin_size)

# USE check_TRSFX_data.py TO CHECK SAMPLING RSATES AND I DISTRIBUTIONS