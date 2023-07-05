#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:49:34 2022

@author: casadei_c
"""
import math
import os

import joblib
import matplotlib.pyplot
import numpy


def make_setts(fn, S, m, q, f_max, path):
    paral_step_A = 400
    n_workers_A = int(math.ceil(float(q) / paral_step_A))
    ncopies = q
    paral_step_reconstruction = 2000
    n_workers_reconstruction = int(
        math.ceil(float(S - q - ncopies + 1) / paral_step_reconstruction)
    )
    data_file = "%s/x.jbl" % path

    fopen = open(fn, "w")
    fopen.write("# -*- coding: utf-8 -*-\n")
    fopen.write("import numpy\n")
    fopen.write("S = %d\n" % S)
    fopen.write("m = %d\n" % m)
    fopen.write("q = %d\n" % q)
    fopen.write("f_max = %d\n" % f_max)
    fopen.write("f_max_considered = f_max\n")
    fopen.write('results_path = "%s"\n' % path)
    fopen.write("paral_step_A = %d\n" % paral_step_A)
    fopen.write("datatype = numpy.float64\n")
    fopen.write('data_file = "%s"\n' % data_file)
    fopen.write("n_workers_A = %d\n" % n_workers_A)
    fopen.write("ncopies = %d\n" % ncopies)
    fopen.write("modes_to_reconstruct = range(20)\n")
    fopen.write("paral_step_reconstruction = %d\n" % paral_step_reconstruction)
    fopen.write("n_workers_reconstruction = %d\n" % n_workers_reconstruction)
    fopen.close()


root_path = "../../synthetic_data_5/test3/fourier_para_search"
m = 7000
S = 30000

#################################
### LPSA PARA SEARCH : q-scan ###
#################################
qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
f_max = 100
flag = 0
if flag == 1:
    for q in qs:

        # MAKE OUTPUT FOLDER
        q_path = "%s/f_max_%d_q_%d" % (root_path, f_max, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)

        # MAKE SETTINGS FILE
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_q_%d.py"
            % q
        )
        make_setts(fn, S, m, q, f_max, q_path)


####################################
### LPSA PARA SEARCH : jmax-scan ###
####################################
q = 1
f_max_s = [1, 5, 10, 50, 100, 150, 300]

flag = 0
if flag == 1:
    for f_max in f_max_s:

        # MAKE OUTPUT FOLDER
        f_max_path = "%s/f_max_%d_q_%d" % (root_path, f_max, q)
        if not os.path.exists(f_max_path):
            os.mkdir(f_max_path)

        # MAKE SETTINGS FILE
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_f_max_%d.py"
            % f_max
        )
        make_setts(fn, S, m, q, f_max, f_max_path)


#### START
flag = 0
if flag == 1:
    import dynamics_retrieval.make_lp_filter

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)
        print(settings.results_path)

        dynamics_retrieval.make_lp_filter.main(settings)


flag = 0
if flag == 1:
    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)
        print(settings.results_path)

        end_worker = settings.n_workers_A - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s"
            % (end_worker, settings.__name__)
        )


flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_A

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)
        print(settings.results_path)

        dynamics_retrieval.util_merge_A.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    print("\n****** RUNNING SVD ******")
    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        results_path = settings.results_path
        datatype = settings.datatype

        A = joblib.load("%s/A_parallel.jbl" % results_path)
        print("Loaded")
        U, S, VH = dynamics_retrieval.SVD.SVD_f_manual(A)
        U, S, VH = dynamics_retrieval.SVD.sorting(U, S, VH)

        print("Done")
        print("U: ", U.shape)
        print("S: ", S.shape)
        print("VH: ", VH.shape)

        joblib.dump(U, "%s/U.jbl" % results_path)
        joblib.dump(S, "%s/S.jbl" % results_path)
        joblib.dump(VH, "%s/VH.jbl" % results_path)

        evecs = joblib.load("%s/F_on_qr.jbl" % (results_path))
        Phi = evecs[:, 0 : 2 * settings.f_max_considered + 1]

        VT_final = dynamics_retrieval.SVD.project_chronos(VH, Phi)
        print("VT_final: ", VT_final.shape)
        joblib.dump(VT_final, "%s/VT_final.jbl" % results_path)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_chronos
    import dynamics_retrieval.plot_SVs

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        dynamics_retrieval.plot_SVs.main(settings)
        dynamics_retrieval.plot_chronos.main(settings)


flag = 0
if flag == 1:
    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        end_worker = settings.n_workers_reconstruction - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
            % (end_worker, settings.__name__)
        )


flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        for mode in settings.modes_to_reconstruct:
            dynamics_retrieval.util_merge_x_r.f(settings, mode)


flag = 0
if flag == 1:
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        rpath = settings.results_path
        q = settings.q
        S = settings.S
        m = settings.m
        ncopies = settings.ncopies

        benchmark = joblib.load("../../synthetic_data_5/test1/x.jbl")
        benchmark = benchmark[:, q : q + (S - q - ncopies + 1)]
        benchmark = benchmark.flatten()

        CCs = []

        x_r_tot = 0
        for mode in settings.modes_to_reconstruct:
            print(mode)
            x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (rpath, mode))

            x_r_large = numpy.zeros((m, S))
            x_r_large[:] = numpy.nan
            x_r_large[:, q : q + (S - q - ncopies + 1)] = x_r

            dynamics_retrieval.plot_syn_data.f(
                x_r_large, "%s/x_r_mode_%d.png" % (rpath, mode)
            )

            x_r_tot += x_r
            x_r_tot_flat = x_r_tot.flatten()
            CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
            CCs.append(CC)

            x_r_large = numpy.zeros((m, S))
            x_r_large[:] = numpy.nan
            x_r_large[:, q : q + (S - q - ncopies + 1)] = x_r_tot

            dynamics_retrieval.plot_syn_data.f(
                x_r_large,
                "%s/x_r_tot_%d_modes.png" % (rpath, mode + 1),
                title="%.4f" % CC,
            )

        joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % rpath)


flag = 0
if flag == 1:
    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        CCs = joblib.load("%s/reconstruction_CC_vs_nmodes.jbl" % settings.results_path)
        matplotlib.pyplot.scatter(list(range(1, len(CCs) + 1)), CCs, c="b")
        matplotlib.pyplot.xticks(list(range(1, len(CCs) + 1, 2)))
        matplotlib.pyplot.savefig(
            "%s/reconstruction_CC_vs_nmodes.png" % (settings.results_path)
        )
        matplotlib.pyplot.close()

flag = 0
if flag == 1:
    import dynamics_retrieval.local_linearity

    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        local_linearity_lst = []
        x_r_tot = 0
        for mode in settings.modes_to_reconstruct:
            print("mode: ", mode)
            x_r = joblib.load(
                "%s/movie_mode_%d_parallel.jbl" % (settings.results_path, mode)
            )
            x_r_tot += x_r
            L = dynamics_retrieval.local_linearity.local_linearity_measure(x_r_tot)

            local_linearity_lst.append(L)
        joblib.dump(
            local_linearity_lst,
            "%s/local_linearity_vs_nmodes.jbl" % settings.results_path,
        )

flag = 0
if flag == 1:
    for q in qs:
        modulename = "settings_q_%d" % q
        # for f_max in f_max_s:
        #     modulename = 'settings_f_max_%d'%f_max

        settings = __import__(modulename)

        print(settings.q)
        print(settings.f_max)

        lls = joblib.load("%s/local_linearity_vs_nmodes.jbl" % settings.results_path)
        matplotlib.pyplot.scatter(list(range(1, len(lls) + 1)), numpy.log(lls), c="b")
        matplotlib.pyplot.xticks(list(range(1, len(lls) + 1, 2)))
        matplotlib.pyplot.savefig(
            "%s/local_linearity_vs_nmodes.png" % (settings.results_path)
        )
        matplotlib.pyplot.close()
