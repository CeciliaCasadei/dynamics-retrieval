# -*- coding: utf-8 -*-
import os
import time

import joblib
import matplotlib.pyplot
import numpy
import settings_synthetic_data as settings


def make_x(settings):
    m = settings.m
    S = settings.S
    q = settings.q
    results_path = settings.results_path

    x = numpy.zeros((m, S))
    mask = numpy.zeros((m, S))

    ts = numpy.asarray(list(range(S)), dtype=numpy.double)
    T = S - q
    omega = 2 * numpy.pi / T

    min_period = T / 11
    jitter_factor = 0.3
    jitter_std_dev = jitter_factor * min_period
    ts = ts + numpy.random.normal(scale=jitter_std_dev, size=S)

    tc = float(S) / 2
    e1 = 1 - numpy.exp(-ts / tc)
    e2 = 1 - e1

    for i in range(m):

        A_i = numpy.cos(0.6 * (2 * numpy.pi / m) * i)
        B_i = numpy.sin(3 * (2 * numpy.pi / m) * i + numpy.pi / 5)
        C_i = numpy.sin(0.8 * (2 * numpy.pi / m) * i + numpy.pi / 7)
        D_i = numpy.cos(2.1 * (2 * numpy.pi / m) * i)
        E_i = numpy.cos(1.2 * (2 * numpy.pi / m) * i + numpy.pi / 10)
        F_i = numpy.sin(1.8 * (2 * numpy.pi / m) * i + numpy.pi / 11)
        x_i = e1 * (
            A_i + B_i * numpy.cos(3 * omega * ts) + C_i * numpy.sin(10 * omega * ts)
        ) + e2 * (
            D_i
            + E_i * numpy.sin(7 * omega * ts)
            + F_i * numpy.sin(11 * omega * ts + numpy.pi / 10)
        )

        partialities = numpy.random.rand(S)
        x_i = x_i * partialities

        sparsities = numpy.random.rand(S)
        thr = 0.982
        sparsities[sparsities < thr] = 0
        sparsities[sparsities >= thr] = 1
        x_i = x_i * sparsities

        mask[i, :] = sparsities
        x[i, :] = x_i

    matplotlib.pyplot.imshow(x, cmap="jet")
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig(
        "%s/x_jitter_factor_%.2f.png" % (results_path, jitter_factor), dpi=96 * 3
    )
    matplotlib.pyplot.close()

    joblib.dump(x, "%s/x.jbl" % results_path)
    joblib.dump(mask, "%s/mask.jbl" % results_path)


########################
##### Make dataset #####
########################

flag = 0
if flag == 1:
    make_x(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_syn_data

    x = joblib.load("%s/x.jbl" % settings.results_path)
    fn = "%s/x.png" % (settings.results_path)
    dynamics_retrieval.plot_syn_data.f(x, fn)

################
###   NLSA   ###
################

flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_distances_utilities

    distance_mode = "onlymeasured_normalised"
    if distance_mode == "allterms":
        dynamics_retrieval.calculate_distances_utilities.calculate_d_sq_dense(settings)
    elif distance_mode == "onlymeasured":
        dynamics_retrieval.calculate_distances_utilities.calculate_d_sq_sparse(settings)
    elif distance_mode == "onlymeasured_normalised":
        dynamics_retrieval.calculate_distances_utilities.calculate_d_sq_SFX_element_n(
            settings
        )
        dynamics_retrieval.calculate_distances_utilities.calculate_d_sq_sparse(settings)
    else:
        print("Undefined distance mode.")

flag = 0
if flag == 1:
    d_sq = joblib.load("%s/d_sq.jbl" % (settings.results_path))
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print(numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_distance_distributions

    dynamics_retrieval.plot_distance_distributions.plot_d_0j(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel.sh %s"
        % (end_worker, settings.__name__)
    )
    os.system(
        "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_n_Dsq_elements.sh %s"
        % (end_worker, settings.__name__)
    )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_D_sq

    dynamics_retrieval.util_merge_D_sq.f(settings)
    dynamics_retrieval.util_merge_D_sq.f_N_D_sq_elements(settings)
    import dynamics_retrieval.calculate_distances_utilities

    dynamics_retrieval.calculate_distances_utilities.normalise(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_distance_distributions

    dynamics_retrieval.plot_distance_distributions.plot_D_0j(settings)

# Select b euclidean nns or b time nns.
flag = 0
if flag == 1:
    # import dynamics_retrieval.calculate_distances_utilities
    # dynamics_retrieval.calculate_distances_utilities.sort_D_sq(settings)
    import dynamics_retrieval.get_D_N

    dynamics_retrieval.get_D_N.main_euclidean_nn(settings)
    # dynamics_retrieval.get_D_N.main_time_nn_1(settings)
    # dynamics_retrieval.get_D_N.main_time_nn_2(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.get_epsilon

    dynamics_retrieval.get_epsilon.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.transition_matrix

    dynamics_retrieval.transition_matrix.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.probability_matrix

    dynamics_retrieval.probability_matrix.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.eigendecompose

    dynamics_retrieval.eigendecompose.main(settings)

flag = 0
if flag == 1:
    evecs = joblib.load("%s/evecs_sorted.jbl" % settings.results_path)
    test = numpy.matmul(evecs.T, evecs)
    diff = abs(test - numpy.eye(settings.l))
    print(numpy.amax(diff))

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_P_evecs

    dynamics_retrieval.plot_P_evecs.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_A - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s"
        % (end_worker, settings.__name__)
    )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_A

    dynamics_retrieval.util_merge_A.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    dynamics_retrieval.SVD.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
        % (end_worker, settings.__name__)
    )
    # os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction_onecopy.sh %s'
    #          %(end_worker, settings.__name__))

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    for mode in settings.modes_to_reconstruct:
        dynamics_retrieval.util_merge_x_r.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    results_path = settings.results_path
    S = settings.S
    q = settings.q
    ncopies = settings.ncopies

    benchmark = joblib.load("../../synthetic_data_4/test5/x.jbl")
    benchmark = benchmark[:, q : q + (S - q - ncopies + 1)]
    benchmark = benchmark.flatten()

    CCs = []
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print(mode)

        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
        dynamics_retrieval.plot_syn_data.f(
            x_r, "%s/x_r_mode_%d.png" % (results_path, mode)
        )

        x_r_tot += x_r
        x_r_tot_flat = x_r_tot.flatten()
        CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
        CCs.append(CC)

        dynamics_retrieval.plot_syn_data.f(
            x_r_tot,
            "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
            title="%.4f" % CC,
        )

    joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % results_path)


#############################
###     Plain SVD of x    ###
#############################

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    x = joblib.load("%s/x.jbl" % settings.results_path)
    U, S, VH = dynamics_retrieval.SVD.SVD_f(x)
    U, S, VH = dynamics_retrieval.SVD.sorting(U, S, VH)

    print("Done")
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("VH: ", VH.shape)

    joblib.dump(U, "%s/U.jbl" % settings.results_path)
    joblib.dump(S, "%s/S.jbl" % settings.results_path)
    joblib.dump(VH, "%s/VT_final.jbl" % settings.results_path)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    results_path = settings.results_path

    benchmark = joblib.load("../../synthetic_data_4/test5/x.jbl")
    benchmark = benchmark.flatten()

    U = joblib.load("%s/U.jbl" % results_path)
    S = joblib.load("%s/S.jbl" % results_path)
    VH = joblib.load("%s/VT_final.jbl" % results_path)

    CCs = []
    x_r_tot = 0
    modes = list(range(20))
    for mode in modes:

        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]

        x_r = sv * numpy.outer(u, vT)
        dynamics_retrieval.plot_syn_data.f(
            x_r, "%s/x_r_mode_%d.png" % (results_path, mode)
        )

        x_r_tot += x_r
        x_r_tot_flat = x_r_tot.flatten()
        CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
        CCs.append(CC)

        dynamics_retrieval.plot_syn_data.f(
            x_r_tot,
            "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
            title="%.4f" % CC,
        )

    joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % results_path)

#########################
######   LPSA   #########
#########################

flag = 0
if flag == 1:
    import dynamics_retrieval.make_lp_filter

    dynamics_retrieval.make_lp_filter.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_A - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s"
        % (end_worker, settings.__name__)
    )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_A

    dynamics_retrieval.util_merge_A.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    print("\n****** RUNNING SVD ******")

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
    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
        % (end_worker, settings.__name__)
    )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    for mode in settings.modes_to_reconstruct:
        dynamics_retrieval.util_merge_x_r.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    results_path = settings.results_path
    S = settings.S
    q = settings.q
    ncopies = settings.ncopies

    benchmark = joblib.load("../../synthetic_data_4/test5/x.jbl")
    benchmark = benchmark[:, q : q + (S - q - ncopies + 1)]
    benchmark = benchmark.flatten()

    CCs = []
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:

        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
        dynamics_retrieval.plot_syn_data.f(
            x_r, "%s/x_r_mode_%d.png" % (results_path, mode)
        )

        x_r_tot += x_r
        x_r_tot_flat = x_r_tot.flatten()
        CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
        CCs.append(CC)
        dynamics_retrieval.plot_syn_data.f(
            x_r_tot,
            "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
            title="%.4f" % CC,
        )

    joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % results_path)


###################
######  SSA  ######
###################

# Calc xTx
flag = 0
if flag == 1:
    x = joblib.load("%s/x.jbl" % settings.results_path)
    print("x: ", x.shape)
    xTx = numpy.matmul(x.T, x)
    joblib.dump(xTx, "%s/xTx.jbl" % settings.results_path)
    print("xTx: ", xTx.shape)

# Calc XTX
flag = 0
if flag == 1:
    s = settings.S - settings.q
    XTX = numpy.zeros((s, s))
    xTx = joblib.load("%s/xTx.jbl" % settings.results_path)
    print("xTx: ", xTx.shape, "XTX: ", XTX.shape)
    start = time.time()
    for i in range(1, settings.q + 1):  # Time ~q seconds
        print(i)
        XTX += xTx[i : i + s, i : i + s]
    print("Time: ", time.time() - start)
    joblib.dump(XTX, "%s/XTX.jbl" % settings.results_path)

# SVD XTX
flag = 0
if flag == 1:
    XTX = joblib.load("%s/XTX.jbl" % settings.results_path)
    evals_XTX, evecs_XTX = numpy.linalg.eigh(XTX)
    print("Done")
    evals_XTX[numpy.argwhere(evals_XTX < 0)] = 0
    SVs = numpy.sqrt(evals_XTX)
    VT = evecs_XTX.T
    print("Sorting")
    sort_idxs = numpy.argsort(SVs)[::-1]
    SVs_sorted = SVs[sort_idxs]
    VT_sorted = VT[sort_idxs, :]
    joblib.dump(SVs_sorted, "%s/S.jbl" % settings.results_path)
    joblib.dump(VT_sorted, "%s/VT_final.jbl" % settings.results_path)

flag = 0
if flag == 1:
    SVs = joblib.load("%s/S.jbl" % settings.results_path)
    VT = joblib.load("%s/VT_final.jbl" % settings.results_path)
    U_temp = numpy.matmul(VT.T, numpy.diag(1.0 / SVs))
    print("VS-1: ", U_temp.shape)
    U_temp = U_temp[:, 0:20]
    x = joblib.load("%s/x.jbl" % settings.results_path)
    m = x.shape[0]
    q = settings.q
    s = x.shape[1] - q
    U = numpy.zeros((m * q, 20))
    start = time.time()
    for j in range(0, q):
        U[j * m : (j + 1) * m, :] = numpy.matmul(x[:, q - j : q + s - j], U_temp)
    # U = numpy.matmul(A, U_temp)
    print("Time: ", time.time() - start)
    joblib.dump(U, "%s/U.jbl" % settings.results_path)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.plot(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system(
        "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
        % (end_worker, settings.__name__)
    )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    for mode in range(20):
        dynamics_retrieval.util_merge_x_r.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    results_path = settings.results_path
    S = settings.S
    q = settings.q
    ncopies = settings.ncopies

    benchmark = joblib.load("../../synthetic_data_4/test5/x.jbl")
    benchmark = benchmark[:, q : q + (S - q - ncopies + 1)]
    print(benchmark.shape)
    benchmark = benchmark.flatten()

    CCs = []

    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print("Mode:", mode)
        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
        dynamics_retrieval.plot_syn_data.f(
            x_r, "%s/x_r_mode_%d.png" % (results_path, mode)
        )

        x_r_tot += x_r
        x_r_tot_flat = x_r_tot.flatten()
        CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
        CCs.append(CC)
        dynamics_retrieval.plot_syn_data.f(
            x_r_tot,
            "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
            title="%.4f" % CC,
        )

    joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % settings.results_path)

flag = 0
if flag == 1:
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
        local_linearity_lst, "%s/local_linearity_vs_nmodes.jbl" % settings.results_path
    )

flag = 0
if flag == 1:

    lls = joblib.load("%s/local_linearity_vs_nmodes.jbl" % settings.results_path)
    matplotlib.pyplot.scatter(list(range(1, len(lls) + 1)), numpy.log(lls), c="b")
    matplotlib.pyplot.xticks(list(range(1, len(lls) + 1, 2)))
    matplotlib.pyplot.savefig(
        "%s/local_linearity_vs_nmodes.png" % (settings.results_path)
    )
    matplotlib.pyplot.close()


# BINNING
flag = 0
if flag == 1:
    import dynamics_retrieval.binning

    dynamics_retrieval.binning.binning_f(settings)
    dynamics_retrieval.binning.plot(settings)
