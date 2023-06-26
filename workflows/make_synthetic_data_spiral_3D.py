# -*- coding: utf-8 -*-
import os

import joblib
import matplotlib.pyplot as plt
import numpy
import settings_synthetic_data_spiral_3D as settings
from mpl_toolkits.mplot3d import Axes3D

results_path = settings.results_path

### Test spiral
# Ideal data
flag = 0
if flag == 1:
    S = settings.S
    m = settings.m
    t = numpy.linspace(0, 8 * numpy.pi / 2, num=S)

    x = numpy.zeros((m, S))

    x[0, :] = t * numpy.cos(t)
    x[1, :] = t * numpy.sin(t)
    x[2, :] = 0.1 * x[1, :]

    fig = plt.figure()
    plt.rcParams["grid.color"] = "white"
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x[0, :], x[1, :], x[2, :], c=t, edgecolors="none")

    ax.w_xaxis.line.set_color("b")
    ax.w_yaxis.line.set_color("b")
    ax.w_zaxis.line.set_color("b")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
    ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
    ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

    plt.savefig("%s/test_underlying_data.pdf" % results_path, dpi=96 * 4)
    joblib.dump(x, "%s/underlying_data.jbl" % results_path)


# Input data
flag = 1
if flag == 1:
    S = settings.S
    m = settings.m
    t = numpy.linspace(0, 8 * numpy.pi / 2, num=S)
    # jitter_std_dev = 0.10*8*numpy.pi/2
    # t = t + numpy.random.normal(scale=jitter_std_dev, size=S)

    x = numpy.zeros((m, S))
    mask = numpy.zeros((m, S))

    thr = 0.8

    x_0 = t * numpy.cos(t)  # +0.5*numpy.random.rand(S)
    sparsities = numpy.random.rand(S)
    sparsities[sparsities < thr] = 0
    sparsities[sparsities >= thr] = 1
    partialities = numpy.random.rand(S)
    x_0 = x_0 * partialities
    x_0 = x_0 * sparsities
    mask[0, :] = sparsities
    x[0, :] = x_0

    x_1 = t * numpy.sin(t)  # +0.5*numpy.random.rand(S)
    sparsities = numpy.random.rand(S)
    sparsities[sparsities < thr] = 0
    sparsities[sparsities >= thr] = 1
    partialities = numpy.random.rand(S)
    x_1 = x_1 * partialities
    x_1 = x_1 * sparsities
    mask[1, :] = sparsities
    x[1, :] = x_1

    x_2 = 0.1 * t * numpy.sin(t)
    sparsities = numpy.random.rand(S)
    sparsities[sparsities < thr] = 0
    sparsities[sparsities >= thr] = 1
    partialities = numpy.random.rand(S)
    x_2 = x_2 * partialities
    x_2 = x_2 * sparsities
    mask[2, :] = sparsities
    x[2, :] = x_2

    x = joblib.load("%s/x.jbl" % settings.results_path)
    fig = plt.figure()
    plt.rcParams["grid.color"] = "white"
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x[0, :], x[1, :], x[2, :], c=t, edgecolors="none")

    ax.w_xaxis.line.set_color("b")
    ax.w_yaxis.line.set_color("b")
    ax.w_zaxis.line.set_color("b")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
    ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
    ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

    plt.savefig("%s/input_data.pdf" % settings.results_path, dpi=96 * 4)

    # joblib.dump(x, '%s/x.jbl'%settings.results_path)
    # joblib.dump(mask, '%s/mask.jbl'%settings.results_path)

#############################
###     Plain SVD of x    ###
#############################

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    x = joblib.load("%s/x.jbl" % results_path)
    U, S, VH = dynamics_retrieval.SVD.SVD_f(x)
    U, S, VH = dynamics_retrieval.SVD.sorting(U, S, VH)

    print("Done")
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("VH: ", VH.shape)

    joblib.dump(U, "%s/U.jbl" % results_path)
    joblib.dump(S, "%s/S.jbl" % results_path)
    joblib.dump(VH, "%s/VT_final.jbl" % results_path)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate

    # reconstruct
    modes = [0, 1, 2]
    U = joblib.load("%s/U.jbl" % results_path)
    S = joblib.load("%s/S.jbl" % results_path)
    VH = joblib.load("%s/VT_final.jbl" % results_path)
    print(U.shape, S.shape, VH.shape)
    x_r_tot = 0
    for mode in modes:
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        x_r = sv * numpy.outer(u, vT)

        fig = plt.figure()
        plt.rcParams["grid.color"] = "white"
        ax = fig.add_subplot(projection="3d")

        ax.scatter(
            x_r[0, :],
            x_r[1, :],
            x_r[2, :],
            c=numpy.linspace(0, 8 * numpy.pi / 2, num=settings.S),
            edgecolors="none",
        )

        ax.w_xaxis.line.set_color("b")
        ax.w_yaxis.line.set_color("b")
        ax.w_zaxis.line.set_color("b")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        plt.savefig("%s/x_r_mode_%d.pdf" % (results_path, mode), dpi=96 * 3)
        plt.close()

        x_r_tot += x_r

        fig = plt.figure()
        plt.rcParams["grid.color"] = "white"
        ax = fig.add_subplot(projection="3d")

        ax.scatter(
            x_r_tot[0, :],
            x_r_tot[1, :],
            x_r_tot[2, :],
            c=numpy.linspace(0, 8 * numpy.pi / 2, num=settings.S),
            edgecolors="none",
        )

        ax.w_xaxis.line.set_color("b")
        ax.w_yaxis.line.set_color("b")
        ax.w_zaxis.line.set_color("b")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        x_underlying = joblib.load(
            "../../synthetic_data_spiral_3D/test1/underlying_data.jbl"
        )
        x_r_tot_flat = x_r_tot.flatten()
        x_underlying = x_underlying.flatten()

        CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot_flat)
        print (CC)

        plt.title("CC: %.4f" % CC)

        plt.savefig("%s/x_r_%d_modes.pdf" % (results_path, mode + 1), dpi=96 * 3)
        plt.close()


#############################
###         NLSA          ###
#############################

# NO LP FILTERING
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

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    for mode in settings.modes_to_reconstruct:
        dynamics_retrieval.util_merge_x_r.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.correlate

    S = settings.S
    q = settings.q
    x_r_tot = 0
    for mode in [0]:
        x_r = joblib.load(
            "%s/movie_mode_%d_parallel.jbl" % (settings.results_path, mode)
        )
        print(x_r[0, :].shape)
        plt.scatter(x_r[0, :], x_r[1, :], s=1, c=numpy.asarray(list(range(S)))[q : S - q + 1])
        plt.savefig("%s/x_r_mode_%d.png" % (results_path, mode), dpi=96 * 3)
        plt.close()
        x_r_tot += x_r
    plt.scatter(
        x_r_tot[0, :], x_r_tot[1, :], s=1, c=numpy.asarray(list(range(S)))[q : S - q + 1]
    )
    plt.savefig("%s/x_r_tot.png" % (settings.results_path), dpi=96 * 3)
    plt.close()

    x_underlying = joblib.load(
        "../../synthetic_data_spiral/underlying_data_%d.jbl" % S
    )[:, q : S - q + 1]
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()

    CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot)
    print (CC)

#############################
###           SSA         ###
#############################

flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_distances_utilities

    dynamics_retrieval.calculate_distances_utilities.concatenate_backward(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    x = joblib.load("%s/X_backward_q_%d.jbl" % (results_path, settings.q))
    U, S, VH = dynamics_retrieval.SVD.SVD_f(x)
    U, S, VH = dynamics_retrieval.SVD.sorting(U, S, VH)

    print("Done")
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("VH: ", VH.shape)

    joblib.dump(U, "%s/U.jbl" % results_path)
    joblib.dump(S, "%s/S.jbl" % results_path)
    joblib.dump(VH, "%s/VT_final.jbl" % results_path)

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

    x_r_tot = 0
    S = settings.S
    q = settings.q
    for mode in settings.modes_to_reconstruct:
        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))

        fig = plt.figure()
        plt.rcParams["grid.color"] = "white"
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            x_r[0, :],
            x_r[1, :],
            x_r[2, :],
            c=numpy.linspace(0, 8 * numpy.pi / 2, num=S)[q - 1 : S - q + 1],
            edgecolors="none",
        )

        ax.w_xaxis.line.set_color("b")
        ax.w_yaxis.line.set_color("b")
        ax.w_zaxis.line.set_color("b")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        plt.savefig("%s/x_r_mode_%d.pdf" % (results_path, mode), dpi=96 * 3)
        plt.close()

        x_r_tot += x_r

        fig = plt.figure()
        plt.rcParams["grid.color"] = "white"
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            x_r_tot[0, :],
            x_r_tot[1, :],
            x_r_tot[2, :],
            c=numpy.linspace(0, 8 * numpy.pi / 2, num=S)[q - 1 : S - q + 1],
            edgecolors="none",
        )

        ax.w_xaxis.line.set_color("b")
        ax.w_yaxis.line.set_color("b")
        ax.w_zaxis.line.set_color("b")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        ax.w_xaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_yaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))
        ax.w_zaxis.set_pane_color((0.1, 0.0, 1.0, 0.08))

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        x_underlying = joblib.load(
            "../../synthetic_data_spiral_3D/test1/underlying_data.jbl"
        )[:, q - 1 : S - q + 1]
        x_r_tot_flat = x_r_tot.flatten()
        x_underlying = x_underlying.flatten()

        CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot_flat)
        print (CC)

        plt.title("CC: %.4f" % CC)

        plt.savefig("%s/x_r_%d_modes.pdf" % (results_path, mode + 1), dpi=96 * 3)
        plt.close()
