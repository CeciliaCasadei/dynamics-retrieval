# -*- coding: utf-8 -*-
import os

import joblib
import matplotlib.pyplot
import numpy
import settings_synthetic_data_spiral as settings

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
    x[1, :] = t * numpy.sin(t)  # +0.5*numpy.random.rand(S)

    matplotlib.pyplot.scatter(x[0, :], x[1, :], s=1, c=t)
    matplotlib.pyplot.savefig("%s/underlying_data.png" % results_path)
    matplotlib.pyplot.close()
    joblib.dump(x, "%s/underlying_data_%d.jbl" % (results_path, S))


# Input data
flag = 0
if flag == 1:
    S = settings.S
    m = settings.m
    t = numpy.linspace(0, 8 * numpy.pi / 2, num=S)
    jitter_std_dev = 0.10 * 8 * numpy.pi / 2
    t = t + numpy.random.normal(scale=jitter_std_dev, size=S)

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

    matplotlib.pyplot.scatter(x[0, :], x[1, :], s=1, c=t)
    matplotlib.pyplot.savefig("%s/input_data.png" % results_path)
    matplotlib.pyplot.close()

    joblib.dump(x, "%s/x.jbl" % results_path)
    joblib.dump(mask, "%s/mask.jbl" % results_path)

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
    modes = [0, 1]
    U = joblib.load("%s/U.jbl" % results_path)
    S = joblib.load("%s/S.jbl" % results_path)
    VH = joblib.load("%s/VT_final.jbl" % results_path)
    print(U.shape, S.shape, VH.shape)
    x_r_tot = 0
    for mode in modes:
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        print(u.shape, sv.shape, vT.shape)
        x_r = sv * numpy.outer(u, vT)
        print(x_r.shape)
        matplotlib.pyplot.scatter(x_r[0, :], x_r[1, :], s=1, c=list(range(x_r.shape[1])))
        matplotlib.pyplot.savefig(
            "%s/x_r_mode_%d.png" % (results_path, mode), dpi=96 * 3
        )
        matplotlib.pyplot.close()
        x_r_tot += x_r
    matplotlib.pyplot.scatter(x_r_tot[0, :], x_r_tot[1, :], s=1, c=list(range(x_r.shape[1])))
    matplotlib.pyplot.savefig("%s/x_r_tot.png" % (results_path), dpi=96 * 3)
    matplotlib.pyplot.close()

    x_underlying = joblib.load(
        "../../synthetic_data_spiral/underlying_data_%d.jbl" % settings.S
    )
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()

    CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot)
    print(CC)


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
    evecs = joblib.load("%s/evecs_sorted.jbl" % results_path)
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
        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
        print(x_r[0, :].shape)
        matplotlib.pyplot.scatter(
            x_r[0, :], x_r[1, :], s=1, c=numpy.asarray(list(range(S)))[q : S - q + 1]
        )
        matplotlib.pyplot.savefig(
            "%s/x_r_mode_%d.png" % (results_path, mode), dpi=96 * 3
        )
        matplotlib.pyplot.close()
        x_r_tot += x_r
    matplotlib.pyplot.scatter(
        x_r_tot[0, :], x_r_tot[1, :], s=1, c=numpy.asarray(list(range(S)))[q : S - q + 1]
    )
    matplotlib.pyplot.savefig("%s/x_r_tot.png" % (results_path), dpi=96 * 3)
    matplotlib.pyplot.close()

    x_underlying = joblib.load(
        "../../synthetic_data_spiral/underlying_data_%d.jbl" % S
    )[:, q : S - q + 1]
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()

    CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot)
    print(CC)

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

    q = settings.q

    x = joblib.load("%s/X_backward_q_%d.jbl" % (results_path, q))
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
    import dynamics_retrieval.Correlate

    S = settings.S
    q = settings.q

    x_r_tot = 0
    for mode in [0]:
        x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
        print(x_r[0, :].shape)
        matplotlib.pyplot.scatter(
            x_r[0, :], x_r[1, :], s=1, c=numpy.asarray(list(range(S)))[q - 1 : S - q + 1]
        )
        matplotlib.pyplot.savefig(
            "%s/x_r_mode_%d.png" % (results_path, mode), dpi=96 * 3
        )
        matplotlib.pyplot.close()
        x_r_tot += x_r
    matplotlib.pyplot.scatter(
        x_r_tot[0, :], x_r_tot[1, :], s=1, c=numpy.asarray(list(range(S)))[q - 1 : S - q + 1]
    )
    matplotlib.pyplot.savefig("%s/x_r_tot.png" % (results_path), dpi=96 * 3)
    matplotlib.pyplot.close()

    x_underlying = joblib.load(
        "../../synthetic_data_spiral/underlying_data_%d.jbl" % S
    )[:, q - 1 : S - q + 1]
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()

    CC = dynamics_retrieval.correlate.Correlate(x_underlying, x_r_tot)
    print(CC)
