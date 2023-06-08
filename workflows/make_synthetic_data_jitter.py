# -*- coding: utf-8 -*-
import matplotlib
import numpy

matplotlib.use("Agg")
import os

import joblib
import matplotlib.pylab
import matplotlib.pyplot

# USE: conda activate myenv_nlsa


def make_settings(sett, fn, q, fmax, path, nmodes):
    fopen = open(fn, "w")
    fopen.write("# -*- coding: utf-8 -*-\n")
    fopen.write("import numpy\n")
    fopen.write("import math\n")
    fopen.write("S = %d\n" % sett.S)
    fopen.write("m = %d\n" % sett.m)
    fopen.write("q = %d\n" % q)
    fopen.write("f_max = %d\n" % fmax)
    fopen.write("f_max_considered = f_max\n")
    fopen.write('results_path = "%s"\n' % path)
    fopen.write("paral_step_A = 500\n")
    fopen.write("datatype = numpy.float64\n")
    fopen.write('data_file = "%s/T_bst_sparse.jbl"\n' % path)
    fopen.write("n_workers_A = int(math.ceil(float(q)/paral_step_A))\n")
    fopen.write("p = 0\n")
    fopen.write("modes_to_reconstruct = range(0, %d)\n" % nmodes)
    fopen.close()


def model(settings, i, ts):
    m = settings.m
    T = settings.T_model
    omega = 2 * numpy.pi / T
    tc = settings.tc

    e1 = 1 - numpy.exp(-ts / tc)
    e2 = 1 - e1

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

    # Pedestal
    pedestal = 3.0 + float(i) / 600
    print("Pedestal:", pedestal)
    x_i = x_i + pedestal

    return x_i


def eval_model(settings, times):
    m = settings.m
    x = numpy.zeros((m, times.shape[0]))
    for i in range(m):
        x[i, :] = model(settings, i, times)
    return x


def make_x(settings):
    m = settings.m
    S = settings.S
    T = settings.T_model
    jitter_factor = settings.jitter_factor
    results_path = settings.results_path
    print("Jitter factor", jitter_factor)

    x = numpy.zeros((m, S))
    mask = numpy.zeros((m, S))

    ts_meas = numpy.sort(S * numpy.random.rand(S))

    # Jitter
    min_period = T / 11
    jitter_stdev = jitter_factor * min_period
    ts_true = ts_meas + numpy.random.normal(loc=0.0, scale=jitter_stdev, size=S)

    # Pixel-dependent sparsities from bR
    thrs = joblib.load("%s/sparsity_thrs.jbl" % results_path)

    for i in range(m):
        # With pedestal
        x_i = model(settings, i, ts_true)

        # Partiality
        partialities = numpy.random.rand(S)
        x_i = x_i * partialities

        # Gaussian noise
        # print 'Gaussian noise'
        # x_i = numpy.random.normal(loc=x_i, scale=numpy.sqrt(abs(x_i)))

        # Sparsity
        sparsities = numpy.random.rand(S)
        thr = thrs[i]  # 0.9998 #0.9990 #0.982
        print(i, "Sparsity thr:", thr)
        sparsities[sparsities < thr] = 0
        sparsities[sparsities >= thr] = 1
        x_i = x_i * sparsities

        mask[i, :] = sparsities
        x[i, :] = x_i

    fn = "%s/x_jitter_factor_%0.2f.png" % (results_path, jitter_factor)
    dynamics_retrieval.plot_syn_data.f(x, fn)
    fn = "%s/x_underlying_ts_meas.png" % (results_path)
    dynamics_retrieval.plot_syn_data.f(eval_model(settings, ts_meas), fn)

    joblib.dump(x, "%s/x.jbl" % results_path)
    joblib.dump(mask, "%s/mask.jbl" % results_path)
    joblib.dump(ts_true, "%s/ts_true.jbl" % results_path)
    joblib.dump(ts_meas, "%s/ts_meas.jbl" % results_path)


##############################
### Make synthetic dataset ###
##############################

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    make_x(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.boost

    dynamics_retrieval.boost.main_syn_data(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.calculate_dI

    dynamics_retrieval.calculate_dI.main(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.boost

    dynamics_retrieval.boost.main(settings)

#################################
### LPSA PARA SEARCH : q-scan ###
#################################

qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    for q in qs:

        # MAKE OUTPUT FOLDER
        q_path = "%s/f_max_%d_q_%d" % (settings.results_path, settings.f_max_q_scan, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)

        # MAKE SETTINGS FILE
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/nlsa/settings_q_%d.py"
            % q
        )
        make_settings(settings, fn, q, settings.f_max_q_scan, q_path, 20)

####################################
### LPSA PARA SEARCH : jmax-scan ###
####################################

f_max_s = [100]

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

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
        fn = (
            "/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/nlsa/settings_f_max_%d.py"
            % f_max
        )
        make_settings(
            settings,
            fn,
            settings.q_f_max_scan,
            f_max,
            f_max_path,
            min(20, 2 * f_max + 1),
        )


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
    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s[-1:]:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)

        end_worker = settings.n_workers_A - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s"
            % (end_worker, settings.__name__)
        )

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_A

    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s[-1:]:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("q: ", settings.q)
        print("jmax: ", settings.f_max)

        dynamics_retrieval.util_merge_A.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD

    print("\n****** RUNNING SVD ******")
    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)

        results_path = settings.results_path

        A = joblib.load("%s/A_parallel.jbl" % results_path)[
            :, 0 : 2 * settings.f_max + 1
        ]
        print("Loaded")
        U, S, VH = dynamics_retrieval.SVD.SVD_f_manual(A)
        U, S, VH = dynamics_retrieval.SVD.sorting(U, S, VH)

        print("Done")
        print("U: ", U.shape)
        print("S: ", S.shape)
        print("VH: ", VH.shape)

        joblib.dump(U[:, 0:20], "%s/U.jbl" % results_path)
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

    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)

        dynamics_retrieval.plot_SVs.main(settings)
        dynamics_retrieval.plot_chronos.main(settings)

# p=0 reconstruction
flag = 0
if flag == 1:
    import dynamics_retrieval.reconstruct_p

    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)

        dynamics_retrieval.reconstruct_p.f(settings)
        dynamics_retrieval.reconstruct_p.f_ts(settings)

# Calculate L of p=0 reconstructed signal (ie L of central block of reconstreucted supervectors)
flag = 0
if flag == 1:
    import dynamics_retrieval.local_linearity

    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)

        dynamics_retrieval.local_linearity.get_L(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_syn_data

    # for q in qs:
    #     modulename = 'settings_q_%d'%q
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)

        m = settings.m
        S = settings.S
        q = settings.q
        p = settings.p
        results_path = "%s/reconstruction_p_%d" % (settings.results_path, p)

        print("jmax: ", settings.f_max)
        print("q: ", settings.q)
        print("p: ", settings.p)

        t_r = joblib.load("%s/t_r_p_%d.jbl" % (results_path, p))
        benchmark = eval_model(settings, t_r)

        start_idx = (S - t_r.shape[0]) / 2
        end_idx = start_idx + t_r.shape[0]
        bm_large = numpy.zeros((m, S))
        bm_large[:] = numpy.nan
        bm_large[:, start_idx:end_idx] = benchmark

        fn = "%s/benchmark.png" % results_path
        dynamics_retrieval.plot_syn_data.f(bm_large, fn)

        benchmark_flat = benchmark.flatten()

        CCs = []
        nmodes = 20
        x_r_tot = 0
        for mode in range(0, min(nmodes, 2 * settings.f_max + 1)):
            print("Mode: ", mode)

            x_r = joblib.load("%s/movie_p_%d_mode_%d.jbl" % (results_path, p, mode))
            x_r_tot += x_r

            x_r_tot_flat = x_r_tot.flatten()
            CC = dynamics_retrieval.correlate.Correlate(benchmark_flat, x_r_tot_flat)
            CCs.append(CC)

            x_r_large = numpy.zeros((m, S))
            x_r_large[:] = numpy.nan
            x_r_large[:, start_idx:end_idx] = x_r_tot

            dynamics_retrieval.plot_syn_data.f(
                x_r_large,
                "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
                title="%.4f" % (CC),
            )

        joblib.dump(CCs, "%s/CCs_to_benchmark.jbl" % (results_path))

        matplotlib.pyplot.scatter(list(range(1, len(CCs) + 1)), CCs)
        matplotlib.pyplot.axhline(y=1.0)
        matplotlib.pyplot.xticks(list(range(1, len(CCs) + 1, 1)))
        matplotlib.pyplot.savefig("%s/CCs_to_benchmark.png" % (results_path))
        matplotlib.pyplot.close()

####### NEW METHOD v. 2
#   avg = time average of existing obs
# 	dI = I - avg
# 	dI = 0 for unmeasured
# 	N_i = number of obs for an hkl
# 	boost: matmul(diag(S/N_i), dI)
# 	LPSA
# 	Re-add average (no weighting)

flag = 0
if flag == 1:

    # CALC AVG
    import settings_synthetic_data_jitter as settings
    from scipy import sparse

    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    T = joblib.load("%s/x.jbl" % (settings.results_path))
    M = joblib.load("%s/mask.jbl" % (settings.results_path))
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

    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)

        m = settings.m
        S = settings.S
        q = settings.q
        p = settings.p
        results_path = "%s/reconstruction_p_%d" % (settings.results_path, p)

        print("jmax: ", settings.f_max)
        print("q: ", settings.q)
        print("p: ", settings.p)

        t_r = joblib.load("%s/t_r_p_%d.jbl" % (results_path, p))
        benchmark = eval_model(settings, t_r)

        start_idx = (settings.S - t_r.shape[0]) / 2
        end_idx = start_idx + t_r.shape[0]
        bm_large = numpy.zeros((m, S))
        bm_large[:] = numpy.nan
        bm_large[:, start_idx:end_idx] = benchmark

        fn = "%s/benchmark.png" % results_path
        dynamics_retrieval.plot_syn_data.f(bm_large, fn)

        benchmark_flat = benchmark.flatten()

        CCs = []
        nmodes = 20

        ### Case with avg subtraction
        x_r_tot = avgs_matrix[:, start_idx:end_idx]
        print("x_r_tot: ", x_r_tot.shape)

        for mode in range(0, min(nmodes, 2 * settings.f_max + 1)):
            print("Mode: ", mode)

            x_r = joblib.load("%s/movie_p_%d_mode_%d.jbl" % (results_path, p, mode))
            x_r_tot += x_r

            x_r_tot_flat = x_r_tot.flatten()
            CC = dynamics_retrieval.correlate.Correlate(benchmark_flat, x_r_tot_flat)
            CCs.append(CC)

            x_r_large = numpy.zeros((m, S))
            x_r_large[:] = numpy.nan
            x_r_large[:, start_idx:end_idx] = x_r_tot

            dynamics_retrieval.plot_syn_data.f(
                x_r_large,
                "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
                title="%.4f" % (CC),
            )

        joblib.dump(CCs, "%s/CCs_to_benchmark.jbl" % (results_path))

        matplotlib.pyplot.scatter(list(range(1, len(CCs) + 1)), CCs)
        matplotlib.pyplot.axhline(y=1.0)
        matplotlib.pyplot.xticks(list(range(1, len(CCs) + 1, 1)))
        matplotlib.pyplot.savefig("%s/CCs_to_benchmark.png" % (results_path))
        matplotlib.pyplot.close()


# Calculate step-wise CC
# in the sequence of reconstructed signal (p=0)
# with progressively larger q.
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.get_successive_CCs

    qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]
    nmodes = 10
    dynamics_retrieval.get_successive_CCs.get_CCs_q_scan(settings, qs, nmodes)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.get_successive_CCs

    qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]
    nmodes = 10
    dynamics_retrieval.get_successive_CCs.plot_CCs_q_scan(settings, qs, nmodes)


# Calculate step-wise CC
# in the sequence of reconstructed signal
# with progressively larger fmax.
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.get_successive_CCs

    f_max_s = [5, 8, 15, 30, 50, 100, 200]
    nmodes = 10
    dynamics_retrieval.get_successive_CCs.get_CCs_jmax_scan(settings, f_max_s, nmodes)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.get_successive_CCs

    f_max_s = [5, 8, 15, 30, 50, 100, 200]
    nmodes = 10
    dynamics_retrieval.get_successive_CCs.plot_CCs_jmax_scan(settings, f_max_s, nmodes)

###############################
### Standard reconstruction ###
###############################
flag = 0
if flag == 1:
    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)
        end_worker = settings.n_workers_reconstruction - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
            % (end_worker, settings.__name__)
        )

# TIME ASSIGNMENTwith p=(q-1)/2
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.reconstruct_p

    dynamics_retrieval.reconstruct_p.f_ts(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.util_merge_x_r

    for f_max in f_max_s:
        modulename = "settings_f_max_%d" % f_max
        settings = __import__(modulename)
        print("jmax: ", settings.f_max)
        print("q: ", settings.q)
        for mode in settings.modes_to_reconstruct:
            dynamics_retrieval.util_merge_x_r.f(settings, mode)


#######################################################
### p-dependendent reconstruction (avg 2p+1 copies) ###
#######################################################

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.reconstruct_p

    dynamics_retrieval.reconstruct_p.f(settings)
    dynamics_retrieval.reconstruct_p.f_ts(settings)

# SVD of result in data space
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    n_modes_to_use = 4
    x_r_tot = 0
    for mode in range(n_modes_to_use):
        print("Mode:", mode)
        x_r = joblib.load(
            "%s/movie_mode_%d_parallel.jbl" % (settings.results_path, mode)
        )
        x_r_tot += x_r
    joblib.dump(
        x_r_tot,
        "%s/x_r_tot_p_%d_%d_modes.jbl"
        % (settings.results_path, settings.p, n_modes_to_use),
    )

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.SVD

    n_modes_to_use = 4
    x = joblib.load(
        "%s/x_r_tot_p_%d_%d_modes.jbl"
        % (settings.results_path, settings.p, n_modes_to_use)
    )
    U, S, VH = dynamics_retrieval.SVD.SVD_f(x)
    print("Sorting")
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
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.plot_SVs

    dynamics_retrieval.plot_SVs.main(settings)
    import dynamics_retrieval.plot_chronos

    dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    rpath = settings.results_path
    p = settings.p
    t_r = joblib.load("%s/t_r_p_%d.jbl" % (rpath, p))
    start_idx = (settings.S - t_r.shape[0]) / 2
    end_idx = start_idx + t_r.shape[0]

    benchmark = eval_model(settings, t_r)
    dynamics_retrieval.plot_syn_data.f(benchmark, "%s/benchmark_at_t_r.png" % (rpath))
    benchmark = benchmark.flatten()

    U = joblib.load("%s/U.jbl" % rpath)
    S = joblib.load("%s/S.jbl" % rpath)
    VH = joblib.load("%s/VT_final.jbl" % rpath)

    x_r_tot = 0
    CCs = []
    for mode in range(4):
        print("Mode: ", mode)
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]

        x_r = sv * numpy.outer(u, vT)

        x_r_tot += x_r
        x_r_tot_flat = x_r_tot.flatten()

        CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
        CCs.append(CC)

        x_r_large = numpy.zeros((settings.m, settings.S))
        x_r_large[:] = numpy.nan
        x_r_large[:, start_idx:end_idx] = x_r_tot
        dynamics_retrieval.plot_syn_data.f(
            x_r_large,
            "%s/x_r_tot_%d_modes_jet_nan.png" % (rpath, mode + 1),
            title="%0.4f" % CC,
        )

    joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % rpath)

    matplotlib.pyplot.scatter(list(range(1, len(CCs) + 1)), CCs, c="b")
    matplotlib.pyplot.xticks(list(range(1, len(CCs) + 1, 2)))
    matplotlib.pyplot.savefig("%s/reconstruction_CC_vs_nmodes.png" % (rpath))
    matplotlib.pyplot.close()


flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.local_linearity

    rpath = settings.results_path
    U = joblib.load("%s/U.jbl" % rpath)
    S = joblib.load("%s/S.jbl" % rpath)
    VH = joblib.load("%s/VT_final.jbl" % rpath)
    t_r = joblib.load("%s/t_r_p_%d.jbl" % (rpath, settings.p))

    x_r_tot = 0
    Ls = []
    for mode in range(20):
        print("Mode: ", mode)
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        x_r = sv * numpy.outer(u, vT)
        x_r_tot += x_r
        L = dynamics_retrieval.local_linearity.local_linearity_measure_jitter(
            x_r_tot, t_r
        )

        Ls.append(L)
    joblib.dump(Ls, "%s/local_linearity_vs_nmodes.jbl" % rpath)

    matplotlib.pyplot.scatter(list(range(1, len(Ls) + 1)), numpy.log10(Ls), c="b")
    matplotlib.pyplot.xticks(list(range(1, len(Ls) + 1, 2)))
    matplotlib.pyplot.savefig("%s/local_linearity_vs_nmodes.png" % (rpath))
    matplotlib.pyplot.close()


###################
###   BINNING   ###
###################

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data

    rpath = settings.results_path

    x = joblib.load("%s/x.jbl" % rpath)
    print("x: ", x.shape)

    ts_meas = joblib.load("%s/ts_meas.jbl" % rpath)
    print("ts_meas: ", ts_meas.shape, ts_meas.dtype)

    bin_sizes = []
    CCs = []

    bin_size = 1
    bin_sizes.append(bin_size)
    CC = dynamics_retrieval.correlate.Correlate(
        eval_model(settings, ts_meas).flatten(), x.flatten()
    )
    CCs.append(CC)
    print("Starting CC: ", CC)

    for n in range(100, 2100, 100):
        bin_size = 2 * n + 1
        bin_sizes.append(bin_size)

        x_binned = numpy.zeros((settings.m, settings.S - bin_size + 1))
        ts_binned = []
        for i in range(x_binned.shape[1]):
            x_avg = numpy.average(x[:, i : i + bin_size], axis=1)
            x_binned[:, i] = x_avg

            t_binned = numpy.average(ts_meas[i : i + bin_size])
            ts_binned.append(t_binned)

        ts_binned = numpy.asarray(ts_binned)

        CC = dynamics_retrieval.correlate.Correlate(
            eval_model(settings, ts_binned).flatten(), x_binned.flatten()
        )
        CCs.append(CC)
        print("Bin size: ", bin_size, "CC: ", CC)

        x_binned_large = numpy.zeros((settings.m, settings.S))
        x_binned_large[:] = numpy.nan
        x_binned_large[:, n : n + x_binned.shape[1]] = x_binned
        dynamics_retrieval.plot_syn_data.f(
            x_binned_large,
            "%s/x_r_binsize_%d_jet_nan_noticks.png" % (rpath, bin_size),
            title="CC=%0.4f" % CC,
        )

    joblib.dump(bin_sizes, "%s/binsizes.jbl" % rpath)
    joblib.dump(CCs, "%s/reconstruction_CC_vs_binsize.jbl" % rpath)

    matplotlib.pyplot.plot(bin_sizes, CCs, "o-", c="b")
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c="k", linewidth=1)
    matplotlib.pyplot.xlabel("bin size", fontsize=14)
    matplotlib.pyplot.ylabel("CC", fontsize=14)
    matplotlib.pyplot.xlim(left=bin_sizes[0] - 100, right=bin_sizes[-1] + 100)
    matplotlib.pyplot.savefig("%s/reconstruction_CC_vs_binsize.pdf" % rpath, dpi=4 * 96)
    matplotlib.pyplot.close()


################
###   NLSA   ###
################
def make_settings_nlsa(results_path, fn, S, m, q, b, l, log10eps, p):
    fopen = open(fn, "w")
    fopen.write("# -*- coding: utf-8 -*-\n")
    fopen.write("import numpy\n")
    fopen.write("import math\n")
    fopen.write('results_path = "%s"\n' % results_path)
    fopen.write('data_file = "%s/x.jbl"\n' % results_path)
    fopen.write("S = %d\n" % S)
    fopen.write("m = %d\n" % m)
    fopen.write("q = %d\n" % q)
    fopen.write("T_model = 26000\n")  # S-q+1
    fopen.write("tc = float(S)/2\n")
    fopen.write("paral_step = 100\n")
    fopen.write("n_workers = int(math.ceil(float(q)/paral_step))\n")
    fopen.write("b = %d\n" % b)
    fopen.write("log10eps = %0.1f\n" % log10eps)
    fopen.write("sigma_sq = 2*10**log10eps\n")
    fopen.write("l = %d\n" % l)
    fopen.write("nmodes = l\n")
    fopen.write("toproject = range(nmodes)\n")
    fopen.write("paral_step_A = 50\n")
    fopen.write("datatype = numpy.float64\n")
    fopen.write("n_workers_A = int(math.ceil(float(q)/paral_step_A))\n")
    fopen.write("p = %d\n" % p)
    fopen.write("modes_to_reconstruct = range(min(l, 20))\n")
    fopen.write("ncopies = q\n")
    fopen.write("paral_step_reconstruction = 4000\n")
    fopen.write(
        "n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))\n"
    )
    fopen.close()


flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

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
    import settings_synthetic_data_jitter as settings

    d_sq = joblib.load("%s/d_sq.jbl" % (settings.results_path))
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print(numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        S = settings.S
        m = settings.m
        b = 3000
        log10eps = 1.0
        l = 50
        p = (q - 1) / 2

        root_f = "../../synthetic_data_jitter/test8"
        results_path = "%s/NLSA/q_%d" % (root_f, q)
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        fn = "./settings_q_%d.py" % q
        make_settings_nlsa(results_path, fn, S, m, q, b, log10eps, l, p)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        end_worker = settings.n_workers - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --array=0-%d --mem=350G ../scripts_parallel_submission/run_parallel.sh %s"
            % (end_worker, settings.__name__)
        )
        os.system(
            "sbatch -p day -t 1-00:00:00 --array=0-%d --mem=350G ../scripts_parallel_submission/run_parallel_n_Dsq_elements.sh %s"
            % (end_worker, settings.__name__)
        )

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.calculate_distances_utilities
    import dynamics_retrieval.util_merge_D_sq

    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.util_merge_D_sq.f(settings)
        dynamics_retrieval.util_merge_D_sq.f_N_D_sq_elements(settings)
        dynamics_retrieval.calculate_distances_utilities.normalise(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    import dynamics_retrieval.plot_distance_distributions

    qs = [1, 51, 101, 501, 1001, 2001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.plot_distance_distributions.plot_D_0j(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    bs = [10, 100, 500, 1000, 2000, 3000]
    for b in bs:
        S = settings.S
        m = settings.m
        q = 1001
        log10eps = 1.0
        l = 50
        p = (q - 1) / 2

        root_f = "../../synthetic_data_jitter/test6"
        results_path = "%s/NLSA/q_%d/b_%d" % (root_f, q, b)
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        fn = "./settings_b_%d.py" % b
        make_settings_nlsa(results_path, fn, S, m, q, b, log10eps, l, p)

flag = 0
if flag == 1:
    # Select b euclidean nns or b time nns.
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.get_D_N

    # bs = [10, 100, 500, 1000, 2000]
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.get_D_N.main_euclidean_nn(settings)
        # dynamics_retrieval.get_D_N.main_time_nn_1(settings)
        # dynamics_retrieval.get_D_N.main_time_nn_2(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.get_epsilon

    # bs = [10, 100, 500, 1000, 2000]
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.get_epsilon.main(settings)


flag = 0
if flag == 1:

    import settings_synthetic_data_jitter as settings

    log10eps_lst = [-2.0, -1.0, 0.0, 3.0, 8.0]
    for log10eps in log10eps_lst:
        S = settings.S
        m = settings.m
        q = 1001
        b = 100
        l = 50
        p = (q - 1) / 2
        root_f = "../../synthetic_data_jitter/test6"
        results_path = "%s/NLSA/q_%d/b_%d/log10eps_%0.1f" % (root_f, q, b, log10eps)
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        fn = "./settings_log10eps_%0.1f.py" % log10eps
        make_settings_nlsa(results_path, fn, S, m, q, b, log10eps, l, p)


flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.transition_matrix

    # bs = [10, 100, 500, 1000, 2000]
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.transition_matrix.main(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.probability_matrix

    # bs = [10, 100, 500, 1000, 2000]
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.probability_matrix.main(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    ls = [50]
    for l in ls:
        S = settings.S
        m = settings.m
        q = 101
        b = 3000
        log10eps = 1.0
        p = (q - 1) / 2
        root_f = "../../synthetic_data_jitter/test7"
        results_path = "%s/NLSA/q_%d/b_%d/log10eps_%0.1f/l_%d" % (
            root_f,
            q,
            b,
            log10eps,
            l,
        )
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        fn = "./settings_l_%d.py" % l
        make_settings_nlsa(results_path, fn, S, m, q, b, log10eps, l, p)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.eigendecompose

    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    #     settings = __import__(modulename)
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.eigendecompose.main(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings

    evecs = joblib.load("%s/evecs_sorted.jbl" % settings.results_path)
    test = numpy.matmul(evecs.T, evecs)
    diff = abs(test - numpy.eye(settings.l))
    print(numpy.amax(diff))

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.plot_P_evecs

    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    #     settings = __import__(modulename)
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.plot_P_evecs.main(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    #     settings = __import__(modulename)
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        end_worker = settings.n_workers_A - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s"
            % (end_worker, settings.__name__)
        )

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.util_merge_A

    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.util_merge_A.main(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.SVD

    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.SVD.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_chronos
    import dynamics_retrieval.plot_SVs

    # import settings_synthetic_data_jitter as settings
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    #     settings = __import__(modulename)
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.plot_SVs.main(settings)
        dynamics_retrieval.plot_chronos.main(settings)

flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
        # for modulename in modulenames:
        #     settings = __import__(modulename)
        # MAKE SUPERVECTOR TIMESTAMPS
        S = settings.S
        q = settings.q
        s = S - q + 1

        ts_meas = joblib.load("%s/ts_meas.jbl" % settings.results_path)
        ts_svs = []
        for i in range(s):
            t_sv = numpy.average(ts_meas[i : i + q])
            ts_svs.append(t_sv)
        ts_svs = numpy.asarray(ts_svs)
        joblib.dump(ts_svs, "%s/ts_svs.jbl" % settings.results_path)

# p-dependent reconstruction
flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.reconstruct_p

    # modulenames = ['settings_log10eps_m1p5', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    #     settings = __import__(modulename)
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        dynamics_retrieval.reconstruct_p.f(settings)
        dynamics_retrieval.reconstruct_p.f_ts(settings)


# STANDARD RECONSTRUCTION
flag = 0
if flag == 1:
    # import settings_q_2001 as settings
    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
        # for modulename in modulenames:
        #     settings = __import__(modulename)
        end_worker = settings.n_workers_reconstruction - 1
        os.system(
            "sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s"
            % (end_worker, settings.__name__)
        )

# TIME ASSIGNMENT with p=(q-1)/2
flag = 0
if flag == 1:
    # import settings_q_2001 as settings
    import dynamics_retrieval.reconstruct_p

    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [1, 101, 1001, 2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)
        # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
        # for modulename in modulenames:
        #     settings = __import__(modulename)
        dynamics_retrieval.reconstruct_p.f_ts(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_x_r

    # ls = [5, 10, 30]
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    qs = [2001, 4001]
    for q in qs:
        modulename = "settings_q_%d" % q
        settings = __import__(modulename)

        for mode in settings.modes_to_reconstruct:
            dynamics_retrieval.util_merge_x_r.f(settings, mode)

# CC TO BENCHMARK
flag = 0
if flag == 1:
    # import settings_synthetic_data_jitter as settings
    import dynamics_retrieval.correlate
    import dynamics_retrieval.plot_syn_data
    import dynamics_retrieval.reconstruct_p

    ls = [50]
    for l in ls:
        modulename = "settings_l_%d" % l
        # qs = [1001]
        # for q in qs:
        #     modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        p = settings.p
        results_path = "%s/reconstruction_p_%d" % (settings.results_path, p)
        t_r = joblib.load("%s/t_r_p_%d.jbl" % (results_path, p))

        benchmark = eval_model(settings, t_r)
        print("Benchmark: ", benchmark.shape)
        dynamics_retrieval.plot_syn_data.f(
            benchmark, "%s/benchmark_at_t_r.png" % results_path
        )
        benchmark = benchmark.flatten()

        x_r_tot = 0
        CCs = []
        for mode in range(min(settings.l, 20)):
            print("Mode: ", mode)

            x_r = joblib.load("%s/movie_mode_%d_parallel.jbl" % (results_path, mode))
            print("x_r: ", x_r.shape)

            x_r_tot += x_r
            x_r_tot_flat = x_r_tot.flatten()

            CC = dynamics_retrieval.correlate.Correlate(benchmark, x_r_tot_flat)
            print("CC: ", CC)
            CCs.append(CC)

            dynamics_retrieval.plot_syn_data.f(
                x_r_tot,
                "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
                title="%.4f" % CC,
            )

            start = (settings.S - x_r_tot.shape[1]) / 2
            end = start + x_r_tot.shape[1]
            x_r_large = numpy.zeros((settings.m, settings.S))
            x_r_large[:] = numpy.nan
            x_r_large[:, start:end] = x_r_tot

            dynamics_retrieval.plot_syn_data.f(
                x_r_large,
                "%s/x_r_tot_%d_modes.png" % (results_path, mode + 1),
                title="%.4f" % CC,
            )

        joblib.dump(CCs, "%s/reconstruction_CC_vs_nmodes.jbl" % results_path)

        matplotlib.pyplot.scatter(list(range(1, len(CCs) + 1)), CCs, c="b")
        matplotlib.pyplot.xticks(list(range(1, len(CCs) + 1, 2)))
        matplotlib.pyplot.savefig("%s/reconstruction_CC_vs_nmodes.png" % results_path)
        matplotlib.pyplot.close()
