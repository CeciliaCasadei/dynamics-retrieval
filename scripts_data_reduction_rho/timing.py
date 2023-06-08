# -*- coding: utf-8 -*-
import h5py
import matplotlib
import numpy

matplotlib.use("Agg")
import os

import joblib
import matplotlib.pyplot as plt
import run_ns
import scipy.io
import scipy.optimize


def extract_SPECENC_data(SPECENC_fn):
    dataFile = h5py.File(SPECENC_fn, "r")

    arr_time_SPECENC = dataFile["arrival_times"][
        :
    ]  # [fs], absurd large value for dark pulses
    arr_time_reliability_SPECENC = dataFile["arrival_times_amplitude"][
        :
    ]  # Very low value for dark pulses
    nom_delay_SPECENC = dataFile["nominal_delay_from_stage"][:]  # [fs]
    pulse_id_SPECENC = dataFile["pulse_ids"][:]
    timestamp_SPECENC = (
        arr_time_SPECENC + nom_delay_SPECENC
    )  # [fs], absurd largevalue for dark pulses

    return pulse_id_SPECENC, timestamp_SPECENC, arr_time_reliability_SPECENC


def extract_BSREAD_data(BSREAD_fn):
    dataFile = h5py.File(BSREAD_fn, "r")

    pulse_Is_BSREAD = dataFile["data/SAROP11-PBPS122:INTENSITY/data"][:]
    pulse_id_BSREAD = dataFile["data/SAROP11-PBPS122:INTENSITY/pulse_id"][:]

    return pulse_id_BSREAD, pulse_Is_BSREAD


def extract_JF_data(JF_fn):
    dataFile = h5py.File(JF_fn, "r")

    pulse_id_JF = dataFile["data/JF06T08V01/pulse_id"][:].flatten()

    #    data = dataFile['data/JF06T08V01/data'] #very slow loading big data matrix
    #    data = numpy.asarray(data)
    #    if not data.shape[0] == pulse_id_JF.shape[0]:
    #        raise Exception('shape issue') # never!

    return pulse_id_JF


def get_ts_distribution():
    path = "/sf/alvra/data/p18594/raw"
    path_out = "./timing_data"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    ts_light = []
    ts_rel_light = []
    FEL_Is_light = []

    n_discarded_runs = 0
    n_selected_runs = 0

    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\n")
        print("SCAN: ", scan_n, "RUNS: ", run_start, run_end)
        for run_n in range(run_start, run_end + 1):
            SPECENC_file = "%s/timetool/rho_nlsa_scan_%d/run_%0.6d.SPECENC.h5" % (
                path,
                scan_n,
                run_n,
            )
            BSREAD_file = "%s/rho_nlsa_scan_%d/run_%0.6d.BSREAD.h5" % (
                path,
                scan_n,
                run_n,
            )
            JF_file = "%s/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.h5" % (
                path,
                scan_n,
                run_n,
            )
            list_file_light = "%s/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.light.lst" % (
                path,
                scan_n,
                run_n,
            )

            pid_SPECENC, ts_SPECENC, ts_rel_SPECENC = extract_SPECENC_data(SPECENC_file)
            pid_BSREAD, pI_BSREAD = extract_BSREAD_data(BSREAD_file)
            pid_JF = extract_JF_data(JF_file)

            if not (
                pid_BSREAD.shape == pid_SPECENC.shape
                and pid_BSREAD.shape == pid_JF.shape
            ):
                n_discarded_runs += 1
            else:
                diff_1 = abs(pid_BSREAD - pid_SPECENC)
                diff_2 = abs(pid_BSREAD - pid_JF)
                if diff_1.sum() > 0 or diff_2.sum() > 0:
                    n_discarded_runs += 1
                else:

                    # THIS RUN IS SELECTED
                    n_selected_runs += 1
                    print("RUN: ", run_n, ", ", pid_BSREAD.shape[
                        0
                    ], " frames.")  # 1000 frames (from 0 to 999) except last run of scan.
                    l = ts_SPECENC.shape[0]
                    out = numpy.zeros((l, 5))
                    out[:, 0] = list(range(l))
                    out[:, 1] = pid_BSREAD
                    out[:, 2] = ts_SPECENC
                    out[:, 3] = ts_rel_SPECENC
                    out[:, 4] = pI_BSREAD.flatten()
                    print("Matrix shape (light and dark):", out.shape)

                    # SELECT ONLY LIGHT
                    f = open(list_file_light, "r")
                    lines = f.read().splitlines()
                    f.close()

                    idxs = numpy.zeros((len(lines),), dtype=numpy.int)
                    i = 0
                    for line in lines:
                        event_n = int(line.split()[1][2:])
                        idxs[i] = event_n
                        i += 1
                    out_light = out[idxs, :]
                    print("Matrix shape (light):", out_light.shape)

                    joblib.dump(
                        out_light,
                        "%s/light_scan_%d_run_%0.6d_event_pid_ts_tsrel_felI.jbl"
                        % (path_out, scan_n, run_n),
                    )

                    ts_light.append(out_light[:, 2])
                    ts_rel_light.append(out_light[:, 3])
                    FEL_Is_light.append(out_light[:, 4])

    ts_light = numpy.asarray([item for sublist in ts_light for item in sublist])
    ts_rel_light = numpy.asarray([item for sublist in ts_rel_light for item in sublist])
    FEL_Is_light = numpy.asarray([item for sublist in FEL_Is_light for item in sublist])

    print("\n")
    print("n selected runs", n_selected_runs)
    print("n discarded runs", n_discarded_runs)

    print("\n")
    print("NaNs in ts_light: ", numpy.any(numpy.isnan(ts_light)))
    print("NaNs in ts_rel_light: ", numpy.any(numpy.isnan(ts_rel_light)))
    print("NaNs in FEL_Is_light: ", numpy.any(numpy.isnan(FEL_Is_light)))

    # FIGURES
    plt.scatter(FEL_Is_light, ts_rel_light, c="b", s=5, alpha=0.3)
    plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
    plt.ylabel("Arrival times amplitude (light shot timestamp reliability.)")
    plt.savefig("%s/plot_light_tsrel_vs_FEL_Is.png" % path_out)
    plt.close()

    plt.figure()
    mybins = numpy.arange(-800, +800, 10)
    plt.hist(ts_light, bins=mybins, color="b")
    plt.title("Light shot timestamps. N.: %d" % (ts_light.shape[0]))
    plt.xlabel("Timestamp [fs]")
    plt.savefig("%s/hist_light_ts.png" % path_out)
    plt.close()

    plt.figure()
    plt.hist(ts_rel_light, color="b")
    plt.title(
        "Arrival times amplitude (light shot timestamp reliability.) N.: %d"
        % ts_rel_light.shape[0]
    )
    plt.xlabel("Timestamp reliability [a.u.]")
    plt.savefig("%s/hist_light_ts_rel.png" % path_out)
    plt.close()

    plt.figure()
    plt.hist(FEL_Is_light, color="b")
    plt.title("Light shot FEL intensities. N.: %d" % FEL_Is_light.shape[0])
    plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
    plt.savefig("%s/hist_light_FEL_Is.png" % path_out)
    plt.close()


def export_unique_ID():
    path_out = "./timing_data"

    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\n")
        print("SCAN: ", scan_n, "RUNS: ", run_start, run_end)
        for run_n in range(run_start, run_end + 1):
            try:
                data = joblib.load(
                    "%s/light_scan_%d_run_%0.6d_event_pid_ts_tsrel_felI.jbl"
                    % (path_out, scan_n, run_n)
                )
                n_frames = data.shape[0]
                path_string = (
                    "/sf/alvra/data/p18594/raw/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.h5_event_"
                    % (scan_n, run_n)
                )
                unique_IDs = []
                for i in range(n_frames):
                    event_n = str(int(data[i, 0]))
                    unique_ID = path_string + event_n
                    unique_IDs.append(unique_ID)
                joblib.dump(
                    unique_IDs,
                    "%s/light_scan_%d_run_%0.6d_uniqueID.jbl"
                    % (path_out, scan_n, run_n),
                )
                print(len(unique_IDs), data.shape)
            except:
                print("Scan", scan_n, ", run ", run_n, " not existing")


def merge_scan_data():

    path_out = "./timing_data"

    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\n")
        print("SCAN: ", scan_n, "RUNS: ", run_start, run_end)
        scan_ts = []
        scan_ts_rel = []
        scan_FEL_Is = []
        scan_uniqueIDs = []
        print(len(scan_ts), len(scan_ts_rel), len(scan_FEL_Is), len(scan_uniqueIDs))
        for run_n in range(run_start, run_end + 1):
            try:
                print("Run: ", run_n)
                data = joblib.load(
                    "%s/light_scan_%d_run_%0.6d_event_pid_ts_tsrel_felI.jbl"
                    % (path_out, scan_n, run_n)
                )
                unique_IDs = joblib.load(
                    "%s/light_scan_%d_run_%0.6d_uniqueID.jbl"
                    % (path_out, scan_n, run_n)
                )
                scan_ts.append(data[:, 2])
                scan_ts_rel.append(data[:, 3])
                scan_FEL_Is.append(data[:, 4])
                scan_uniqueIDs.append(unique_IDs)
            except:
                print("Scan", scan_n, ", run ", run_n, " not existing")

        scan_ts_flat = numpy.asarray([item for sublist in scan_ts for item in sublist])
        scan_ts_rel_flat = numpy.asarray(
            [item for sublist in scan_ts_rel for item in sublist]
        )
        scan_FEL_Is_flat = numpy.asarray(
            [item for sublist in scan_FEL_Is for item in sublist]
        )
        scan_uniqueIDs_flat = [item for sublist in scan_uniqueIDs for item in sublist]

        scan_out = numpy.zeros((scan_ts_flat.shape[0], 3))
        scan_out[:, 0] = scan_ts_flat
        scan_out[:, 1] = scan_ts_rel_flat
        scan_out[:, 2] = scan_FEL_Is_flat

        joblib.dump(scan_out, "%s/light_scan_%d_ts_tsrel_felI.jbl" % (path_out, scan_n))

        joblib.dump(
            scan_uniqueIDs_flat, "%s/light_scan_%d_uniqueID.jbl" % (path_out, scan_n)
        )


def twoD_Gaussian_simple(xxx_todo_changeme, amplitude, xo, yo, sigma_x, sigma_y):
    (x, y) = xxx_todo_changeme
    g = amplitude * numpy.exp(
        -((x - xo) ** 2) / (2 * sigma_x ** 2) - ((y - yo) ** 2) / (2 * sigma_y ** 2)
    )
    # return g.ravel()
    return g.flatten()


def twoD_Gaussian_rotated(xxx_todo_changeme1, amplitude, x0, y0, sigma_x, sigma_y, theta):
    (x, y) = xxx_todo_changeme1
    a = (numpy.cos(theta) ** 2) / (2 * sigma_x ** 2) + (numpy.sin(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    b = (-numpy.sin(2 * theta)) / (4 * sigma_x ** 2) + (numpy.sin(2 * theta)) / (
        4 * sigma_y ** 2
    )
    c = (numpy.sin(theta) ** 2) / (2 * sigma_x ** 2) + (numpy.cos(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    dx = x - x0
    dy = y - y0
    expo = a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2
    g = amplitude * numpy.exp(-expo)
    return g.flatten()


def twoD_Gaussian_rotated_bimodal(xxx_todo_changeme2,
    amplitude_1,
    x0_1,
    y0_1,
    sigma_x_1,
    sigma_y_1,
    theta_1,
    amplitude_2,
    x0_2,
    y0_2,
    sigma_x_2,
    sigma_y_2,
    theta_2,
):

    (x, y) = xxx_todo_changeme2
    a_1 = (numpy.cos(theta_1) ** 2) / (2 * sigma_x_1 ** 2) + (
        numpy.sin(theta_1) ** 2
    ) / (2 * sigma_y_1 ** 2)
    b_1 = (-numpy.sin(2 * theta_1)) / (4 * sigma_x_1 ** 2) + (
        numpy.sin(2 * theta_1)
    ) / (4 * sigma_y_1 ** 2)
    c_1 = (numpy.sin(theta_1) ** 2) / (2 * sigma_x_1 ** 2) + (
        numpy.cos(theta_1) ** 2
    ) / (2 * sigma_y_1 ** 2)
    dx_1 = x - x0_1
    dy_1 = y - y0_1
    expo_1 = a_1 * dx_1 ** 2 + 2 * b_1 * dx_1 * dy_1 + c_1 * dy_1 ** 2

    a_2 = (numpy.cos(theta_2) ** 2) / (2 * sigma_x_2 ** 2) + (
        numpy.sin(theta_2) ** 2
    ) / (2 * sigma_y_2 ** 2)
    b_2 = (-numpy.sin(2 * theta_2)) / (4 * sigma_x_2 ** 2) + (
        numpy.sin(2 * theta_2)
    ) / (4 * sigma_y_2 ** 2)
    c_2 = (numpy.sin(theta_2) ** 2) / (2 * sigma_x_2 ** 2) + (
        numpy.cos(theta_2) ** 2
    ) / (2 * sigma_y_2 ** 2)
    dx_2 = x - x0_2
    dy_2 = y - y0_2
    expo_2 = a_2 * dx_2 ** 2 + 2 * b_2 * dx_2 * dy_2 + c_2 * dy_2 ** 2

    g = amplitude_1 * numpy.exp(-expo_1) + amplitude_2 * numpy.exp(-expo_2)
    return g.flatten()


def select():
    path_out = "./timing_data"
    for scan_n in range(1, 24):
        print("\n")
        print("SCAN: ", scan_n)

        scan_data = joblib.load(
            "%s/light_scan_%d_ts_tsrel_felI.jbl" % (path_out, scan_n)
        )

        scan_unique_IDs = joblib.load(
            "%s/light_scan_%d_uniqueID.jbl" % (path_out, scan_n)
        )

        print("NaNs in light data: ", numpy.any(numpy.isnan(scan_data)))

        FEL_Is_light = scan_data[:, 2]
        ts_rel_light = scan_data[:, 1]

        # FIGURES
        plt.scatter(FEL_Is_light, ts_rel_light, c="b", s=5, alpha=0.3)
        plt.xlim((1.0, 4.0))
        plt.ylim((0.0, 0.2))
        plt.title("Scan %d" % scan_n)
        # plt.xlabel('FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)')
        # plt.ylabel('Arrival times amplitude (light shot timestamp reliability.)')
        plt.savefig(
            "%s/scan_%d_plot_light_tsrel_vs_FEL_Is_lims.png" % (path_out, scan_n)
        )
        plt.close()

        # Calculate local density by nearest neighbors number
        n_pts_x = 300
        n_pts_y = 200
        x = numpy.linspace(1.0, 4.0, n_pts_x)
        y = numpy.linspace(0.0, 0.2, n_pts_y)
        x_grid, y_grid = numpy.meshgrid(x, y)
        density_grid = numpy.zeros(x_grid.shape)

        a = 0.5 / 10  # 100
        b = 0.02 / 10  # 100
        for i in range(n_pts_y):
            # print i
            for j in range(n_pts_x):
                x = x_grid[i, j]  # FEL I
                y = y_grid[i, j]  # Arrival time amplitude

                dI = abs(FEL_Is_light - x)
                dtsrel = abs(ts_rel_light - y)

                idxs = numpy.argwhere(
                    (dI ** 2) / (a ** 2) + (dtsrel ** 2 / (b ** 2)) < 1
                )
                n_nn = idxs.shape[0]
                # print x, y, n_nn
                if n_nn == 0:
                    n_nn = 1  # to take ln
                density_grid[i, j] = n_nn

        ln_density_grid = numpy.log(density_grid)

        # FIGURES
        plt.scatter(
            x_grid.flatten(), y_grid.flatten(), c=density_grid.flatten(), s=5, alpha=0.3
        )
        plt.title("Scan %d" % scan_n)
        plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
        plt.ylabel("Arrival times amplitude (light shot timestamp reliability.)")
        plt.savefig(
            "%s/scan_%d_density_plot_light_tsrel_vs_FEL_Is.png" % (path_out, scan_n)
        )
        plt.close()

        plt.scatter(
            x_grid.flatten(),
            y_grid.flatten(),
            c=ln_density_grid.flatten(),
            s=5,
            alpha=0.3,
        )
        plt.title("Scan %d" % scan_n)
        plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
        plt.ylabel("Arrival times amplitude (light shot timestamp reliability.)")
        plt.savefig(
            "%s/scan_%d_ln_density_plot_light_tsrel_vs_FEL_Is.png" % (path_out, scan_n)
        )
        plt.close()

        # Fit ln(density) with uni/bi-modal 2D elliptical Gaussian.
        initial_i_idx = numpy.unravel_index(
            numpy.argmax(ln_density_grid, axis=None), ln_density_grid.shape
        )[0]
        initial_j_idx = numpy.unravel_index(
            numpy.argmax(ln_density_grid, axis=None), ln_density_grid.shape
        )[1]
        initial_x = x_grid[initial_i_idx, initial_j_idx]  # 2.1
        initial_y = y_grid[initial_i_idx, initial_j_idx]  # 0.15
        initial_amplitude = ln_density_grid[initial_i_idx, initial_j_idx]  # 124
        initial_theta = numpy.pi / 50

        if scan_n == 8:
            initial_guess = (
                initial_amplitude,
                2.6,
                0.11,
                0.15,
                0.01 / 3,
                initial_theta / 10,
                initial_amplitude / 2,
                3.1,
                0.10,
                0.25,
                0.01 / 2,
                initial_theta / 10,
            )
            f = twoD_Gaussian_rotated_bimodal
            thresh = 0.1
        elif scan_n == 14:
            initial_guess = (
                initial_amplitude,
                initial_x,
                0.13,
                0.5,
                0.01 / 3,
                initial_theta / 100,
                initial_amplitude / 2,
                initial_x,
                0.08,
                0.5,
                0.01 / 3,
                initial_theta / 100,
            )
            f = twoD_Gaussian_rotated_bimodal
            thresh = 0.1
        elif scan_n == 20:
            initial_guess = (
                initial_amplitude,
                2.6,
                0.125,
                0.2,
                0.01 / 2,
                initial_theta / 100,
                initial_amplitude / 2,
                3.1,
                0.125,
                0.2,
                0.01 / 2,
                initial_theta / 100,
            )
            f = twoD_Gaussian_rotated_bimodal
            thresh = 0.1
        else:
            initial_guess = (
                initial_amplitude,
                initial_x,
                initial_y,
                0.5,
                0.01,
                initial_theta,
            )
            f = twoD_Gaussian_rotated
            thresh = 0.05

        popt, pcov = scipy.optimize.curve_fit(
            f, (x_grid, y_grid), ln_density_grid.flatten(), p0=initial_guess
        )

        # Calculate model values at grid pts and at experimental pts.
        data_fitted = f((x_grid, y_grid), *popt)
        amplis = f((FEL_Is_light, ts_rel_light), *popt)

        refined_amplitude = popt[0]

        ### PLOT GAUSS FIT ###
        plt.scatter(
            x_grid.flatten(),
            y_grid.flatten(),
            c=ln_density_grid.flatten(),
            s=5,
            alpha=0.3,
        )
        plt.xlim((1.0, 4.0))
        plt.ylim((0.0, 0.2))
        plt.gca().contour(
            x_grid, y_grid, data_fitted.reshape(n_pts_y, n_pts_x), 4, colors="w"
        )
        plt.title("Scan %d" % scan_n)
        # plt.xlabel('FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)')
        # plt.ylabel('Arrival times amplitude (light shot timestamp reliability.)')
        plt.savefig(
            "%s/scan_%d_ln_density_plot_light_tsrel_vs_FEL_Is_fit_rotated.png"
            % (path_out, scan_n)
        )
        plt.close()

        # Select only data pts at positions where density model value is bigger than thresh*refined amplitude.
        idxs = numpy.argwhere(amplis / refined_amplitude > thresh).flatten()
        FEL_Is_light_selected = FEL_Is_light[idxs]
        ts_rel_light_selected = ts_rel_light[idxs]

        scan_data_selected = scan_data[idxs, :]
        scan_unique_IDs_selected = [scan_unique_IDs[i] for i in idxs]

        joblib.dump(
            scan_data_selected,
            "%s/light_scan_%d_ts_tsrel_felI_selected.jbl" % (path_out, scan_n),
        )
        joblib.dump(
            scan_unique_IDs_selected,
            "%s/light_scan_%d_uniqueID_selected.jbl" % (path_out, scan_n),
        )
        print("NaNs in selected light data: ", numpy.any(
            numpy.isnan(scan_data_selected)
        ))

        # FIGURES
        plt.scatter(FEL_Is_light, ts_rel_light, c="b", s=5)
        plt.scatter(FEL_Is_light_selected, ts_rel_light_selected, c="m", s=5)
        # plt.xlim((1.0,4.0))
        # plt.ylim((0.0,0.2))
        plt.title("Scan %d" % scan_n)
        # plt.xlabel('FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)')
        # plt.ylabel('Arrival times amplitude (light shot timestamp reliability.)')
        plt.savefig(
            "%s/scan_%d_plot_light_tsrel_vs_FEL_Is_selected.png" % (path_out, scan_n)
        )
        plt.close()


def check_t_distribution():

    path_out = "./timing_data"

    ts_light = []
    ts_rel_light = []
    FEL_Is_light = []

    for scan_n in range(1, 24):

        print("\n")
        print("SCAN: ", scan_n)
        scan_data_selected = joblib.load(
            "%s/light_scan_%d_ts_tsrel_felI_selected.jbl" % (path_out, scan_n)
        )
        print("\n")
        print("NaNs in scan_data_selected: ", numpy.any(numpy.isnan(scan_data_selected)))

        ts_light.append(scan_data_selected[:, 0])
        ts_rel_light.append(scan_data_selected[:, 1])
        FEL_Is_light.append(scan_data_selected[:, 2])

    ts_light = numpy.asarray([item for sublist in ts_light for item in sublist])
    ts_rel_light = numpy.asarray([item for sublist in ts_rel_light for item in sublist])
    FEL_Is_light = numpy.asarray([item for sublist in FEL_Is_light for item in sublist])

    print(ts_light.shape)
    print(ts_rel_light.shape)
    print(FEL_Is_light.shape)

    # FIGURES
    plt.scatter(FEL_Is_light, ts_rel_light, c="b", s=5, alpha=0.3)
    plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
    plt.ylabel("Arrival times amplitude (light shot timestamp reliability.)")
    plt.savefig("%s/plot_light_tsrel_vs_FEL_Is_selected.png" % path_out)
    plt.close()

    plt.figure()
    mybins = numpy.arange(-800, +800, 10)
    plt.hist(ts_light, bins=mybins, color="b")
    plt.title("Light shot timestamps. N.: %d" % (ts_light.shape[0]))
    plt.xlabel("Timestamp [fs]")
    plt.savefig("%s/hist_light_ts_selected.png" % path_out)
    plt.close()

    plt.figure()
    plt.hist(ts_rel_light, color="b")
    plt.title(
        "Arrival times amplitude (light shot timestamp reliability.) N.: %d"
        % ts_rel_light.shape[0]
    )
    plt.xlabel("Timestamp reliability [a.u.]")
    plt.savefig("%s/hist_light_ts_rel_selected.png" % path_out)
    plt.close()

    plt.figure()
    plt.hist(FEL_Is_light, color="b")
    plt.title("Light shot FEL intensities. N.: %d" % FEL_Is_light.shape[0])
    plt.xlabel("FEL Intensity [a.u.] (SAROP11-PBPS122:INTENSITY)")
    plt.savefig("%s/hist_light_FEL_Is_selected.png" % path_out)
    plt.close()


def export_unique_ID_ts():

    path_out = "./timing_data"

    scan_start = 20
    scan_end = 23

    fn_out = "%s/scans_%d_%d_uniqueIDs_ts.txt" % (path_out, scan_start, scan_end)
    fn_write = open(fn_out, "w")

    for scan_n in range(scan_start, scan_end + 1):

        print("\nScan", scan_n)

        data_ts = joblib.load(
            "%s/light_scan_%d_ts_tsrel_felI_selected.jbl" % (path_out, scan_n)
        )
        ts = data_ts[:, 0]
        n_frames = ts.shape[0]

        data_IDs = joblib.load(
            "%s/light_scan_%d_uniqueID_selected.jbl" % (path_out, scan_n)
        )

        if abs(n_frames - len(data_IDs)) > 0:
            print("Problem")
        for i in range(n_frames):
            fn_write.write("%s  %.1f\n" % (data_IDs[i], ts[i]))

    fn_write.close()


# def match_unique_ID():
#     thr = 0.06
#     n = 0
#     n_tot = 0
#     for scan_n in range(1, 24):
#         run_start, run_end = run_ns.run_ns(scan_n)
#         print '\nScan', scan_n, 'run range: ', run_start, run_end
#         for run_n in range(run_start, run_end+1):
#             path_string = '/sf/alvra/data/p18594/raw/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.h5_event_'%(scan_n,
#                                                                                                        run_n)
#             try:
#                 data = joblib.load('./run_pid_ts_thr/scan_%d_run_%0.6d_event_pid_ts_tsrel_thr_%0.2f.jbl'%(scan_n,
#                                                                                                           run_n,
#                                                                                                           thr))
#                 n_frames = data.shape[0]
#                 n_tot = n_tot + n_frames
#                 unique_ID_timed = []
#                 for i in range(n_frames):
#                     event_n = str(int(data[i, 0]))
#                     unique_ID = path_string + event_n
#                     unique_ID_timed.append(unique_ID)

#                 fn = './run_pid_ts_thr/scan_%d_run_%0.6d_uniqueIDs_thr_%0.2f'%(scan_n,
#                                                                                run_n,
#                                                                                thr)
#                 joblib.dump(unique_ID_timed, '%s.jbl'%fn)
# #                myDict = {"unique_ID_timed": unique_ID_timed}
# #                scipy.io.savemat('%s.mat'%fn, myDict)
#             except:
#                 print 'Scan', scan_n, ', run ', run_n, ' not existing'
#                 n = n+1

#     print 'N. non existing runs: ', n
#     print 'Total n timed frames', n_tot


if __name__ == "__main__":
    ### 1. Using the event lists, separate light and dark events in NLSA runs.
    ### 2. Light events: make per run matrix
    ###    eventNumber pulseID timestamp timestampReliability FELintensity
    # get_ts_distribution()

    ### Light events: make per run matrix
    ### uniqueID
    ### with uniqueID = <filename>_<eventN>
    # export_unique_ID()

    ### Merge all data from runs within the same scan.
    ### Light events: make per scan matrix
    ### timestamp timestampReliability FELintensity
    ### and
    ### uniqueID
    # merge_scan_data()

    ### Approximate local density of points
    ### in the space (FEL_intensity, arrival_time_amplitude)
    ### with nearest neighbor number.
    ### Model ln(density) with a 2D gaussian with (rotated) elliptical section.
    ### Keep only experimental points that lie in model region where gaussian height is above 5% of its maximum.
    ### Some scans require superposition of 2 ellipses!
    # select()

    # check_t_distribution()

    export_unique_ID_ts()