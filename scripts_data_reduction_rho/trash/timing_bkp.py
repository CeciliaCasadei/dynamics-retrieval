# -*- coding: utf-8 -*-
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy
import run_ns
import scipy.io


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
    ts_selected = []
    ts_discarded = []
    Is_selected = []
    Is_discarded = []
    ts_rel_selected = []
    ts_rel_discarded = []

    thr = 0.06

    n_discarded_runs = 0
    n_selected_runs = 0

    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print(run_start, run_end)
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

            print(SPECENC_file)
            print(BSREAD_file)
            print(JF_file)

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
                    l = ts_SPECENC.shape[0]
                    out = numpy.zeros((l, 4))
                    out[:, 0] = list(range(l))
                    out[:, 1] = pid_BSREAD
                    out[:, 2] = ts_SPECENC
                    out[:, 3] = ts_rel_SPECENC
                    joblib.dump(
                        out,
                        "./run_pid_ts/scan_%d_run_%0.6d_event_pid_ts_tsrel.jbl"
                        % (scan_n, run_n),
                    )

                    tstamps_selected = [
                        ts_SPECENC[k] for k in range(l) if ts_rel_SPECENC[k] > thr
                    ]
                    ts_selected.append(tstamps_selected)

                    tstamps_discarded = [
                        ts_SPECENC[k] for k in range(l) if ts_rel_SPECENC[k] <= thr
                    ]
                    ts_discarded.append(tstamps_discarded)

                    pI_selected = [
                        pI_BSREAD[k, 0] for k in range(l) if ts_rel_SPECENC[k] > thr
                    ]
                    Is_selected.append(pI_selected)

                    pI_discarded = [
                        pI_BSREAD[k, 0] for k in range(l) if ts_rel_SPECENC[k] <= thr
                    ]
                    Is_discarded.append(pI_discarded)

                    ### ADDED ###
                    tstamp_rel_selected = [
                        ts_rel_SPECENC[k] for k in range(l) if ts_rel_SPECENC[k] > thr
                    ]
                    ts_rel_selected.append(tstamp_rel_selected)

                    tstamp_rel_discarded = [
                        ts_rel_SPECENC[k] for k in range(l) if ts_rel_SPECENC[k] <= thr
                    ]
                    ts_rel_discarded.append(tstamp_rel_discarded)
                    ####

    print("n selected runs", n_selected_runs)
    print("n discarded runs", n_discarded_runs)

    plt.scatter(
        numpy.hstack(Is_selected), numpy.hstack(ts_rel_selected), c="b", s=5, alpha=0.3
    )
    plt.scatter(
        numpy.hstack(Is_discarded),
        numpy.hstack(ts_rel_discarded),
        c="m",
        s=5,
        alpha=0.3,
    )
    plt.xlabel("SAROP11-PBPS122:INTENSITY")
    plt.ylabel("Arrival times amplitude")
    plt.savefig("plot_Is_vs_tsrel_thresh_%.2f.png" % thr)
    plt.close()

    idxs = numpy.argwhere(numpy.hstack(Is_selected) < 1.0)
    print(idxs.shape[0], "selected Is smaller than 1.0 out of", numpy.hstack(
        Is_selected
    ).shape[0], "selected")
    idxs = numpy.argwhere(numpy.hstack(Is_discarded) < 1.0)
    print(idxs.shape[0], "discarded Is smaller than 1.0 out of", numpy.hstack(
        Is_discarded
    ).shape[0], "discarded")

    ts_selected = numpy.hstack(ts_selected)
    n_selected = ts_selected.shape[0]
    ts_selected = [
        ts_selected[i]
        for i in range(ts_selected.shape[0])
        if not numpy.isnan(ts_selected[i])
    ]
    n_nan = n_selected - len(ts_selected)
    print("N. selected shots", n_selected, " of wich, nan timestamp:", n_nan)

    mybins = numpy.arange(-800, +800, 10)
    plt.figure()
    plt.hist(ts_selected, bins=mybins)
    plt.title("Selected shot timestamps. N.: %d, of which %d nan" % (n_selected, n_nan))
    plt.xlabel("Timestamp [fs]")
    plt.savefig("hist_ts_thresh_%.2f.png" % thr)
    plt.close()

    ts_discarded = numpy.hstack(ts_discarded)
    n_discarded = ts_discarded.shape[0]
    ts_discarded = [
        ts_discarded[i]
        for i in range(ts_discarded.shape[0])
        if not numpy.isnan(ts_discarded[i])
    ]
    n_nan = n_discarded - len(ts_discarded)
    print("N. discarded shots", n_discarded, " of wich, nan timestamp:", n_nan)

    mybins = numpy.arange(-1000, +1500, 30)
    plt.figure()
    plt.hist(ts_discarded)
    plt.title(
        "Discarded shot timestamps. N.: %d, of which %d nan" % (n_discarded, n_nan)
    )
    plt.xlabel("Timestamp [fs]")
    plt.savefig("hist_ts_discarded_thresh_%.2f.png" % thr)
    plt.close()

    Is_selected = numpy.hstack(Is_selected)
    print("N. of selected shots", Is_selected.shape[0])

    plt.figure()
    plt.hist(Is_selected)
    plt.title("Selected shot intensities. N.: %d" % Is_selected.shape[0])
    plt.xlabel("Intensity")
    plt.savefig("hist_Is_thresh_%.2f.png" % thr)
    plt.close()

    Is_discarded = numpy.hstack(Is_discarded)
    print("N. discarded shots:", Is_discarded.shape[0])

    plt.figure()
    plt.hist(Is_discarded)
    plt.title("Discarded shot intensities. N.: %d" % Is_discarded.shape[0])
    plt.xlabel("Intensity")
    plt.savefig("hist_Is_discarded_thresh_%.2f.png" % thr)
    plt.close()

    print(float(len(ts_selected)) / len(ts_discarded))


def apply_thr():
    thr = 0.06
    n = 0
    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\nScan", scan_n, "run range: ", run_start, run_end)
        for run_n in range(run_start, run_end + 1):
            try:
                data = joblib.load(
                    "./run_pid_ts/scan_%d_run_%0.6d_event_pid_ts_tsrel.jbl"
                    % (scan_n, run_n)
                )
                # print data.shape
                idxs = numpy.argwhere(data[:, 3] > thr).flatten()
                data_thr = data[idxs, :]
                if numpy.isnan(data_thr.sum()):
                    "Scan", scan_n, ", run ", run_n, " contains nans"
                else:
                    fn = (
                        "./run_pid_ts_thr/scan_%d_run_%0.6d_event_pid_ts_tsrel_thr_%0.2f"
                        % (scan_n, run_n, thr)
                    )
                    joblib.dump(data_thr, "%s.jbl" % fn)
                    myDict = {"data_thr": data_thr}
                    scipy.io.savemat("%s.mat" % fn, myDict)
            except:
                print("Scan", scan_n, ", run ", run_n, " not existing")
                n = n + 1

    print("N. non existing runs: ", n)


def match_unique_ID():
    thr = 0.06
    n = 0
    n_tot = 0
    for scan_n in range(1, 24):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\nScan", scan_n, "run range: ", run_start, run_end)
        for run_n in range(run_start, run_end + 1):
            path_string = (
                "/sf/alvra/data/p18594/raw/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.h5_event_"
                % (scan_n, run_n)
            )
            try:
                data = joblib.load(
                    "./run_pid_ts_thr/scan_%d_run_%0.6d_event_pid_ts_tsrel_thr_%0.2f.jbl"
                    % (scan_n, run_n, thr)
                )
                n_frames = data.shape[0]
                n_tot = n_tot + n_frames
                unique_ID_timed = []
                for i in range(n_frames):
                    event_n = str(int(data[i, 0]))
                    unique_ID = path_string + event_n
                    unique_ID_timed.append(unique_ID)

                fn = "./run_pid_ts_thr/scan_%d_run_%0.6d_uniqueIDs_thr_%0.2f" % (
                    scan_n,
                    run_n,
                    thr,
                )
                joblib.dump(unique_ID_timed, "%s.jbl" % fn)
            #                myDict = {"unique_ID_timed": unique_ID_timed}
            #                scipy.io.savemat('%s.mat'%fn, myDict)
            except:
                print("Scan", scan_n, ", run ", run_n, " not existing")
                n = n + 1

    print("N. non existing runs: ", n)
    print("Total n timed frames", n_tot)


def export_unique_ID():
    thr = 0.06
    n = 0

    fn_out = "./run_pid_ts_thr/scans_20_to_23_uniqueIDs_ts.txt"
    fn_write = open(fn_out, "w")
    for scan_n in range(20, 23 + 1):
        run_start, run_end = run_ns.run_ns(scan_n)
        print("\nScan", scan_n, "run range: ", run_start, run_end)

        for run_n in range(run_start, run_end + 1):
            try:
                data_ts = joblib.load(
                    "./run_pid_ts_thr/scan_%d_run_%0.6d_event_pid_ts_tsrel_thr_%0.2f.jbl"
                    % (scan_n, run_n, thr)
                )
                ts = data_ts[:, 2]
                n_frames = ts.shape[0]

                data_IDs = joblib.load(
                    "./run_pid_ts_thr/scan_%d_run_%0.6d_uniqueIDs_thr_%0.2f.jbl"
                    % (scan_n, run_n, thr)
                )
                if abs(n_frames - len(data_IDs)) > 0:
                    print("Problem")
                for i in range(n_frames):
                    fn_write.write("%s  %.1f\n" % (data_IDs[i], ts[i]))
            except:
                print("Scan", scan_n, ", run ", run_n, " not existing")
                n = n + 1
    fn_write.close()
    print("N. non existing runs: ", n)


if __name__ == "__main__":
    get_ts_distribution()
#    apply_thr()
#    match_unique_ID()
#    export_unique_ID()
