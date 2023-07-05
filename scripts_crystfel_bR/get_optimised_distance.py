# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot
import numpy
import scipy.optimize

# for i in {60..65}; do python get_optimised_distance.py $i; done


def quadratic(x, x0, offset, a, b):
    y = offset + a * (x - x0) + b * (x - x0) ** 2
    return y


def f(run_n):
    # out_dir = 'run_%s'%run_n
    out_figs = "./figs"
    if not os.path.exists(out_figs):
        os.mkdir(out_figs)

    stream_list = os.popen("ls %s/*.stream" % out_dir).read()
    stream_list = stream_list.split("\n")[:-1]

    a_avg_list = []
    a_std_list = []
    b_avg_list = []
    b_std_list = []
    c_avg_list = []
    c_std_list = []
    idx_rate_list = []
    ds = []
    std_cmb_list = []

    for stream_file in stream_list:
        try:
            print(stream_file)

            img_lines = os.popen('grep "Image filename" %s' % stream_file).read()
            img_lines = img_lines.split("\n")[:-1]
            n_tot = len(img_lines)
            print(n_tot)

            uc_lines = os.popen('grep "Cell" %s' % stream_file).read()
            uc_lines = uc_lines.split("\n")[:-1]
            n_idxd = len(uc_lines)
            print(n_idxd)

            idx_rate = float(n_idxd) / n_tot
            print(idx_rate)
            idx_rate_list.append(idx_rate)

            det_dist = float("0.0" + stream_file.split(".")[-3][-5:])
            print(det_dist)
            ds.append(det_dist)

            a_s = []
            b_s = []
            c_s = []
            for uc_line in uc_lines:
                # print uc_line
                uc_split = uc_line.split(" ")
                a = float(uc_split[2])
                b = float(uc_split[3])
                c = float(uc_split[4])
                # print a, b, c
                a_s.append(a)
                b_s.append(b)
                c_s.append(c)

            a_s = numpy.asarray(a_s)
            b_s = numpy.asarray(b_s)
            c_s = numpy.asarray(c_s)

            a_avg = numpy.average(a_s)
            a_avg_list.append(a_avg)
            b_avg = numpy.average(b_s)
            b_avg_list.append(b_avg)
            c_avg = numpy.average(c_s)
            c_avg_list.append(c_avg)

            a_std = numpy.std(a_s)
            a_std_list.append(a_std)
            b_std = numpy.std(b_s)
            b_std_list.append(b_std)
            c_std = numpy.std(c_s)
            c_std_list.append(c_std)

            std_cmb = numpy.sqrt(a_std ** 2 + b_std ** 2 + c_std ** 2)
            std_cmb_list.append(std_cmb)

            print("a: %.3f +- %.3f A" % (10 * a_avg, 10 * a_std))
            print("b: %.3f +- %.3f A" % (10 * b_avg, 10 * b_std))
            print("c: %.3f +- %.3f A" % (10 * c_avg, 10 * c_std))
        except:
            print("Exception")

    p0 = [0.082974, 0.0, 1.0, 1.0]
    popt, pcov = scipy.optimize.curve_fit(quadratic, ds, std_cmb_list, p0)
    ys = quadratic(ds, *popt)
    d_opt = ds[numpy.argmin(ys)]

    fig = matplotlib.pyplot.figure()

    ax1 = fig.add_subplot(5, 1, 1)
    matplotlib.pyplot.title("Run %s, optimised d = %.6f" % (run_n, d_opt))
    ax1.plot(ds, std_cmb_list, "b")
    ax1.plot(ds, ys, "m")
    ax1.axvline(x=d_opt, ymin=0, ymax=1, c="k")
    ax1.set_ylabel("std_cmb")
    ax1.tick_params(axis="x", labelbottom="off", direction="in")

    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(ds, a_std_list, "b")
    ax2.set_ylabel("std(a)")
    ax2.tick_params(axis="x", labelbottom="off", direction="in")

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(ds, b_std_list, "b")
    ax3.set_ylabel("std(b)")
    ax3.tick_params(axis="x", labelbottom="off", direction="in")

    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(ds, c_std_list, "b")
    ax4.set_ylabel("std(c)")
    ax4.tick_params(axis="x", labelbottom="off", direction="in")

    ax5 = fig.add_subplot(5, 1, 5)
    ax5.plot(ds, idx_rate_list, "b")
    matplotlib.pyplot.setp(ax5.get_xticklabels(), rotation=45)
    ax5.set_ylabel("Idx rate")
    ax5.set_xlabel("Detector distance [m]")

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig("%s/out_r%s.png" % (out_figs, run_n))
    matplotlib.pyplot.close()

    return d_opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Determine detector distance that minimises std(a)"
    )
    parser.add_argument(
        "run_n", type=int, help="A required integer positional argument: run number"
    )
    args = parser.parse_args()
    run_n_str = str("%0.4d" % args.run_n)
    print(run_n_str)
    out_dir = "run_%s" % run_n_str
    if os.path.exists(out_dir):
        d_optimised = f(run_n_str)
        print("Optimal distance: ", d_optimised)

        glob_path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/indexing"
        idx_dir = "%s/indexing" % glob_path
        if not os.path.exists(idx_dir):
            os.mkdir(idx_dir)
        idx_dir_run = "%s/run_%s" % (idx_dir, run_n_str)
        if not os.path.exists(idx_dir_run):
            os.mkdir(idx_dir_run)

        geom_start = "%s/geometry/start.geom" % glob_path
        os.system("cp %s %s/geo_optimised_d.geo" % (geom_start, idx_dir_run))
        os.system(
            "sed -i 's/clen.*/clen = %.6f/' %s/geo_optimised_d.geo"
            % (d_optimised, idx_dir_run)
        )
        os.system(
            "ln -s %s/event_lists/lists_new/event_list_run_%s.txt %s"
            % (glob_path, run_n_str, idx_dir_run)
        )
        os.system("ln -s %s/cell/BR.cell %s" % (glob_path, idx_dir_run))

        start_idx_flag = 1
        if start_idx_flag == 1:
            list_file = "%s/event_list_run_%s.txt" % (idx_dir_run, run_n_str)
            geo_file = "%s/geo_optimised_d.geo" % idx_dir_run
            stream_dir = "%s/streams" % idx_dir_run
            os.system(
                "./turbo_index.sh %s %s %s %s"
                % (list_file, run_n_str, geo_file, stream_dir)
            )
