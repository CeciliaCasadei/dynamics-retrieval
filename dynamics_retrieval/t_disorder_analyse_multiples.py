# -*- coding: utf-8 -*-
import matplotlib
import numpy

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.
import os
import warnings

import matplotlib.pyplot
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# VALUES FOR RHODOPSIN
t_d_y = 0.245
a = 61.45
b = 90.81
c = 150.82
# Rho data set: swissfel_combined_dark_xshere_precise_1ps-dark
# a =   61.51
# b =   91.01
# c =  151.11


def get_reso_P222(a, b, c, h, k, l):
    return 1.0 / numpy.sqrt(
        (float(h) / a) ** 2 + (float(k) / b) ** 2 + (float(l) / c) ** 2
    )


def get_I_avg_vs_k(fn):
    # The first 4 columns must be: h k l I
    # Subsequent columns and number of spaces do not matter.
    # Calculate, for each value of Miller index k, the average intensity.
    Is_avg = []

    for k in range(0, 40):
        # print k
        fo = open(fn, "r")
        Is = []
        for i in fo:
            # print i
            if int(i.split()[1]) == k:
                h = int(i.split()[0])
                l = int(i.split()[2])
                Intensity = float(i.split()[3])
                d = get_reso_P222(a, b, c, h, k, l)
                if d > 1.8:
                    Is.append(Intensity)
        if len(Is) > 0:
            Is = numpy.asarray(Is)
            I_avg = numpy.average(Is)
            Is_avg.append(I_avg)
        fo.close()
    return Is_avg


def get_I_avg_vs_h(fn):
    # The first 4 columns must be: h k l I
    # Subsequent columns and number of spaces do not matter.
    # Calculate, for each value of Miller index h, the average intensity.
    Is_avg = []

    for h in range(0, 40):
        # print k
        fo = open(fn, "r")
        Is = []
        for i in fo:
            # print i
            if int(i.split()[0]) == h:
                k = int(i.split()[1])
                l = int(i.split()[2])
                Intensity = float(i.split()[3])
                d = get_reso_P222(a, b, c, h, k, l)
                if d > 1.8:
                    Is.append(Intensity)
        if len(Is) > 0:
            Is = numpy.asarray(Is)
            I_avg = numpy.average(Is)
            Is_avg.append(I_avg)
        fo.close()
    return Is_avg


def get_I_avg_vs_l(fn):
    # The first 4 columns must be: h k l I
    # Subsequent columns and number of spaces do not matter.
    # Calculate, for each value of Miller index k, the average intensity.
    Is_avg = []

    for l in range(0, 40):
        # print k
        fo = open(fn, "r")
        Is = []
        for i in fo:
            # print i
            if int(i.split()[2]) == l:
                h = int(i.split()[0])
                k = int(i.split()[1])
                Intensity = float(i.split()[3])
                d = get_reso_P222(a, b, c, h, k, l)
                if d > 1.8:
                    Is.append(Intensity)
        if len(Is) > 0:
            Is = numpy.asarray(Is)
            I_avg = numpy.average(Is)
            Is_avg.append(I_avg)
        fo.close()
    return Is_avg


def correct(f_w, f_r, f_alpha, f_beta):
    print "Writing corrected intensities in: ", f_w
    fo_w = open(f_w, "w")
    fo_r = open(f_r, "r")
    f_gamma = 1 - f_alpha - f_beta
    sum_of_sq = f_alpha * f_alpha + f_beta * f_beta + f_gamma * f_gamma
    for i in fo_r:
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[4])

        factor = 1.0 / (
            sum_of_sq
            + 2
            * (f_alpha * f_beta + f_beta * f_gamma)
            * numpy.cos(2 * numpy.pi * k * t_d_y)
            + 2 * f_alpha * f_gamma * numpy.cos(2 * numpy.pi * k * 2 * t_d_y)
        )
        I_corr = factor * I
        sigI_corr = factor * sigI
        fo_w.write("%4d%5d%5d%15.2f%15.2f\n" % (h, k, l, I_corr, sigI_corr))

    fo_w.close()
    fo_r.close()


def rewrite(
    f_w,
    f_r,
):
    fo_w = open(f_w, "w")
    fo_r = open(f_r, "r")

    for i in fo_r:
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[5])
        fo_w.write("%4d%5d%5d%15.2f%15.2f\n" % (h, k, l, I, sigI))

    fo_w.close()
    fo_r.close()


def plot(I_avg, fign, xlabel):
    print "Plotting average intensities in: ", fign
    matplotlib.pyplot.figure(figsize=(20, 8))
    matplotlib.pyplot.plot(range(len(I_avg)), I_avg, "-o")
    matplotlib.pyplot.xlabel(r"$%s$" % xlabel, fontsize=34),
    matplotlib.pyplot.ylabel(
        r"$\left< I \right>_%s$" % xlabel, fontsize=34, rotation=0, labelpad=30
    )
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(fign, dpi=96 * 2)
    matplotlib.pyplot.close()


if __name__ == "__main__":

    path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA/translation_correction_dark_from_scan"
    # label = 'sacla18_combined_dark_100ps_std_xsphere-combi_dark'
    label = "rho_dark_from_scans"
    # label = 'swissfel_combined_dark_xsphere_precise_1ps-1ps_light_good'

    # Get average I vs k
    filename = "%s/%s.txt" % (path, label)
    Is_avg = get_I_avg_vs_k(filename)

    # Plot
    fign = "%s/%s_original_Is.png" % (path, label)
    plot(Is_avg, fign, "k")

    # Rewrite original in different format
    fo_w_fn_original = "%s/%s_original.txt" % (path, label)
    rewrite(fo_w_fn_original, filename)

    # Get average I vs h
    filename = "%s/%s.txt" % (path, label)
    Is_avg_vs_h = get_I_avg_vs_h(filename)
    # Plot
    fign = "%s/%s_original_Is_vs_h.png" % (path, label)
    plot(Is_avg_vs_h, fign, "h")

    # Get average I vs l
    filename = "%s/%s.txt" % (path, label)
    Is_avg_vs_l = get_I_avg_vs_l(filename)
    # Plot
    fign = "%s/%s_original_Is_vs_l.png" % (path, label)
    plot(Is_avg_vs_l, fign, "l")

    ### METHOD: BRUTE FORCE ###
    flag = 1
    if flag == 1:
        out_path = "%s/method_bf" % path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print "\n*** Method: brute force (%s) ***" % out_path
        print "\n*** Determine translated domain fraction. ***"
        fn_r = "%s/%s_original.txt" % (path, label)

        tp_lst = []
        fractions_alpha = numpy.arange(0.00, 0.51, 0.01)
        fractions_beta = numpy.arange(0.00, 0.51, 0.01)
        for fraction_alpha in fractions_alpha:
            fraction_alpha = round(fraction_alpha, 2)
            for fraction_beta in fractions_beta:
                fraction_beta = round(fraction_beta, 2)
                fraction_gamma = 1 - fraction_alpha - fraction_beta
                fraction_gamma = round(fraction_gamma, 2)

                tp = (fraction_alpha, fraction_beta, fraction_gamma)
                tp_s = tuple(sorted(tp))
                tp_lst.append(tp_s)  # list of sorted tuples
        tp_unique = set(tp_lst)
        print len(tp_unique)

        for fraction_set in tp_unique:
            fraction_alpha = fraction_set[0]
            fraction_beta = fraction_set[1]
            fraction_gamma = fraction_set[2]

            print "\nTesting domain fractions: %.2f %.2f %.2f" % (
                fraction_alpha,
                fraction_beta,
                fraction_gamma,
            )

            # Correct intensities
            fn_w = "%s/%s_corrected_frac_%.2f_%.2f_%.2f.txt" % (
                out_path,
                label,
                fraction_alpha,
                fraction_beta,
                fraction_gamma,
            )
            correct(fn_w, fn_r, fraction_alpha, fraction_beta)

            # Get average corrected I vs k
            Is_avg = get_I_avg_vs_k(fn_w)

            # Plot
            fign = "%s/%s_corrected_frac_%.2f_%.2f_%.2f.png" % (
                out_path,
                label,
                fraction_alpha,
                fraction_beta,
                fraction_gamma,
            )
            plot(Is_avg, fign, "k")

    # Plot summary figure
    flag = 0
    if flag == 1:
        print "\n*** Summary figure ***"
        out_path = "%s/method_bf" % path

        # Get average I vs k
        filename = "%s/%s.txt" % (path, label)
        Is_uncorrected = get_I_avg_vs_k(filename)

        # colors = matplotlib.pylab.cm.viridis(numpy.linspace(0,1,7))

        matplotlib.pyplot.figure(figsize=(20, 8))

        matplotlib.pyplot.xlabel(r"$k$", fontsize=34),
        matplotlib.pyplot.ylabel(
            r"$\left< I \right>_k$", fontsize=34, rotation=0, labelpad=30
        )
        matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)

        matplotlib.pyplot.plot(
            range(len(Is_uncorrected)), Is_uncorrected, "-o", label="uncorrected", c="b"
        )

        # Swissfel
        fn = "%s/txts/%s_corrected_frac_0.01_0.19_0.80.txt" % (out_path, label)

        # SACLA
        # fn = '%s/txts/%s_corrected_frac_0.00_0.15_0.85.txt'%(out_path, label)
        Is_avg = get_I_avg_vs_k(fn)

        matplotlib.pyplot.plot(
            range(len(Is_avg)), Is_avg, "-o", label="corrected", c="m"
        )

        matplotlib.pyplot.legend(frameon=False, fontsize=34)
        fign = "%s/%s_corrected_Is_xlim.png" % (out_path, label)
        matplotlib.pyplot.xlim((0, 40))
        matplotlib.pyplot.gca().axes.xaxis.set_ticklabels([])
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(fign)
        matplotlib.pyplot.close()
        print "Summary figure in:", fign

        ##########
        # Get average I vs h
        filename = "%s/%s.txt" % (path, label)
        Is_uncorrected = get_I_avg_vs_h(filename)

        # colors = matplotlib.pylab.cm.viridis(numpy.linspace(0,1,7))

        matplotlib.pyplot.figure(figsize=(20, 8))

        matplotlib.pyplot.xlabel(r"$h$", fontsize=34),
        matplotlib.pyplot.ylabel(
            r"$\left< I \right>_h$", fontsize=34, rotation=0, labelpad=30
        )
        matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)

        matplotlib.pyplot.plot(
            range(len(Is_uncorrected)), Is_uncorrected, "-o", label="uncorrected", c="b"
        )

        # Swissfel
        fn = "%s/txts/%s_corrected_frac_0.01_0.19_0.80.txt" % (out_path, label)

        # SACLA
        # fn = '%s/txts/%s_corrected_frac_0.00_0.15_0.85.txt'%(out_path, label)
        Is_avg = get_I_avg_vs_h(fn)

        matplotlib.pyplot.plot(
            range(len(Is_avg)), Is_avg, "-o", label="corrected", c="m"
        )

        matplotlib.pyplot.legend(frameon=False, fontsize=34)
        fign = "%s/%s_corrected_Is_vs_h.png" % (out_path, label)
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(fign)
        matplotlib.pyplot.close()
        print "Summary figure in:", fign

        ##########
        # Get average I vs h
        filename = "%s/%s.txt" % (path, label)
        Is_uncorrected = get_I_avg_vs_l(filename)

        # colors = matplotlib.pylab.cm.viridis(numpy.linspace(0,1,7))

        matplotlib.pyplot.figure(figsize=(20, 8))

        matplotlib.pyplot.xlabel(r"$l$", fontsize=34),
        matplotlib.pyplot.ylabel(
            r"$\left< I \right>_l$", fontsize=34, rotation=0, labelpad=30
        )
        matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)

        matplotlib.pyplot.plot(
            range(len(Is_uncorrected)), Is_uncorrected, "-o", label="uncorrected", c="b"
        )

        # Swissfel
        fn = "%s/txts/%s_corrected_frac_0.01_0.19_0.80.txt" % (out_path, label)

        # SACLA
        # fn = '%s/txts/%s_corrected_frac_0.00_0.15_0.85.txt'%(out_path, label)
        Is_avg = get_I_avg_vs_l(fn)

        matplotlib.pyplot.plot(
            range(len(Is_avg)), Is_avg, "-o", label="corrected", c="m"
        )

        matplotlib.pyplot.legend(frameon=False, fontsize=34)
        fign = "%s/%s_corrected_Is_vs_l.png" % (out_path, label)
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(fign)
        matplotlib.pyplot.close()
        print "Summary figure in:", fign
