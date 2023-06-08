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
# a =   61.45
# b =   90.81
# c =  150.82
# Rho data set: swissfel_combined_dark_xshere_precise_1ps-dark
a = 61.51
b = 91.01
c = 151.11


def get_reso_P222(a, b, c, h, k, l):
    return 1.0 / numpy.sqrt(
        (float(h) / a) ** 2 + (float(k) / b) ** 2 + (float(l) / c) ** 2
    )


def f(x, A, B):
    return A + B * numpy.cos(2 * numpy.pi * x * t_d_y)


# def f_new(x, frac):
#     return (2*frac*frac - 2*frac + 1) + (2*frac*(1-frac))*numpy.cos(2*numpy.pi*x*t_d_y)


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
                I = float(i.split()[3])
                d = get_reso_P222(a, b, c, h, k, l)
                if d > 1.8:
                    Is.append(I)
        if len(Is) > 0:
            Is = numpy.asarray(Is)
            I_avg = numpy.average(Is)
            Is_avg.append(I_avg)
        fo.close()
    return Is_avg


def fit(Is_avg, k_range):

    xdata = list(range(k_range))
    ydata = numpy.asarray(Is_avg)[0:k_range]
    popt, pcov = curve_fit(f, xdata, ydata)

    A_opt = popt[0]
    B_opt = popt[1]

    r = B_opt / A_opt
    fraction = 0.5 * numpy.sqrt((1 - r) / (1 + r))
    fraction = 0.5 - fraction

    return A_opt, B_opt, fraction


def correct(f_w, f_r, A, B):
    print("Writing corrected intensities in: ", f_w)
    fo_w = open(f_w, "w")
    fo_r = open(f_r, "r")

    for i in fo_r:
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[4])

        factor = 1.0 / (A + B * numpy.cos(2 * numpy.pi * k * t_d_y))
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


def plot(I_avg, fign):
    print("Plotting average intensities in: ", fign)
    matplotlib.pyplot.figure(figsize=(20, 8))
    matplotlib.pyplot.plot(list(range(len(Is_avg))), Is_avg, "-o")
    matplotlib.pyplot.xlabel(r"$k$", fontsize=34),
    matplotlib.pyplot.ylabel(
        r"$\left< I \right>_k$", fontsize=34, rotation=0, labelpad=30
    )
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(fign, dpi=96 * 2)
    matplotlib.pyplot.close()


def get_A(fraction):
    return 2 * fraction * fraction - 2 * fraction + 1


def get_B(fraction):
    return 2 * fraction * (1 - fraction)


if __name__ == "__main__":

    path = "/das/work/p17/p17491/Cecilia_Casadei/rho_translation_correction_paper"
    label = "swissfel_combined_dark_xsphere_precise_1ps-dark"

    # Get average I vs k
    filename = "%s/%s.txt" % (path, label)
    Is_avg = get_I_avg_vs_k(filename)

    # Plot
    fign = "%s/%s_original_Is.pdf" % (path, label)
    plot(Is_avg, fign)

    # Rewrite original in different format
    fo_w_fn_original = "%s/%s_original.txt" % (path, label)
    rewrite(fo_w_fn_original, filename)

    print("\n*** Determine translated domain fraction. ***")
    fn_r = "%s/%s_original.txt" % (path, label)

    ### METHOD 1: FIT ###
    flag = 1
    if flag == 1:
        # DOES NOT GUARANTEE A+B = 1
        out_path = "%s/method_fit" % path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print("\n*** Method 1: fit (%s) ***" % out_path)
        print("(does not guarantee A+B=1)")

        # Fit
        k_max = 10  # For fitting purposes
        A, B, fraction_fit = fit(Is_avg, k_max)
        # fraction_fit = fit(Is_avg, k_max)
        # A = get_A(fraction_fit)
        # B = get_B(fraction_fit)

        print("Translated domain fraction: ", fraction_fit)
        print("A: ", A)
        print("B: ", B)
        print("A+B", A + B)

        # Correct intensities
        fn_w = "%s/%s_corrected_fraction_%.3f.txt" % (out_path, label, fraction_fit)
        correct(fn_w, fn_r, A, B)

        # Get average corrected I vs k
        Is_avg = get_I_avg_vs_k(fn_w)

        # Plot
        fign = "%s/%s_corrected_Is_fraction_%.3f.pdf" % (out_path, label, fraction_fit)
        plot(Is_avg, fign)

    ### METHOD 2: BRUTE FORCE ###
    flag = 0
    if flag == 1:
        out_path = "%s/method_bf" % path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print("\n*** Method 2: brute force (%s) ***" % out_path)
        print("(A+B=1)")

        fractions = numpy.arange(0.01, 0.50, 0.01)
        for fraction in fractions:
            print("\nTesting translated domain fraction: %.2f" % fraction)
            A = get_A(fraction)
            B = get_B(fraction)

            # Correct intensities
            fn_w = "%s/%s_corrected_frac_%.2f.txt" % (out_path, label, fraction)
            correct(fn_w, fn_r, A, B)

            # Get average corrected I vs k
            Is_avg = get_I_avg_vs_k(fn_w)

            # Plot
            fign = "%s/%s_corrected_Is_frac_%.2f.pdf" % (out_path, label, fraction)
            plot(Is_avg, fign)

    # Plot summary figure
    flag = 0
    if flag == 1:
        print("\n*** Summary figure ***")
        out_path = "%s/method_bf" % path

        # Get average I vs k
        filename = "%s/%s.txt" % (path, label)
        Is_uncorrected = get_I_avg_vs_k(filename)

        colors = matplotlib.pylab.cm.viridis(numpy.linspace(0, 1, 7))

        matplotlib.pyplot.figure(figsize=(20, 8))

        matplotlib.pyplot.xlabel(r"$k$", fontsize=34),
        matplotlib.pyplot.ylabel(
            r"$\left< I \right>_k$", fontsize=34, rotation=0, labelpad=30
        )
        matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)

        matplotlib.pyplot.plot(
            list(range(len(Is_uncorrected))),
            Is_uncorrected,
            "-o",
            label="uncorrected",
            c=colors[0],
        )

        fraction_best = 0.21
        fractions = numpy.arange(
            numpy.round(fraction_best, 2) - 0.03,
            numpy.round(fraction_best, 2) + 0.03,
            0.01,
        )
        for idx, fraction in enumerate(fractions):
            # Get average corrected I vs k
            fn = "%s/%s_corrected_frac_%.2f.txt" % (out_path, label, fraction)
            Is_avg = get_I_avg_vs_k(fn)
            # Plot
            matplotlib.pyplot.plot(
                list(range(len(Is_avg))),
                Is_avg,
                "-o",
                label=r"$\alpha=%.2f$" % fraction,
                c=colors[idx + 1],
            )

        matplotlib.pyplot.legend(frameon=False, fontsize=34)
        fign = "%s/%s_corrected_Is_frac_range.pdf" % (out_path, label)
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(fign)
        matplotlib.pyplot.close()
        print("Summary figure in:", fign)

    ### APPLY CORRECTION ###
    ### GUARANTEES A+B = 1
    flag = 0
    if flag == 1:
        fraction_best = 0.21
        out_path = "%s/final" % path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print("\n*** Apply results from fit (%s) ***" % out_path)
        print("(enforcing A+B=1)")

        A = get_A(fraction_best)
        B = get_B(fraction_best)

        # Correct intensities
        fn_w = "%s/%s_corrected_frac_%.2f.txt" % (out_path, label, fraction_best)
        correct(fn_w, fn_r, A, B)

        # Get average corrected I vs k
        Is_avg = get_I_avg_vs_k(fn_w)

        # Plot
        fign = "%s/%s_corrected_Is_frac_%.2f.png" % (out_path, label, fraction_best)
        plot(Is_avg, fign)
