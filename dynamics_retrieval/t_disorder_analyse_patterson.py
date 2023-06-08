# -*- coding: utf-8 -*-

# Produce and visualise patterson maps
# ./generate_mtzs_rho
# cctbx.patterson_map blahblah.mtz
# mapslicer blahblah_patt.ccp4

### To use this code:
# conda activate myenv_gemmi
# python t_disorder_analyse_patterson
# conda deactivate

import gemmi
import matplotlib
import numpy

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.


import joblib
import matplotlib.pylab
from matplotlib import pyplot

# ccp4 = gemmi.read_ccp4_map('./rho_dark_ortho_patt.ccp4')
# ccp4.setup()
# arr = numpy.array(ccp4.grid, copy=False)
# print(ccp4.grid.unit_cell.a, ccp4.grid.unit_cell.b)
# x = numpy.linspace(0, ccp4.grid.unit_cell.a, num=arr.shape[0], endpoint=False)
# y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
# X, Y = numpy.meshgrid(x, y, indexing='ij')
# pyplot.contourf(X, Y, arr[:,:,0], vmin=0, vmax=50)
# pyplot.gca().set_aspect('equal', adjustable='box')
# pyplot.colorbar()
# pyplot.savefig('./rho_dark_ortho_patt_z0.png')
# pyplot.close()
#
# pyplot.plot(y,arr[0,:,0])
# pyplot.savefig('./rho_dark_ortho_patt_x0_z0.png')
# pyplot.close()


path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA/translation_correction_light"
label = "I_light_avg"
outfolder = "%s/method_bf" % path


pyplot.figure(figsize=(14, 14))
pyplot.ylim([-20, +50])

fn = "%s/ccp4s/%s_original_patt.ccp4" % (outfolder, label)
ccp4 = gemmi.read_ccp4_map(fn)
ccp4.setup()
arr = numpy.array(ccp4.grid, copy=False)
y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
pyplot.plot(y,arr[0,:,0],label='uncorrected')

t_d_y_Ang = 0.245 * ccp4.grid.unit_cell.b
diff = abs(y-t_d_y_Ang)
idx_t1 = numpy.argmin(diff)
diff = abs(y-2*t_d_y_Ang)
idx_t2 = numpy.argmin(diff)

t1_uncorrected = arr[0,idx_t1,0]
t2_uncorrected = arr[0,idx_t2,0]
print((t1_uncorrected, t2_uncorrected))

pyplot.tight_layout()
pyplot.savefig("%s/%s_original_patt_x0_z0.png" % (outfolder, label), dpi=96 * 2)
pyplot.close()


flag = 0
if flag == 1:
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
    tp_unique = list(tp_unique)

    alpha_lst = []
    beta_lst = []
    gamma_lst = []
    patt_t1_lst = []
    patt_t2_lst = []

    for n, fraction_set in enumerate(tp_unique):
        print(n)
        fraction_alpha = fraction_set[0]
        fraction_beta = fraction_set[1]
        fraction_gamma = fraction_set[2]

        print((
            "\nTesting domain fractions: %.2f %.2f %.2f"
            % (fraction_alpha, fraction_beta, fraction_gamma)
        ))
        fn = "%s/ccp4s/%s_corrected_frac_%.2f_%.2f_%.2f_patt.ccp4" % (
            outfolder,
            label,
            fraction_alpha,
            fraction_beta,
            fraction_gamma,
        )
        ccp4 = gemmi.read_ccp4_map(fn)
        ccp4.setup()
        arr = numpy.array(ccp4.grid, copy=False)
        print((arr[0, idx_t1, 0], arr[0, idx_t2, 0]))
        if (
            abs(arr[0, idx_t1, 0]) < abs(t1_uncorrected) / 4
            and abs(arr[0, idx_t2, 0]) < abs(t2_uncorrected) / 2
        ):
            # if (abs(arr[0,idx_t1,0]) < abs(t1_uncorrected)/2 and abs(arr[0,idx_t2,0]) < abs(t2_uncorrected)):

            alpha_lst.append(fraction_alpha)
            beta_lst.append(fraction_beta)
            gamma_lst.append(fraction_gamma)
            patt_t1_lst.append(abs(arr[0, idx_t1, 0]))
            patt_t2_lst.append(abs(arr[0, idx_t2, 0]))

    joblib.dump(alpha_lst, "%s/lst_alpha.jbl" % outfolder)
    joblib.dump(beta_lst, "%s/lst_beta.jbl" % outfolder)
    joblib.dump(gamma_lst, "%s/lst_gamma.jbl" % outfolder)
    joblib.dump(patt_t1_lst, "%s/lst_patt_t1.jbl" % outfolder)
    joblib.dump(patt_t2_lst, "%s/lst_patt_t2.jbl" % outfolder)

    matplotlib.pyplot.scatter(
        alpha_lst,
        beta_lst,
        s=20,
        c=patt_t1_lst,
        alpha=0.6,
        cmap="cool",
        edgecolors=None,
    )
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=7)
    matplotlib.pyplot.savefig("%s/abs_patt_t1_selected.png" % outfolder, dpi=96 * 4)
    matplotlib.pyplot.close()

    matplotlib.pyplot.scatter(
        alpha_lst,
        beta_lst,
        s=20,
        c=patt_t2_lst,
        alpha=0.6,
        cmap="cool",
        edgecolors=None,
    )
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=7)
    matplotlib.pyplot.savefig("%s/abs_patt_t2_selected.png" % outfolder, dpi=96 * 4)
    matplotlib.pyplot.close()

flag = 1
if flag == 1:

    pyplot.figure(figsize=(14, 14))
    pyplot.ylim([-20, +50])

    fn = "%s/ccp4s/%s_original_patt.ccp4" % (outfolder, label)
    ccp4 = gemmi.read_ccp4_map(fn)
    ccp4.setup()
    arr = numpy.array(ccp4.grid, copy=False)
    y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
    pyplot.plot(y, arr[0, :, 0], label="uncorrected", c="b")

    # SWissfel
    fn = "%s/ccp4s/%s_corrected_frac_0.00_0.14_0.86_patt.ccp4" % (outfolder, label)
    # SACLA
    # fn = '%s/ccp4s/%s_corrected_frac_0.00_0.15_0.85_patt.ccp4'%(outfolder, label)

    ccp4 = gemmi.read_ccp4_map(fn)
    ccp4.setup()
    arr = numpy.array(ccp4.grid, copy=False)
    y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
    pyplot.plot(y, arr[0, :, 0], label="corrected", c="m")
    matplotlib.pyplot.gca().tick_params(axis="both", labelsize=28)

    matplotlib.pyplot.axhline(y=0, c="k")
    matplotlib.pyplot.legend(frameon=False, fontsize=34)
    matplotlib.pyplot.xlabel(r"$y \;\; [\AA]$", fontsize=34),
    matplotlib.pyplot.ylabel(
        "Patterson map value", fontsize=34, rotation=90, labelpad=4
    )

    pyplot.tight_layout()
    pyplot.savefig(
        "%s/%s_corrected_frac_0.00_0.14_0.86_patt_x0_z0.png" % (outfolder, label),
        dpi=96 * 2,
    )
    pyplot.close()