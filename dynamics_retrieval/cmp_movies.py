# -*- coding: utf-8 -*-
import os
import pickle

import joblib
import matplotlib.pyplot as pl
import numpy

t_points = 2400

# NLSA of T anomaly
modes = [8, 9]

# Load  average
f = open("../data_Giannakis_PNAS_2012/results_1/T_average.pkl", "rb")
data_ave = pickle.load(f)
f.close()
data_ave = numpy.repeat(data_ave[:, :, numpy.newaxis], t_points, axis=2)
data_ave = data_ave[:-1, :-1, 0:t_points]

movie_combined = numpy.zeros((16 - 1, 46 - 1, t_points))
for mode in modes:
    print(mode)
    movie = joblib.load(
        "../data_Giannakis_PNAS_2012/results_1/movie_2D_mode_%d_norm_rightEV.jbl"
        % (mode)
    )
    movie = movie[:-1, :-1, 0:t_points]
    movie_combined = movie_combined + movie
movie_T_anomaly = movie_combined  # + data_ave

# NLSA of T full
modes = [0, 9, 10]
label = "0_9_10_minus_avgTfull"
movie_combined = numpy.zeros((16 - 1, 46 - 1, t_points))
for mode in modes:
    print(mode)
    movie = joblib.load(
        "../data_Giannakis_PNAS_2012/results_2_full_Ts/movie_2D_mode_%d_norm_rightEV.jbl"
        % (mode)
    )
    movie = movie[:-1, :-1, 0:t_points]
    movie_combined = movie_combined + movie
movie_T_full = movie_combined - data_ave

diff = movie_T_anomaly - movie_T_full
diff_sq = numpy.multiply(diff, diff)
diff_sq = diff_sq.flatten()
diff_sq = [diff_sq[i] for i in range(diff_sq.shape[0]) if not numpy.isnan(diff_sq[i])]
rmsd = numpy.sqrt(numpy.average(diff_sq))
print(rmsd)

v = 0.2
outdir = "../data_Giannakis_PNAS_2012/results_2_full_Ts/movie_mode_%s" % label
if not os.path.exists(outdir):
    os.mkdir(outdir)

pl.imshow(movie_T_full[:, :, 0], vmin=-v, vmax=v, origin="lower", cmap="jet")
# pl.imshow(movie_T_full[:,:,0], vmin=-1, vmax=26, origin='lower', cmap='jet')
pl.colorbar()

n = -1
for i in range(1176, 2400):  # 2400
    pl.cla()
    pl.imshow(movie_T_full[:, :, i], vmin=-v, vmax=v, origin="lower", cmap="jet")
    # pl.imshow(movie_combined[:,:,i], vmin=-1, vmax=26, origin='lower', cmap='jet')
    if i % 12 == 0:
        n = n + 1
    pl.title("Year: %d   Month: %d" % (n, i % 12))
    pl.savefig("%s/Modes_%s_index_%05d.png" % (outdir, label, i))

print("Done!")
