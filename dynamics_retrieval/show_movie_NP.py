# -*- coding: utf-8 -*-
import os
import pickle

import joblib
import matplotlib.pylab as pl
import numpy
import settings_NP as settings

# import matplotlib


results_path = settings.results_path

modes = [8, 9]
label = "8_9"

# Load  average
# f = open('%s/T_average.pkl'%results_path, 'rb')
# data_ave = pickle.load(f)
# f.close()
# t_points = 2400
# data_ave = numpy.repeat(data_ave[:, :, numpy.newaxis], t_points, axis=2)
# data_ave = data_ave[:-1, :-1, 0:t_points]

t_points = 8381
# Exclude last latitude point and last longitude point
movie_combined = numpy.zeros((16 - 1, 46 - 1, t_points))
for mode in modes:
    print(mode)
    movie = joblib.load(
        "%s/movie_2D_mode_%d_extended_fwd_bwd.jbl" % (results_path, mode)
    )
    movie = movie[:-1, :-1, 0:t_points]
    movie_combined = movie_combined + movie

# movie_combined = movie_combined + data_ave

outdir = "%s/movie_modes_%s_extended_fwd_bwd" % (results_path, label)
if not os.path.exists(outdir):
    os.mkdir(outdir)

v = 0.2

pl.imshow(movie_combined[:, :, 0], vmin=-v, vmax=v, origin="lower", cmap="jet")
# pl.imshow(movie_combined[:,:,0], vmin=-1, vmax=26, origin='lower', cmap='jet')
pl.colorbar()

n = -1
for i in range(t_points):  # (1176,2400):
    pl.cla()
    pl.imshow(movie_combined[:, :, i], vmin=-v, vmax=v, origin="lower", cmap="jet")
    # pl.imshow(movie_combined[:,:,i], vmin=-1, vmax=26, origin='lower', cmap='jet')
    if i % 12 == 0:
        n = n + 1
    pl.title("Year: %d   Month: %d" % (n, i % 12))
    pl.savefig("%s/Modes_%s_index_%05d_extended_fwd_bwd.png" % (outdir, label, i))

print("Done!")

# Look for 1st frame in PNAS movies
# idxs_candidates = []
# for i in range(4800):
#    img = movie_combined[:,:,i]
#    a = img[7:8,8:11]
#    b = img[8:9,10:12]
#    c = img[7,11]
#    if numpy.all([a<-0.15]) and numpy.all([b>0.15]) and c<img[8,11] and img[7,12]<img[7,11]:
#        idxs_candidates.append(i)
#        matplotlib.pyplot.figure()
#        matplotlib.pyplot.imshow(img, origin='lower', vmin=-0.2, vmax=+0.2, cmap='jet')
#        matplotlib.pyplot.colorbar()
#        matplotlib.pyplot.savefig('%s/candidates_modes_9_10/%05d.png'%(results_path,i))
#        matplotlib.pyplot.close()
