# -*- coding: utf-8 -*-
import os
import pickle

import matplotlib
import matplotlib.pylab as pl
import numpy
import settings

results_path = settings.results_path
l = settings.l

modes = list(range(l))
label = "all"

# Exclude last latitude point and last longitude point
movie_combined = numpy.zeros((16 - 1, 46 - 1, 4800))
for mode in modes:
    print(mode)
    f = open("%s/P_sym/movie_2D_mode_%d.pkl" % (results_path, mode), "rb")
    movie = pickle.load(f)
    f.close()
    movie = movie[:-1, :-1, :]
    movie_combined = movie_combined + movie


outdir = "%s/movie_modes_%s" % (results_path, label)
if not os.path.exists(outdir):
    os.mkdir(outdir)


pl.imshow(movie_combined[:, :, 0], origin="lower", cmap="jet")
pl.colorbar()

n = -1
for i in range(1176, 2400):
    pl.cla()
    pl.imshow(movie_combined[:, :, i], origin="lower", cmap="jet")
    if i % 12 == 0:
        n = n + 1
    pl.title("Year: %d   Month: %d" % (n, i % 12))
    pl.savefig("%s/Modes_%s_index_%05d.png" % (outdir, label, i))

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
#
