# -*- coding: utf-8 -*-
import pickle

import joblib
import numpy
import settings_NP as settings

results_path = settings.results_path
l = settings.l

f = open("%s/mask.pkl" % results_path, "rb")
mask = pickle.load(f)
f.close()

for mode in range(l):

    print(mode)
    x = joblib.load("%s/movie_mode_%d_extended_fwd_bwd.jbl" % (results_path, mode))

    t_pts = x.shape[1]

    print(t_pts)

    movie_2D = numpy.zeros((mask.shape[0], mask.shape[1], t_pts))

    for t in range(t_pts):
        frame = x[:, t]  # 16 chunks of 46 ?!?
        frame_2D = numpy.zeros(mask.shape)
        frame_2D[mask == 0] = numpy.nan
        test = frame_2D.T
        test[mask.T == 1] = frame
        movie_2D[:, :, t] = test.T

    joblib.dump(
        movie_2D, "%s/movie_2D_mode_%d_extended_fwd_bwd.jbl" % (results_path, mode)
    )


#    test = joblib.load('%s/movie_2D_mode_%d_norm_leftEV.jbl'%(results_path, mode))
#    diff = abs(movie_2D-test)
#    print numpy.nanmax(diff)
