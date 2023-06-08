# -*- coding: utf-8 -*-
import joblib
import numpy
import settings_NP as settings

results_path = settings.results_path
print(results_path)
q = settings.q
ncopies = settings.ncopies


test = joblib.load("%s/X_reconstructed_1.jbl" % (results_path))
print(test.shape)


for i in range(7):
    print(i)
    movie_opt = joblib.load("%s/movie_mode_%d_optimised.jbl" % (results_path, i))
    movie = joblib.load("%s/movie_mode_%d.jbl" % (results_path, i))

    diff = movie - movie_opt
    print(numpy.amax(diff), numpy.amin(diff))
    print(movie.shape)
    m = movie.shape[0]
    n_reconstructed = movie.shape[1]  # S-2q
    S = n_reconstructed + 2 * q
    s = S - q

    movie_ext_forward = joblib.load(
        "%s/movie_mode_%d_extended_fwd.jbl" % (results_path, i)
    )
    print(movie_ext_forward.shape)
    if abs(movie_ext_forward.shape[1] - (s - ncopies + 1)) > 0:
        print("Problem")
    diff = movie_ext_forward[:, 0 : (S - 2 * q)] - movie
    print(numpy.amax(diff), numpy.amin(diff))

    movie_ext_fwd_bwd = joblib.load(
        "%s/movie_mode_%d_extended_fwd_bwd.jbl" % (results_path, i)
    )
    print(movie_ext_fwd_bwd.shape)
    diff = movie_ext_fwd_bwd[:, q - ncopies : q - ncopies + (S - 2 * q)] - movie
    print(numpy.amax(abs(diff)))

    movie_ext_forward_opt = joblib.load(
        "%s/movie_mode_%d_optimised_extended_fwd.jbl" % (results_path, i)
    )
    print(movie_ext_forward_opt.shape)
    diff = movie_ext_forward - movie_ext_forward_opt
    print(numpy.amax(diff), numpy.amin(diff))

    movie_ext_fwd_bwd_opt = joblib.load(
        "%s/movie_mode_%d_optimised_extended_fwd_bwd.jbl" % (results_path, i)
    )
    print(movie_ext_fwd_bwd_opt.shape)
    diff = movie_ext_fwd_bwd_opt - movie_ext_fwd_bwd
    print(numpy.amax(diff), numpy.amin(diff))

    movie_ext_fwd_bwd_para = joblib.load(
        "%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl" % (results_path, i)
    )
    print(movie_ext_fwd_bwd_para.shape)
    diff = movie_ext_fwd_bwd_para - movie_ext_fwd_bwd
    print(numpy.amax(diff), numpy.amin(diff))
