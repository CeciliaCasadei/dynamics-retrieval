# -*- coding: utf-8 -*-
import time

import joblib
import numpy


def reconstruct(settings):
    print("\n****** RUNNING reconstruct ******")
    results_path = settings.results_path

    U = joblib.load("%s/U_norm_rightEV.jbl" % results_path)
    S = joblib.load("%s/S_norm_rightEV.jbl" % results_path)
    VT_final = joblib.load("%s/VT_final_norm_rightEV.jbl" % results_path)

    l = VT_final.shape[0]

    for i in range(l):
        print(i)
        X_r = S[i] * numpy.outer(U[:, i], VT_final[i, :])
        joblib.dump(X_r, "%s/X_reconstructed_%d.jbl" % (results_path, i))
        print(X_r.shape)


def unwrap(settings):
    print("\n****** RUNNING unwrap ******")
    results_path = settings.results_path
    l = settings.l
    q = settings.q
    ncopies = settings.ncopies

    for mode in range(l):

        print("Mode: ", mode)
        X_r = joblib.load("%s/X_reconstructed_%d.jbl" % (results_path, mode))

        n = X_r.shape[0]
        s = X_r.shape[1]

        if float(n) % q != 0:
            print("Problem!")
        m = n / q

        movie_mode = numpy.zeros((m, s - q))
        for j in range(s - q):  # j is the column in movie_mode
            snapshot_one_mode = numpy.zeros((m,))
            for k in range(ncopies):
                snapshot_one_mode = snapshot_one_mode + X_r[k * m : (k + 1) * m, j + k]
            snapshot_one_mode = snapshot_one_mode / ncopies
            movie_mode[:, j] = snapshot_one_mode

        joblib.dump(movie_mode, "%s/movie_mode_%d.jbl" % (results_path, mode))


def unwrap_extended_bwd_fwd(settings):
    print("\n****** RUNNING unwrap ******")
    results_path = settings.results_path
    l = settings.l
    q = settings.q
    ncopies = settings.ncopies

    for mode in range(l):

        print("Mode: ", mode)
        X_r = joblib.load("%s/X_reconstructed_%d.jbl" % (results_path, mode))

        n = X_r.shape[0]
        s = X_r.shape[1]

        S = s + q

        if float(n) % q != 0:
            print("Problem!")
        m = n / q

        # middle + forward
        movie_mode = numpy.zeros((m, s - ncopies + 1))
        for j in range(s - ncopies + 1):  # j is the column in movie_mode
            snapshot_one_mode = numpy.zeros((m,))
            for k in range(ncopies):
                snapshot_one_mode = snapshot_one_mode + X_r[k * m : (k + 1) * m, j + k]
            snapshot_one_mode = snapshot_one_mode / ncopies
            movie_mode[:, j] = snapshot_one_mode

        joblib.dump(
            movie_mode, "%s/movie_mode_%d_extended_fwd.jbl" % (results_path, mode)
        )

        # backward
        movie_mode_bkw = numpy.zeros((m, q - ncopies))
        for j in range(q - ncopies):
            snapshot_one_mode = numpy.zeros((m,))
            for k in range(ncopies):
                snapshot_one_mode = (
                    snapshot_one_mode + X_r[(j + k + 1) * m : (j + k + 2) * m, k]
                )
            snapshot_one_mode = snapshot_one_mode / ncopies
            movie_mode_bkw[:, j] = snapshot_one_mode
        movie_mode_bkw = numpy.flip(movie_mode_bkw, axis=1)

        movie_mode_extended = numpy.zeros((m, S - 2 * ncopies + 1))
        movie_mode_extended[:, 0 : q - ncopies] = movie_mode_bkw
        movie_mode_extended[:, q - ncopies :] = movie_mode

        joblib.dump(
            movie_mode_extended,
            "%s/movie_mode_%d_extended_fwd_bwd.jbl" % (results_path, mode),
        )


def reconstruct_unwrap_loop(settings):
    print("\n****** RUNNING reconstruct_unwrap_loop ******")
    results_path = settings.results_path
    q = settings.q
    ncopies = settings.ncopies

    U = joblib.load("%s/U_norm_rightEV.jbl" % results_path)
    S = joblib.load("%s/S_norm_rightEV.jbl" % results_path)
    VT_final = joblib.load("%s/VT_final_norm_rightEV.jbl" % results_path)

    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]

    m = U.shape[0] / q

    VT_final = VT_final.T

    for k in range(nmodes):
        print("Mode: ", k)
        x_r = numpy.zeros((m, s - q))
        for j in range(s - q):
            for i in range(ncopies):
                # print i, j
                i1 = i * m
                i2 = (i + 1) * m
                x_r[:, j] = x_r[:, j] + U[i1:i2, k] * S[k] * VT_final[j + i, k]
            x_r[:, j] = x_r[:, j] / ncopies
        joblib.dump(x_r, "%s/movie_mode_%d_optimised.jbl" % (results_path, k))
        print(x_r.shape)


def reconstruct_unwrap_loop_extended_bwd_fwd(settings):
    print("\n****** RUNNING reconstruct_unwrap_loop ******")
    results_path = settings.results_path
    q = settings.q
    ncopies = settings.ncopies

    U = joblib.load("%s/U_norm_rightEV.jbl" % results_path)
    SVs = joblib.load("%s/S_norm_rightEV.jbl" % results_path)
    VT_final = joblib.load("%s/VT_final_norm_rightEV.jbl" % results_path)

    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    S = s + q
    m = U.shape[0] / q

    VT_final = VT_final.T

    for k in range(nmodes):
        print("Mode: ", k)

        # middle + forward
        x_r = numpy.zeros((m, s - ncopies + 1))
        for j in range(s - ncopies + 1):
            for i in range(ncopies):
                # print i, j
                i1 = i * m
                i2 = (i + 1) * m
                x_r[:, j] = x_r[:, j] + U[i1:i2, k] * SVs[k] * VT_final[j + i, k]
            x_r[:, j] = x_r[:, j] / ncopies
        joblib.dump(
            x_r, "%s/movie_mode_%d_optimised_extended_fwd.jbl" % (results_path, k)
        )
        print(x_r.shape)

        # backward
        x_r_bkw = numpy.zeros((m, q - ncopies))
        for j in range(q - ncopies):
            for i in range(ncopies):
                # print i, j
                i1 = (j + 1 + i) * m
                i2 = (j + 1 + i + 1) * m
                x_r_bkw[:, j] = x_r_bkw[:, j] + U[i1:i2, k] * SVs[k] * VT_final[i, k]
            x_r_bkw[:, j] = x_r_bkw[:, j] / ncopies

        x_r_bkw = numpy.flip(x_r_bkw, axis=1)

        x_r_extended = numpy.zeros((m, S - 2 * ncopies + 1))
        x_r_extended[:, 0 : q - ncopies] = x_r_bkw
        x_r_extended[:, q - ncopies :] = x_r

        joblib.dump(
            x_r_extended,
            "%s/movie_mode_%d_optimised_extended_fwd_bwd.jbl" % (results_path, k),
        )


def reconstruct_unwrap_loop_chunck_bwd(settings):
    print("\n****** RUNNING reconstruct_unwrap_loop_chunck_bwd ******")
    results_path = settings.results_path
    q = settings.q
    ncopies = settings.ncopies
    datatype = settings.datatype

    U = joblib.load("%s/U.jbl" % results_path)
    SVs = joblib.load("%s/S.jbl" % results_path)
    VT_final = joblib.load("%s/VT_final.jbl" % results_path)

    nmodes = VT_final.shape[0]
    m = U.shape[0] / q

    VT_final = VT_final.T

    for k in range(nmodes):
        print("Mode: ", k)

        # backward
        x_r_bkw = numpy.zeros((m, q - ncopies), dtype=datatype)
        for j in range(q - ncopies):
            for i in range(ncopies):
                # print i, j
                i1 = (j + 1 + i) * m
                i2 = (j + 1 + i + 1) * m
                x_r_bkw[:, j] = x_r_bkw[:, j] + U[i1:i2, k] * SVs[k] * VT_final[i, k]
            x_r_bkw[:, j] = x_r_bkw[:, j] / ncopies

        x_r_bkw = numpy.flip(x_r_bkw, axis=1)
        print(x_r_bkw.shape)

        joblib.dump(
            x_r_bkw, "%s/movie_mode_%d_optimised_chunk_bwd.jbl" % (results_path, k)
        )


# if __name__ == '__main__':
# reconstruct()
# unwrap()
# unwrap_extended_bwd_fwd()
# reconstruct_unwrap_loop_chunck_bwd()
