# -*- coding: utf-8 -*-
# import scipy.io
import h5py
import joblib
import numpy


def get_meanden_simple():

    sigmacutoffs = [0]
    here = "."

    fn = "%s/results/mapd0.mat" % (here)

    print("Loading")

    f = h5py.File(fn, "r")
    mapd0 = numpy.asarray(f["/mapd0"], dtype=numpy.float32)
    print("mapd0 :", mapd0.shape, mapd0.dtype)

    for sigmacutoff in sigmacutoffs:
        print("Sigma cutoff: ", sigmacutoff)
        meanposden = numpy.mean(numpy.where(mapd0 < sigmacutoff, 0, mapd0), axis=0)
        meannegden = numpy.mean(numpy.where(mapd0 > -sigmacutoff, 0, mapd0), axis=0)
        joblib.dump(
            meanposden, "%s/results/meanposden_sigcutoff_%.1f.jbl" % (here, sigmacutoff)
        )
        joblib.dump(
            meannegden, "%s/results/meannegden_sigcutoff_%.1f.jbl" % (here, sigmacutoff)
        )
        print("meanposden: ", meanposden.shape, meanposden.dtype)
        print("meannegden: ", meannegden.shape, meannegden.dtype)


def get_meanden(mode):

    radius = 1.7
    distance = 0.2
    sigmacutoffs = [2.5, 3.0, 3.5, 4.0, 4.5]
    label = "_step_10_range_105000_145000"
    l = "_125010_145000"

    here = "."
    fn = "%s/results_m_0_%d%s/mapd0%s.mat" % (here, mode, label, l)

    print("Loading")

    f = h5py.File(fn, "r")
    mapd0 = numpy.asarray(f["/mapd0"], dtype=numpy.float32)
    print("mapd0 :", mapd0.shape, mapd0.dtype)

    for sigmacutoff in sigmacutoffs:
        print("Sigma cutoff: ", sigmacutoff)
        meanposden = numpy.mean(numpy.where(mapd0 < sigmacutoff, 0, mapd0), axis=0)
        meannegden = numpy.mean(numpy.where(mapd0 > -sigmacutoff, 0, mapd0), axis=0)
        joblib.dump(
            meanposden,
            "%s/results_m_0_%d%s/meanposden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l, radius, distance, sigmacutoff),
        )
        joblib.dump(
            meannegden,
            "%s/results_m_0_%d%s/meannegden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l, radius, distance, sigmacutoff),
        )
        print("meanposden: ", meanposden.shape, meanposden.dtype)
        print("meannegden: ", meannegden.shape, meannegden.dtype)


def merge():
    mode = 1
    radius = 1.7
    distance = 0.2
    sigmacutoffs = [2.5, 3.0, 3.5, 4.0, 4.5]

    label = "_step_10_range_50000_99990"
    l_1 = "_50000_74990"
    l_2 = "_75000_99990"
    here = "."

    for sigmacutoff in sigmacutoffs:
        meanposden_1 = joblib.load(
            "%s/results_m_0_%d%s/meanposden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l_1, radius, distance, sigmacutoff)
        )
        meanposden_2 = joblib.load(
            "%s/results_m_0_%d%s/meanposden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l_2, radius, distance, sigmacutoff)
        )

        meannegden_1 = joblib.load(
            "%s/results_m_0_%d%s/meannegden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l_1, radius, distance, sigmacutoff)
        )
        meannegden_2 = joblib.load(
            "%s/results_m_0_%d%s/meannegden_m_0_%d%s_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, l_2, radius, distance, sigmacutoff)
        )

        n_atoms = meanposden_1.shape[0]
        n_ts_1 = meanposden_1.shape[1]
        n_ts_2 = meanposden_2.shape[1]

        print(meanposden_1.shape, meanposden_2.shape, meannegden_1.shape, meannegden_2.shape)
        meanposden = numpy.zeros((n_atoms, n_ts_1 + n_ts_2))
        meannegden = numpy.zeros((n_atoms, n_ts_1 + n_ts_2))
        print(meanposden.shape, meannegden.shape)
        meanposden[:, 0:n_ts_1] = meanposden_1
        meanposden[:, n_ts_1 : n_ts_1 + n_ts_2] = meanposden_2
        meannegden[:, 0:n_ts_1] = meannegden_1
        meannegden[:, n_ts_1 : n_ts_1 + n_ts_2] = meannegden_2

        joblib.dump(
            meanposden,
            "%s/results_m_0_%d%s/meanposden_m_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, radius, distance, sigmacutoff),
        )
        joblib.dump(
            meannegden,
            "%s/results_m_0_%d%s/meannegden_m_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl"
            % (here, mode, label, mode, radius, distance, sigmacutoff),
        )


if __name__ == "__main__":
    # ms = [1]
    # for m in ms:
    #     get_meanden(m)

    # merge()

    get_meanden_simple()
