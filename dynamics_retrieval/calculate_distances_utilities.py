# -*- coding: utf-8 -*-
import joblib
import numpy
import scipy

# import pickle


def calculate_d_sq_SFX_steps(settings):
    from . import util_calculate_d_sq

    data_path = settings.data_path
    data_file = settings.data_file
    label = settings.label

    # EXTRACT DATA
    T_sparse = joblib.load(data_file)

    print("Data file: ", data_file)
    print("T_sparse: ", T_sparse.shape, T_sparse.dtype)
    print("T_sparse nonzero: ", T_sparse.count_nonzero())

    M_sparse = joblib.load("%s/M_sparse_%s.jbl" % (data_path, label))

    print("M_sparse: ", M_sparse.shape, M_sparse.dtype)
    print("M_sparse nonzero: ", M_sparse.count_nonzero())

    # CALCULATE TERM BY TERM
    util_calculate_d_sq.f_sparse_x_T_x(settings, T_sparse)
    util_calculate_d_sq.f_sparse_x_sq_T_mask(settings, T_sparse, M_sparse)
    util_calculate_d_sq.f_sparse_mask_T_x_sq(settings, T_sparse, M_sparse)
    util_calculate_d_sq.f_add_1(settings)
    util_calculate_d_sq.f_add_2(settings)
    util_calculate_d_sq.regularise_d_sq(settings)


def calculate_d_sq_SFX_element_n(settings):
    from . import util_calculate_d_sq

    # data_path = settings.data_path
    # label = settings.label
    # M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    M_sparse = joblib.load("%s/mask.jbl" % (settings.results_path))

    print("M_sparse: ", M_sparse.shape, M_sparse.dtype)

    try:
        print("M_sparse nonzero: ", M_sparse.count_nonzero())
    except:
        print("M is not sparse")

    util_calculate_d_sq.f_sparse_m_T_m(settings, M_sparse)


def calculate_d_sq_SFX(settings):
    from . import util_calculate_d_sq

    datatype = settings.datatype
    data_path = settings.data_path
    data_file = settings.data_file
    label = settings.label

    # EXTRACT DATA AND CONVERT SPARSE -> DENSE
    T_sparse = joblib.load(data_file)

    print("Data file: ", data_file)
    print("T_sparse: is  sparse?", scipy.sparse.isspmatrix(T_sparse))
    print("T_sparse: ", T_sparse.shape, T_sparse.dtype)
    print("T_sparse nonzero: ", T_sparse.count_nonzero())
    print("Converting sparse to dense.")

    x = T_sparse[:, :].todense()
    print("x: is  sparse?", scipy.sparse.isspmatrix(x))
    print("x: ", x.shape, x.dtype)
    x = numpy.asarray(x, dtype=datatype)
    print("x: ", x.shape, x.dtype)

    M_sparse = joblib.load("%s/M_sparse_%s.jbl" % (data_path, label))

    print("M_sparse: is  sparse?", scipy.sparse.isspmatrix(M_sparse))
    print("M_sparse: ", M_sparse.shape, M_sparse.dtype)
    print("M_sparse nonzero: ", M_sparse.count_nonzero())
    print("Converting sparse to dense.")

    mask = M_sparse[:, :].todense()
    print("mask: is  sparse?", scipy.sparse.isspmatrix(mask))
    print("mask: ", mask.shape, mask.dtype)
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print("mask: ", mask.shape, mask.dtype)

    # CALCULATE d_sq DENSE INPUT
    util_calculate_d_sq.f_dense(settings, x, mask)
    util_calculate_d_sq.regularise_d_sq(settings, "_dense_input")


def compare(settings):
    d_sq = joblib.load("%s/d_sq.jbl" % settings.results_path)
    d_sq_dense_input = joblib.load("%s/d_sq_dense_input.jbl" % settings.results_path)
    diff = abs(d_sq - d_sq_dense_input)
    print("max(diff): ", numpy.amax(diff))
    diff_rel = diff / d_sq
    print("max(diff_rel): ", numpy.nanmax(diff_rel))


def merge_D_sq(settings):
    from . import util_merge_D_sq

    util_merge_D_sq.f(settings)


def merge_N_D_sq_elements(settings):
    from . import util_merge_D_sq

    util_merge_D_sq.f_N_D_sq_elements(settings)


def sort_D_sq(settings):
    from . import util_sort_D_sq

    results_path = settings.results_path

    D_sq = joblib.load("%s/D_sq_normalised.jbl" % results_path)
    D, N = util_sort_D_sq.f_opt(D_sq, settings)

    print("Saving")
    joblib.dump(D, "%s/D.jbl" % results_path)
    joblib.dump(N, "%s/N.jbl" % results_path)
    # joblib.dump(v, '%s/v.jbl'%results_path)
    print("\n")


def normalise(settings):
    results_path = settings.results_path

    D_sq = joblib.load("%s/D_sq_parallel.jbl" % results_path)
    print("D_sq: ", D_sq.shape, D_sq.dtype)  # S-q+1 = s+1

    N_D_sq = joblib.load("%s/N_D_sq_parallel.jbl" % results_path)
    print("N_D_sq: ", N_D_sq.shape, N_D_sq.dtype)  # S-q+1 = s+1

    D_sq = D_sq / N_D_sq
    print("After normalisation D_sq:", D_sq.shape, D_sq.dtype)
    joblib.dump(D_sq, "%s/D_sq_normalised.jbl" % results_path)


### UNUSED


def concatenate_backward(settings):
    from . import util_concatenate_backward

    util_concatenate_backward.f(settings)


# def calculate_D_N_v_optimised_dense():
#     import util_calculate_D_N_v_optimised_dense
#     util_calculate_D_N_v_optimised_dense.f()

# def calculate_D_N_v_optimised_sparse():
#     import util_calculate_D_N_v_optimised_sparse
#     util_calculate_D_N_v_optimised_sparse.f()

# def calculate_D_N_v_dense():
#     import util_calculate_D_N_v_dense
#     util_calculate_D_N_v_dense.f()

# def calculate_D_N_v_sparse():
#     import util_calculate_D_N_v_sparse
#     util_calculate_D_N_v_sparse.f()


def calculate_d_sq_dense(settings):

    from . import util_calculate_d_sq

    results_path = settings.results_path

    x = joblib.load("%s/x.jbl" % results_path)
    print("Data: ", x.shape)

    mask = numpy.ones(x.shape)

    d_sq = util_calculate_d_sq.f_dense(settings, x, mask)
    print("d_sq: ", d_sq.shape, d_sq.dtype)

    print("Saving d_sq")
    joblib.dump(d_sq, "%s/d_sq.jbl" % results_path)
    print("\n")


def calculate_d_sq_sparse(settings):

    from . import util_calculate_d_sq

    results_path = settings.results_path

    x = joblib.load("%s/x.jbl" % results_path)
    print("Data: ", x.shape)

    mask = joblib.load("%s/mask.jbl" % results_path)
    print("Mask: ", mask.shape)

    d_sq = util_calculate_d_sq.f_dense(settings, x, mask)
    print("d_sq: ", d_sq.shape, d_sq.dtype)

    print("Saving d_sq")
    joblib.dump(d_sq, "%s/d_sq.jbl" % results_path)
    print("\n")


def calculate_D_sq(settings):

    from . import util_calculate_D_sq2

    results_path = settings.results_path
    q = settings.q
    datatype = settings.datatype
    d_sq = joblib.load("%s/d_sq.jbl" % results_path)
    print("d_sq: ", d_sq.shape)

    D_sq = util_calculate_D_sq2.f(d_sq, q, datatype)
    print("D_sq: ", D_sq.shape, D_sq.dtype)

    print("Saving D_sq")
    joblib.dump(D_sq, "%s/D_sq.jbl" % results_path)
    print("\n")
