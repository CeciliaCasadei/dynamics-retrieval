# -*- coding: utf-8 -*-
import joblib
import numpy
import scipy.sparse


def f_sparse_m_T_m(settings, mask):
    "Write mask.T @ mask to n_dsq_elements.jbl"
    try:
        mask = mask[:, :].todense()
        mask = numpy.asarray(mask, dtype=settings.datatype)
        print("mask (dense): ", mask.shape, mask.dtype)
    except:
        print("mask is not sparse", mask.shape, mask.dtype)

    mask_T = mask.T
    n_dsq_elements = numpy.matmul(mask_T, mask)
    print("n_dsq_elements:", n_dsq_elements.shape, n_dsq_elements.dtype)
    joblib.dump(n_dsq_elements, "%s/n_dsq_elements.jbl" % settings.results_path)


def calc_x_sq_T_mask_term(settings, x_sq, mask):
    mask = mask[:, :].todense()
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    x_sq = x_sq[:, :].todense()
    x_sq = numpy.asarray(x_sq, dtype=settings.datatype)
    x_sq_T_mask = numpy.matmul(x_sq.T, mask)
    print("x_sq_T_mask:", x_sq_T_mask.shape, x_sq_T_mask.dtype)
    print("issparse: ", scipy.sparse.isspmatrix(x_sq_T_mask))
    print("Saving x_sq_T_mask_term")
    joblib.dump(x_sq_T_mask, "%s/term_xsqTmask.jbl" % settings.results_path)


def calc_mask_T_x_sq_term(settings, x_sq, mask):
    mask = mask[:, :].todense()
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    x_sq = x_sq[:, :].todense()
    x_sq = numpy.asarray(x_sq, dtype=settings.datatype)
    mask_T_x_sq = numpy.matmul(mask.T, x_sq)
    print("mask_T_x_sq:", mask_T_x_sq.shape, mask_T_x_sq.dtype)
    print("issparse: ", scipy.sparse.isspmatrix(mask_T_x_sq))
    print("Saving mask_T_x_sq_term")
    joblib.dump(mask_T_x_sq, "%s/term_maskTxsq.jbl" % settings.results_path)


def calc_x_T_x_term(settings, x):
    x = x[:, :].todense()
    x = numpy.asarray(x, settings.datatype)
    term_xTx = -2 * numpy.matmul(x.T, x)
    print("term_xTx:", term_xTx.shape, term_xTx.dtype)
    print("issparse: ", scipy.sparse.isspmatrix(term_xTx))
    print("Saving x_T_x_term")
    joblib.dump(term_xTx, "%s/term_xTx.jbl" % settings.results_path)


def f_sparse_x_T_x(settings, x):
    print("\n****** RUNNING f_sparse_x_T_x ******")
    m = x.shape[0]
    S = x.shape[1]
    print(m, " pixels")
    print(S, " samples")
    print("Calculate x.T x (sparse data in general)")
    print("Start\n")
    print("Sparse data. Is sparse?", scipy.sparse.isspmatrix(x), x.dtype)

    calc_x_T_x_term(settings, x)


def f_sparse_x_sq_T_mask(settings, x, mask):
    print("\n****** RUNNING f_sparse_x_sq_T_mask ******")
    m = x.shape[0]
    S = x.shape[1]
    print(m, " pixels")
    print(S, " samples")
    print("Calculate x_sq.T mask (sparse data in general)")
    print("Start\n")
    print("sparse data. Is sparse?", scipy.sparse.isspmatrix(x), x.dtype)

    x_sq = x.multiply(x)
    print("sparse x_sq. Is  sparse?", scipy.sparse.isspmatrix(x_sq), x_sq.dtype)

    calc_x_sq_T_mask_term(settings, x_sq, mask)


def f_sparse_mask_T_x_sq(settings, x, mask):
    print("\n****** RUNNING f_sparse_mask_T_x_sq ******")
    m = x.shape[0]
    S = x.shape[1]
    print(m, " pixels")
    print(S, " samples")
    print("Calculate mask.T x_sq (sparse data in general)")
    print("Start\n")
    print("sparse data. Is sparse?", scipy.sparse.isspmatrix(x), x.dtype)

    x_sq = x.multiply(x)
    print("sparse x_sq. Is  sparse?", scipy.sparse.isspmatrix(x_sq), x_sq.dtype)

    calc_mask_T_x_sq_term(settings, x_sq, mask)


def f_add_1(settings):
    print("\n****** RUNNING f_add_1 ******")
    temp = joblib.load("%s/term_xsqTmask.jbl" % settings.results_path)
    print("x_sq_T_mask:", temp.dtype, temp.shape)
    to_add = joblib.load("%s/term_maskTxsq.jbl" % settings.results_path)
    print("mask_T_x_sq:", to_add.dtype, temp.shape)
    temp += to_add
    print("Saving temp")
    joblib.dump(temp, "%s/temp.jbl" % settings.results_path)


def f_add_2(settings):
    print("\n****** RUNNING f_add_2 ******")
    temp = joblib.load("%s/temp.jbl" % settings.results_path)
    to_add = joblib.load("%s/term_xTx.jbl" % settings.results_path)
    print("-2 x_T_x:", to_add.dtype, temp.shape)
    temp += to_add
    print("Saving d_sq")
    joblib.dump(temp, "%s/d_sq_tmp.jbl" % settings.results_path)


def regularise_d_sq(settings, label=""):
    d_sq = joblib.load("%s/d_sq_tmp.jbl" % settings.results_path)
    print("d_sq: ", d_sq.shape)
    test = numpy.argwhere(d_sq < 0)
    print("\n")
    print(test.shape[0], "negative values in d_sq")
    test = numpy.argwhere(numpy.diag(d_sq) < 0)
    print(test.shape[0], "negative values in diag(d_sq)")

    print("d_sq min value: ", numpy.amin(d_sq))
    print("d_sq max value: ", numpy.amax(d_sq))

    print("\nSet diag d_sq values to zero")
    numpy.fill_diagonal(d_sq, 0)

    test = numpy.argwhere(d_sq < 0)
    print(test.shape[0], "negative values in d_sq")
    test = numpy.argwhere(numpy.diag(d_sq) < 0)
    print(test.shape[0], "negative values in diag(d_sq)")

    print("d_sq min value: ", numpy.amin(d_sq))
    print("d_sq max value: ", numpy.amax(d_sq))

    print("\nSet negative values to zero.")
    d_sq[d_sq < 0] = 0

    diff = d_sq - d_sq.T
    print("d_sq is symmetric:")
    print(numpy.amin(diff), numpy.amax(diff))
    print("d_sq calculation done.")

    joblib.dump(d_sq, "%s/d_sq%s.jbl" % (settings.results_path, label))


def f_dense(settings, x, mask):
    "Compute distances between all non-masked columns of x"
    print("\n****** RUNNING util_calculate_d_sq.f_dense (DENSE INPUT) ******")
    m = x.shape[0]
    S = x.shape[1]

    print(m, " pixels")
    print(S, " samples")

    print("Calculate d^2 (sparse data in general)")

    print("Start")

    x_sq = numpy.multiply(x, x)

    d_sq = (
        numpy.matmul(x_sq.T, mask)
        + numpy.matmul(mask.T, x_sq)  # TODO could avoid one matmul here
        - 2 * numpy.matmul(x.T, x)
    )

    return d_sq


### UNUSED

# def calc_x_sq_T_mask(x_sq, mask):
#     x_sq_T_mask = x_sq.T * mask
#     x_sq_T_mask = x_sq_T_mask[:,:].todense()
#     x_sq_T_mask = numpy.asarray(x_sq_T_mask, dtype=numpy.float32)
#     return x_sq_T_mask

# def calc_mask_T_x_sq(x_sq, mask):
#     mask_T_x_sq = mask.T * x_sq
#     mask_T_x_sq = mask_T_x_sq[:,:].todense()
#     mask_T_x_sq = numpy.asarray(mask_T_x_sq, dtype=numpy.float32)
#     return mask_T_x_sq

# def calc_x_T_x(x):
#     x_T_x = x.T * x
#     x_T_x = x_T_x[:,:].todense()
#     x_T_x = numpy.asarray(x_T_x, dtype=numpy.float32)
#     return x_T_x

# def f_sparse(x, mask):
#     print '\n****** RUNNING calculate_d_sq SPARSE INPUT ******'
#     m = x.shape[0]
#     S = x.shape[1]

#     print m, ' pixels'
#     print S, ' samples'

#     print 'Calculate d^2 (sparse data in general)'
#     print 'Start\n'

#     print 'sparse data. Is sparse?', scipy.sparse.isspmatrix(x),    x.dtype
#     print 'sparse mask. Is sparse?', scipy.sparse.isspmatrix(mask), mask.dtype

# #    d_sq = x_sq.T * mask \
# #           + mask.T * x_sq \
# #           - 2 * x.T * x
# #    print 'sparse d_sq: Is sparse?', scipy.sparse.isspmatrix(d_sq), '- Size:', d_sq.data.nbytes + d_sq.indptr.nbytes + d_sq.indices.nbytes, d_sq.dtype
# #    print 'sparse d_sq nonzero: ', d_sq.count_nonzero()
# #    d_sq = d_sq[:,:].todense()
# #    print 'dense d_sq. Is sparse?', scipy.sparse.isspmatrix(d_sq), '- Size: ', d_sq.nbytes, d_sq.dtype


#     d_sq = -2 * calc_x_T_x(x)

#     x_sq = x.multiply(x)
#     print 'sparse x_sq. Is  sparse?', scipy.sparse.isspmatrix(x_sq), x_sq.dtype

#     d_sq += calc_x_sq_T_mask(x_sq, mask) # Test +=
#     d_sq += calc_mask_T_x_sq(x_sq, mask) # Test +=

#     print 'dense d_sq. Is sparse?', scipy.sparse.isspmatrix(d_sq), '- Size: ', d_sq.nbytes, d_sq.dtype

# #    diff = abs(d_sq - d_sq_test)
# #    print 'Diff: ', numpy.amax(diff)

#     print 'd_sq: ', d_sq.shape
#     test = numpy.argwhere(d_sq < 0)
#     print '\n'
#     print test.shape[0], 'negative values in d_sq'
#     test = numpy.argwhere(numpy.diag(d_sq) < 0)
#     print test.shape[0], 'negative values in diag(d_sq)'

#     print 'd_sq min value: ', numpy.amin(d_sq)
#     print 'd_sq max value: ', numpy.amax(d_sq)

#     print '\nSet diag d_sq values to zero'
#     numpy.fill_diagonal(d_sq, 0)

# #    test = numpy.argwhere(d_sq < 0)
# #    print test.shape[0], 'negative values in d_sq'
# #    test = numpy.argwhere(numpy.diag(d_sq) < 0)
# #    print test.shape[0], 'negative values in diag(d_sq)'
# #
# #    print 'd_sq min value: ', numpy.amin(d_sq)
# #    print 'd_sq max value: ', numpy.amax(d_sq)

#     print '\nSet negative values to zero.'
#     d_sq[d_sq < 0] = 0

# #    diff = d_sq - d_sq.T
# #    print 'd_sq is symmetric:'
# #    print numpy.amin(diff), numpy.amax(diff)
# #    print 'd_sq calculation done.'

#     return d_sq
