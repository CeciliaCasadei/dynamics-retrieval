# -*- coding: utf-8 -*-
import numpy
import joblib
import scipy
#import pickle

def calculate_d_sq_SFX_steps(settings):    
    import util_calculate_d_sq
    
    data_path = settings.data_path
    data_file = settings.data_file
    label = settings.label
       
    # EXTRACT DATA
    T_sparse = joblib.load(data_file)
    
    print 'Data file: ', data_file
    print 'T_sparse: ', T_sparse.shape, T_sparse.dtype
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    
    print 'M_sparse: ', M_sparse.shape, M_sparse.dtype
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
        
    # CALCULATE TERM BY TERM    
    util_calculate_d_sq.f_sparse_x_T_x(settings, T_sparse)
    util_calculate_d_sq.f_sparse_x_sq_T_mask(settings, T_sparse, M_sparse)
    util_calculate_d_sq.f_sparse_mask_T_x_sq(settings, T_sparse, M_sparse)
    util_calculate_d_sq.f_add_1(settings)
    util_calculate_d_sq.f_add_2(settings)
    util_calculate_d_sq.regularise_d_sq(settings)
    


def calculate_d_sq_SFX(settings):    
    import util_calculate_d_sq
    
    datatype = settings.datatype    
    data_path = settings.data_path
    data_file = settings.data_file
    label = settings.label
       
    # EXTRACT DATA AND CONVERT SPARSE -> DENSE
    T_sparse = joblib.load(data_file)
    
    print 'Data file: ', data_file
    print 'T_sparse: is  sparse?', scipy.sparse.isspmatrix(T_sparse)  
    print 'T_sparse: ', T_sparse.shape, T_sparse.dtype
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    print 'Converting sparse to dense.'
    
    x = T_sparse[:,:].todense()
    print 'x: is  sparse?', scipy.sparse.isspmatrix(x)  
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    
    print 'M_sparse: is  sparse?', scipy.sparse.isspmatrix(M_sparse)  
    print 'M_sparse: ', M_sparse.shape, M_sparse.dtype
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    print 'Converting sparse to dense.'
    
    mask = M_sparse[:,:].todense()
    print 'mask: is  sparse?', scipy.sparse.isspmatrix(mask)  
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    # CALCULATE d_sq DENSE INPUT  
    util_calculate_d_sq.f_dense(settings, x, mask)
    util_calculate_d_sq.regularise_d_sq(settings, '_dense_input')
    

def compare(settings):
    d_sq = joblib.load('%s/d_sq.jbl'%settings.results_path)   
    d_sq_dense_input = joblib.load('%s/d_sq_dense_input.jbl'%settings.results_path)    
    diff = abs(d_sq - d_sq_dense_input)
    print 'max(diff): ', numpy.amax(diff)
    diff_rel = diff/d_sq
    print 'max(diff_rel): ', numpy.nanmax(diff_rel)
    
def merge_D_sq(settings):
    import util_merge_D_sq
    util_merge_D_sq.f(settings)    
    
def sort_D_sq(settings):
    import util_sort_D_sq
    
    results_path = settings.results_path
    b = settings.b
    datatype = settings.datatype
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    D, N = util_sort_D_sq.f_opt(D_sq, b, datatype)
    
    print 'Saving'    
    joblib.dump(D, '%s/D_loop.jbl'%results_path)
    joblib.dump(N, '%s/N_loop.jbl'%results_path)
    #joblib.dump(v, '%s/v.jbl'%results_path)
    print '\n'
    
# #    # CALCULATE d_sq SPARSE INPUT
# #    d_sq_sparse = util_calculate_d_sq.f_sparse(T_sparse, M_sparse)
# #    print 'd_sq_sparse: ', d_sq_sparse.shape, d_sq_sparse.dtype
# #        
# #    print 'Saving d_sq_sparse'    
# #    joblib.dump(d_sq_sparse, '%s/d_sq_sparseinput.jbl'%results_path)
# #    print '\n' 
        

# #    term_xTx = util_calculate_d_sq.f_sparse_x_T_x(T_sparse)
# #    print 'Saving x_T_x'    
# #    joblib.dump(term_xTx, '%s/term_xTx.jbl'%results_path)
    
    




### UNUSED

# def concatenate_backward():
#     import util_concatenate_backward
#     util_concatenate_backward.f()
    
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
    
# def calculate_d_sq_dense():
#     import settings_NP as settings
#     import util_calculate_d_sq
    
#     results_path = settings.results_path
#     f = open('%s/T_anomaly.pkl'%results_path, 'rb')
#     #f = open('%s/T.pkl'%results_path, 'rb')
#     x = pickle.load(f)
#     f.close()
#     print 'Data: ', x.shape
    
#     mask = numpy.ones(x.shape)       
    
#     d_sq = util_calculate_d_sq.f(x, mask)
#     print 'd_sq: ', d_sq.shape, d_sq.dtype
    
#     print 'Saving d_sq'    
#     joblib.dump(d_sq, '%s/d_sq.jbl'%results_path)
#     print '\n'
    
# def calculate_d_sq_sparse():
#     import settings
#     import util_calculate_d_sq
    
#     results_path = settings.results_path
#     sparsity = settings.sparsity
#     f = open('%s/T_anomaly_sparse_%.2f.pkl'%(results_path, sparsity), 'rb')
#     x = pickle.load(f)
#     f.close()
#     print 'Data: ', x.shape
    
#     mask = numpy.ones(x.shape) 
#     mask[numpy.isnan(x)] = 0
#     x[numpy.isnan(x)] = 0      
    
#     d_sq = util_calculate_d_sq.f(x, mask)
#     print 'd_sq: ', d_sq.shape, d_sq.dtype
    
#     print 'Saving d_sq'    
#     joblib.dump(d_sq, '%s/d_sq.jbl'%results_path)
#     print '\n'  
        
# def calculate_D_sq():  
#     import settings_NP as settings
#     import util_calculate_D_sq
    
#     results_path = settings.results_path
#     q = settings.q
#     datatype = settings.datatype
#     d_sq = joblib.load('%s/d_sq.jbl'%results_path)    
#     print 'd_sq: ', d_sq.shape
    
#     D_sq = util_calculate_D_sq.f(d_sq, q, datatype)
#     print 'D_sq: ', D_sq.shape, D_sq.dtype
    
#     print 'Saving D_sq'    
#     joblib.dump(D_sq, '%s/D_sq.jbl'%results_path)
#     print '\n'