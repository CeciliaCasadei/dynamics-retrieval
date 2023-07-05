# -*- coding: utf-8 -*-
import joblib
import matplotlib
import numpy

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot

def f(toplot, fn):
    matplotlib.pyplot.figure(figsize=(20,10))
    matplotlib.pyplot.plot(list(range(toplot.shape[0])), toplot, c='b', s=1)        
    ax = matplotlib.pyplot.gca()
    ax.tick_params(axis='both', which='major', labelsize=38)
    ax.tick_params(axis='both', which='minor', labelsize=38)
    matplotlib.pyplot.savefig(fn)
    matplotlib.pyplot.close()
        
def plot_d_0j(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    d_sq = joblib.load("%s/d_sq.jbl" % (results_path))
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print(numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))

    numpy.fill_diagonal(d_sq, 0)
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print(numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))
    d = numpy.sqrt(d_sq)

    for idx in idxs:
    
        d_0j = d[idx,:]
        print('d_0j: ', d_0j.shape, d_0j.dtype)   
        fn = '%s/dist_%d_j.png'%(results_path, idx)
        f(d_0j, fn)
        


def plot_D_0j(settings):
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    d_sq = joblib.load("%s/D_sq_normalised.jbl" % (results_path))
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print("Diag:", numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))

    numpy.fill_diagonal(d_sq, 0)
    print(numpy.amax(d_sq), numpy.amin(d_sq))
    print('Diag:', numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq)))
    d = numpy.sqrt(d_sq)

    for idx in idxs:
    
        d_0j = d[idx,:]
        print('d_0j: ', d_0j.shape, d_0j.dtype) 
        fn = '%s/D_%d_j.png'%(results_path, idx)
        f(d_0j, fn)
        