# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot
import os
import numpy

def main(settings):

    results_path = settings.results_path
        
    S = joblib.load('%s/S.jbl'%results_path)
    print S.shape
    if S.shape[0] >= 20:
        nmodes = 20
    else:
        nmodes = S.shape[0]
    print 'nmodes: ', nmodes
    for i in range(nmodes):
        print 'mode', i, 'S', S[i]
    
    out_folder = '%s/chronos'%results_path
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xticks(range(1,nmodes+1,2))   
    matplotlib.pyplot.scatter(range(1,nmodes+1), S[0:nmodes], s=5, c='b')
    matplotlib.pyplot.savefig('%s/SVs.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()
    
    s0 = S[0]
    S_norm = S/s0
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xticks(range(1,nmodes+1,2))   
    matplotlib.pyplot.scatter(range(1,nmodes+1), numpy.log10(S_norm[0:nmodes]), s=5, c='b')
    matplotlib.pyplot.savefig('%s/SVs_norm_log.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.xticks(range(1,nmodes+1,2))   
    matplotlib.pyplot.scatter(range(1,nmodes+1), S_norm[0:nmodes], s=5, c='b')
    matplotlib.pyplot.ylim([0, 1])
    matplotlib.pyplot.savefig('%s/SVs_norm.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()
    
    matplotlib.pyplot.figure()
    #matplotlib.pyplot.xticks(range(1,nmodes+1,2))   
    matplotlib.pyplot.scatter(range(1,S.shape[0]+1), S_norm[0:S.shape[0]], s=5, c='b')
    matplotlib.pyplot.ylim([0, 1])
    matplotlib.pyplot.savefig('%s/SVs_norm_ext.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()