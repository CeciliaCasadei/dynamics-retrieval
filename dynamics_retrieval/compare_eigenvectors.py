# -*- coding: utf-8 -*-
import pickle

import numpy
import settings

results_path = settings.results_path
print(results_path)
l = settings.l

f = open("%s/P_sym_evecs_ARPACK_sorted.pkl" % results_path, "rb")
evecs_ARPACK = pickle.load(f)
f.close()
evecs_ARPACK = evecs_ARPACK[:, 0:l]

f = open("%s/P_sym_evecs_sorted.pkl" % results_path, "rb")
evecs = pickle.load(f)
f.close()
evecs = evecs[:, 0:l]

# f = open('%s/VT_final_manual.pkl'%results_path, 'rb')
# VT_final_manual = pickle.load(f)
# f.close()

diff = abs(evecs) - abs(evecs_ARPACK)
print(numpy.amax(diff), numpy.amin(diff))

# diff = abs(VT_final) - abs(VT_final_manual)
# print numpy.amax(diff), numpy.amin(diff)
#
# diff = abs(VT_final_loop) - abs(VT_final_manual)
# print numpy.amax(diff), numpy.amin(diff)
