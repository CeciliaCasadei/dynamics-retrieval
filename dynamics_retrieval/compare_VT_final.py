# -*- coding: utf-8 -*-
import pickle

import numpy
import settings

results_path = settings.results_path
print(results_path)

f = open("%s/VT_final_loop.pkl" % results_path, "rb")
VT_final_loop = pickle.load(f)
f.close()

f = open("%s/VT_final.pkl" % results_path, "rb")
VT_final = pickle.load(f)
f.close()

f = open("%s/VT_final_manual.pkl" % results_path, "rb")
VT_final_manual = pickle.load(f)
f.close()

diff = abs(VT_final) - abs(VT_final_loop)
print(numpy.amax(diff), numpy.amin(diff))

diff = abs(VT_final) - abs(VT_final_manual)
print(numpy.amax(diff), numpy.amin(diff))

diff = abs(VT_final_loop) - abs(VT_final_manual)
print(numpy.amax(diff), numpy.amin(diff))
