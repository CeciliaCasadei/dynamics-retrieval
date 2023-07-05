#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:01:13 2023

@author: casadei_c
"""


def main(sett, fn, q, fmax, path, datafile, nmodes):
    fopen = open(fn, "w")
    fopen.write("# -*- coding: utf-8 -*-\n")
    fopen.write("import numpy\n")
    fopen.write("import math\n")
    fopen.write("S = %d\n" % sett.S)
    fopen.write("m = %d\n" % sett.m)
    fopen.write("q = %d\n" % q)
    fopen.write("f_max = %d\n" % fmax)
    fopen.write("f_max_considered = f_max\n")
    fopen.write('results_path = "%s"\n' % path)
    # fopen.write('paral_step_A = 1000\n')
    fopen.write("datatype = numpy.float64\n")
    fopen.write('data_file = "%s"\n' % datafile)
    # fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
    fopen.write("p = 0\n")
    fopen.write("modes_to_reconstruct = range(0, %d)\n" % nmodes)
    fopen.close()
