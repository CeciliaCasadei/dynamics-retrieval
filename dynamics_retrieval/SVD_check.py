# -*- coding: utf-8 -*-
import pickle

import numpy
import numpy.linalg
import settings

results_path = settings.results_path


def calculate_A_n(U, S, VH, n):
    A_n = numpy.zeros((U.shape[0], VH.shape[0]))
    for i in range(n + 1):
        temp = numpy.outer(U[:, i], VH[i, :])
        temp = S[i] * temp
        A_n = A_n + temp
    print(n + 1, "terms")
    return A_n


f = open("%s/U.pkl" % results_path, "rb")
U = pickle.load(f)
f.close()

f = open("%s/S.pkl" % results_path, "rb")
S = pickle.load(f)
f.close()

f = open("%s/VH.pkl" % results_path, "rb")
VH = pickle.load(f)
f.close()

print("U: ", U.shape)
print("S: ", S.shape)
print("VH:", VH.shape)

f = open("%s/A.pkl" % results_path, "rb")
A = pickle.load(f)
f.close()

print("A: ", A.shape)
l = A.shape[1]

for n in range(l):
    A_n = calculate_A_n(U, S, VH, n)
    diff = A - A_n
    print(numpy.average(abs(diff)))
