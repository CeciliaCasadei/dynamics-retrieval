# -*- coding: utf-8 -*-
import time

import numpy


def f(d_sq, q, datatype):
    print("\n****** RUNNING calculate_D_sq ******")

    S = d_sq.shape[0]
    s = S - q

    print(S, " samples")

    D_sq = numpy.zeros((s + 1, s + 1), dtype=datatype)  # will need to be s,s
    starttime = time.time()

    for i in range(0, q):
        if i % 10 == 0:
            print(i, "/", q)
        term = d_sq[i : i + s + 1, i : i + s + 1]
        D_sq = D_sq + term

    print("Time: ", time.time() - starttime)

    return D_sq
