# -*- coding: utf-8 -*-
import numpy
# RHO
a = 61.36
b = 91.15
c = 150.43

h = 35
k = 0
l = 0


q = numpy.sqrt((float(h) / a) ** 2 + (float(k) / b) ** 2 + (float(l) / c) ** 2)
print(q)
print(1 / q)