#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:41:08 2022

@author: casadei_c
"""
import numpy


def Correlate(x1, x2):
    x1Avg = numpy.average(x1)
    x2Avg = numpy.average(x2)
    numTerm = numpy.multiply(x1 - x1Avg, x2 - x2Avg)
    num = numTerm.sum()
    resX1Sq = numpy.multiply(x1 - x1Avg, x1 - x1Avg)
    resX2Sq = numpy.multiply(x2 - x2Avg, x2 - x2Avg)
    den = numpy.sqrt(numpy.multiply(resX1Sq.sum(), resX2Sq.sum()))
    CC = num / den
    return CC
