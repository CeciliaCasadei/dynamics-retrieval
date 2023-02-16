#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:54:08 2023

@author: casadei_c
"""
import matplotlib.pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

def f(x, fn, title=''):
    cmap = matplotlib.cm.jet
    cmap.set_bad('white')
    im = matplotlib.pyplot.imshow(x, cmap=cmap)   
    ax = matplotlib.pyplot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)   
    cb = matplotlib.pyplot.colorbar(im, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.update_ticks()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(fn, dpi=96*3)
    matplotlib.pyplot.close() 
    
    
    