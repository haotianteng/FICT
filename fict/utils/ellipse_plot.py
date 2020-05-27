#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 05:59:49 2020

@author: haotian teng
"""
from matplotlib import cm
import numpy as np
def centroid_ellipse(data,y,m,axs):
    tags = list(set(y))
    colors = cm.get_cmap('Set2', len(tags))
    y_color = [tags.index(x) for x in y]
    axs.scatter(gene_batch[:,0],gene_batch[:,1],c = y_color,cmap = colors,s = 1)
    for c in np.arange(len(tags)):
        value,vectors = scipy.linalg.eig(m.p['g_cov'][c][0:2,0:2])
        print(value)
        print(np.rad2deg(np.arctan(vectors[1][1]/vectors[1][0])))
        print(vectors)
        ellipse = matplotlib.patches.Ellipse((1,2),
                                   width = np.sqrt(value[0]),
                                   height = np.sqrt(value[1]),
                                   angle = np.rad2deg(np.arctan(vectors[1][1]/vectors[1][0])),
                                   facecolor = 'none',
                                   edgecolor = colors(c))
        (m.p['g_mean'][c,0],m.p['g_mean'][c,1])
        ellipse.set_transform(axs.transData)
        axs.add_patch(ellipse)
        axs.autoscale_view()
        return ellipse