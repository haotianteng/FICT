#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 01:01:56 2020

@author: haotian teng
"""

from fict.fict_input import RealDataLoader
from fict.utils import data_op as dop
import h5py
import numpy as np
from matplotlib import pyplot as plt
file1 = "/home/heavens/twilight/data/osmFISH/osmFISH_SScortex_mouse_all_cells.loom"
f = h5py.File(file1,mode = 'r')
gene_expression =  np.asarray(f['matrix'])
gene_expression = np.transpose(gene_expression)
gene_sum = np.sum(gene_expression,axis = 1,keepdims = True)
zero_mask = gene_sum != 0
zero_mask = np.reshape(zero_mask,len(zero_mask))
gene_expression = gene_expression[zero_mask,:]
gene_sum = gene_sum[zero_mask,:]
meta = f['col_attrs']
genes = np.asarray(f['row_attrs']['Gene'])
genes = [x.decode() for x in genes]
x = np.asarray(meta['X'])[zero_mask]
y = np.asarray(meta['Y'])[zero_mask]
coordinates = np.stack((x,y),axis=1)
plt.scatter(x,y,s = 1)
cell_types = np.asarray(meta['ClusterID'])[zero_mask]
n_c = len(set(cell_types))
save_f = "/home/heavens/twilight/data/osmFISH/df_"
smfishHmrf_f = "/home/heavens/CMU/FISH_Clustering/Benchmark/osmFISH/data/"
### Data preprocessing
type_tags = np.unique(cell_types)
gene_expression = gene_expression/gene_sum
bregma = np.zeros(len(x))
split_n = 4
threshold_distance = 50
boarder = np.arange(min(x),max(x)+1,(max(x)-min(x))/4)
for i in np.arange(split_n):
    bregma[np.logical_and(x>boarder[i],x<boarder[i+1])] = i
for i in np.arange(split_n-1):
    plt.axvline(x=boarder[i+1])
real_df = RealDataLoader(gene_expression,
                         coordinates,
                         threshold_distance = threshold_distance,
                         cell_labels = cell_types,
                         num_class = n_c,
                         field = bregma,
                         for_eval = False)
dop.save_loader(real_df,save_f+'0')
for i in np.arange(split_n):
    mask = bregma == i
    loader = RealDataLoader(gene_expression[mask,:],
                            coordinates[mask,:],
                            threshold_distance = threshold_distance,
                            cell_labels = cell_types[mask],
                            num_class = n_c,
                            field = bregma[mask],
                            gene_list = genes,
                            for_eval = False)
    dop.save_loader(loader,smfishHmrf_f+str(i+1))
    dop.save_smfish(loader,smfishHmrf_f+str(i+1))