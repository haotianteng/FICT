#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:04:33 2020

@author: haotian teng
"""

from fict.fict_input import RealDataLoader
import numpy as np
import pandas as pd
from fict.utils import data_op as dop
from matplotlib import pyplot as plt
import os

def read_smfish_data(data_f,data_type = None):
    data_matrix = []
    with open(data_f,'r') as f:
        for line in f:
            split_line = line.strip('\n').split()
            data_matrix.append([float(x) for x in split_line[1:]])
    return np.asarray(data_matrix,dtype = data_type)

def read_smfish_gene(gene_f):
    genes = []
    with open(gene_f,'r') as f:
        for line in f:
            split_line = line.strip().split()
            genes.append(split_line[1:])
    return genes

def split_field(coordinate,x_bin = None, y_bin = None):
    """Split the samples by the given x split and(or) y split.
    c | c | c | c
    """
    sample_n = len(coordinate)
    fields = np.empty(sample_n,dtype = int)
    x = coordinate[:,0]
    y = coordinate[:,1]
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    if y_bin is None:
        y_bin = [max_y]
    if x_bin is None:
        x_bin = [max_x]
    x_bin = list(np.sort(x_bin))
    y_bin = list(np.sort(y_bin))
    field = 0
    if x_bin[0]<=min_x:
        x_bin = x_bin[1:]
    if y_bin[0]<=min_y:
        y_bin = y_bin[1:]
    if x_bin[-1]<max_x:
        x_bin.append(max_x)
    if y_bin[-1]<max_y:
        y_bin.append(max_y)
    pre_x = min_x-1
    pre_y = min_y-1
    for x_i in x_bin:
        for y_i in y_bin:
            x_mask = np.logical_and(x<=x_i,x>pre_x)
            y_mask = np.logical_and(y<=y_i,y>pre_y)
            fields[np.logical_and(x_mask,y_mask)]=field
            field += 1
            pre_y = y_i
        pre_x = x_i
        pre_y = min_y-1
    return fields

if __name__ == "__main__":
    
    data_f = "/home/heavens/twilight/data/Seqfish/"
    save_f = "/home/heavens/lanec1/data/Benchmark/seqFISH/data/"
    prefix = "fcortex"
    threshold_distance = 40
    n_c = 7
    coordinate_f = os.path.join(data_f,prefix+".coordinates")
    expression_f = os.path.join(data_f,prefix+".expression")
    gene_f = os.path.join(data_f,prefix+".genes")
    coordinate = read_smfish_data(coordinate_f,data_type = np.int)
    coordinate = coordinate[:,1:]
    expression = read_smfish_data(expression_f,data_type = np.float)
    gene_list = read_smfish_gene(gene_f)
    fields = split_field(coordinate,y_bin = [-2000])
    fig,ax = plt.subplots()
    ax.scatter(x = coordinate[:,0],y = coordinate[:,1],c = fields,cmap = 'tab20c')
    coordinate = np.concatenate((coordinate,np.zeros((len(coordinate),1))),axis=1)
    real_df = RealDataLoader(expression,
                             coordinate,
                             threshold_distance = threshold_distance,
                             cell_labels = np.zeros(len(expression)),
                             num_class = n_c,
                             field = fields,
                             for_eval = False)
    dop.save_loader(real_df,save_f+str(0))
    for i,f in enumerate(set(fields)):
        mask = fields==f
        smfish_loader = RealDataLoader(expression[mask,:],
									   coordinate[mask,:],
									   threshold_distance = threshold_distance,
	                                   gene_list = gene_list,
	                                   cell_labels = np.zeros(len(expression)),
	                                   num_class = n_c,
	                                   field = [f]*sum(mask),
	                                   for_eval = False)
        dop.save_smfish(smfish_loader,save_f+str(i+1))
        dop.save_loader(smfish_loader,save_f+str(i+1))
    