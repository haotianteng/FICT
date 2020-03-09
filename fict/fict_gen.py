#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:39:09 2020

@author: heavens
"""
import numpy as np
import pandas as pd
from fict.utils.opt import valid_neighbourhood_frequency
from fict.utils.joint_simulator import get_gene_prior,get_nf_prior,Simulator
from fict.utils.joint_simulator import SimDataLoader
import pickle
### Modulize the main function in joins_simulator.py here.

def get_prior_from_data(data_f,
                        gene_col,
                        coor_col,
                        header,
                        n_cell_type,
                        label_col_name = 'Cell_class'):
    data = pd.read_excel(data_f,header = header)
    gene_expression = data.iloc[:,gene_col]
    cell_types = data[label_col_name]
    type_tags = np.unique(cell_types)
    coordinates = data.iloc[:,coor_col]
    ### Choose only the n_c type cells
    if len(type_tags)<n_c:
        raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
    mask = np.asarray([False]*len(cell_types))
    for tag in type_tags[:n_c]:
        mask = np.logical_or(mask,cell_types==tag)
    gene_expression = gene_expression[mask]
    cell_types = np.asarray(cell_types[mask])
    coordinates = np.asarray(coordinates[mask])
    ### Generate prior from the given dataset.
    gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
    neighbour_freq_prior,tags,type_count = get_nf_prior(coordinates,cell_types)
    type_prior = type_count/np.sum(type_count)
    target_freq = (neighbour_freq_prior+0.1)/np.sum(neighbour_freq_prior+0.1,axis=1,keepdims=True)
    result = valid_neighbourhood_frequency(target_freq)
    target_freq = result[0]
    return gene_mean,gene_std,target_freq,type_prior
    
def gen_simulation(sample_n,
                   gene_n,
                   cell_type_n,
                   density = 20,
                   threshold_distance = 1,
                   target_gene_mean = None,
                   target_gene_std = None,
                   target_neighbourhood_frequency = None,
                   assign_cell_type_method = 'assign-neighbour'):
    sim = Simulator(sample_n,n_g,n_c,density)
    sim.gen_parameters(gene_mean_prior = gene_mean[:,:n_g])
    sim.gen_coordinate(density = density)
    sim.assign_cell_type(target_neighbourhood_frequency=target_freq, method = "assign-neighbour")
    return sim


if __name__=="__main__":
    ### Hyper parameter setting
    sample_n = 1000 #Number of samples
    n_g = 100 #Number of genes
    n_c = 10 #Number of cell type
    density = 20 #The average number of neighbour for each cells.
    threshold_distance = 1 # The threshold distance of neighbourhood.
    gene_col = np.arange(9,164)
    coor_col = [5,6]
    header = 1
    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/aau5324_Moffitt_Table-S7.xlsx"
    gene_mean,gene_std,target_freq,type_prior = get_prior_from_data(data_f,
                                                                    gene_col,
                                                                    coor_col,
                                                                    header,
                                                                    n_c)
    ### Generate simulation dataset and load
    sim = gen_simulation(sample_n,
                         n_g,
                         n_c,
                         target_gene_mean=gene_mean,
                         target_gene_std=gene_std,
                         target_neighbourhood_frequency=target_freq)
    gene_expression,cell_type,cell_neighbour = sim.gen_expression()
    df = SimDataLoader(gene_expression,
                       cell_neighbour,
                       cell_type_assignment = cell_type)
    batch_size = 100
    for i in range(10):
        x_batch, y_batch = df.next_batch(batch_size,shuffle = True)