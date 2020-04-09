#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:51 2020

@author: heavens
"""
import numpy as np
import pandas as pd
from fict.utils import data_op as dop
import pickle 

class RealDataLoader(dop.DataLoader):
    """Data loader for the real FISH data.
    Args:
        gene_expression: A N-by-M matrix indicate the gene expression, where N
            is the number of samples, M is the number of genes.
        cell_coordinate: A N-by-2 matrix indicate the location of the cells in
            the 2D plane, N is the number of samples.
        threshold_distance: A float indicate the thershold distance of two cells
            being considered as neighbour.
        cell_type_probability: A N-by-K matrix indicate the probability of type
            for each cell, N is the number of samples(cells), K is the number of
            possible cell types.
        cell_labels: A N-by-1 matrix indicate the true label of cell types.
    """
    def __init__(self,
                 gene_expression,
                 cell_coordinate,
                 threshold_distance,
                 cell_type_probability,
                 cell_labels = None,
                 for_eval = False):
        self.gene_expression = gene_expression
        self.coordinate = cell_coordinate
        self.adjacency = dop.get_adjacency(self.coordinate,threshold_distance)
        self.for_eval = for_eval
        self.cell_labels = cell_labels
        self.renew_neighbourhood(cell_type_probability)
        super().__init__(xs = (self.gene_expression,self.nb_count),
                       y = self.cell_labels,
                       for_eval = self.for_eval)
        
    def renew_neighbourhood(self,type_prob,threshold_distance = None,exclude_self= False):
        if threshold_distance is not None:
            self.adjacency = dop.get_adjacency(self.coordinate,threshold_distance)
        self.nb_count = dop.get_neighbourhood_count(self.adjacency,
                                                type_prob,
                                                exclude_self = exclude_self,
                                                one_hot_label = True)
        self.xs = (self.gene_expression,self.nb_count)

if __name__ == "__main__":
    ### Hyper parameter setting
    print("Setting hyper parameter")
    sample_n = 1000 #Number of samples
    n_g = 15 #Number of genes
    n_c = 10 #Number of cell type
    density = 20 #The average number of neighbour for each cells.
    threshold_distance = 100 # The threshold distance of neighbourhood.
    gene_col = np.arange(9,164)
    coor_col = [5,6]
    header = 0
    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/aau5324_Moffitt_Table-S7.xlsx"
#    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
    save_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/df_"
    ### Data preprocessing
    print("Reading data from %s"%(data_f))
    if data_f.endswith('.xlsx'):
        data_all = pd.read_excel(data_f,header = header)
    elif data_f.endswith('.csv'):
        data_all = pd.read_csv(data_f,header = header)
    animal_idxs = np.unique(data_all['Animal_ID'])
    gene_expression_all = data_all.iloc[:,gene_col]
    nan_cols = np.unique(np.where(np.isnan(gene_expression_all))[1])
    for nan_col in nan_cols:
        gene_col = np.delete(gene_col,nan_col)
    for animal_id in animal_idxs:
        print("Extract the data for animal %d"%(animal_id))
        data = data_all[data_all['Animal_ID']==animal_id]
        cell_types = data['Cell_class']
        data = data[cell_types!= 'Ambiguous']
        cell_types = data['Cell_class']
        gene_expression = data.iloc[:,gene_col]
        type_tags = np.unique(cell_types)
        coordinates = data.iloc[:,coor_col]
        coordinates = np.asarray(coordinates)
        ### Choose only the n_c type cells
#        print("Choose the subdataset of %d cell types"%(n_c))
#        if len(type_tags)<n_c:
#            raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
#        mask = np.asarray([False]*len(cell_types))
#        for tag in type_tags[:n_c]:
#            mask = np.logical_or(mask,cell_types==tag)
#        gene_expression = gene_expression[mask]
#        cell_types = np.asarray(cell_types[mask])
#        coordinates = np.asarray(coordinates[mask])
        gene_expression = np.asarray(gene_expression)
        gene_expression = gene_expression/np.sum(gene_expression,axis = 1,keepdims = True)
        #gene_expression_reduced = tsne_reduce(gene_expression,dims = 2)
        gene_expression_reduced = dop.pca_reduce(gene_expression,dims = n_g)
        init_prob = np.ones((gene_expression_reduced.shape[0],n_c))*1.0/n_c
        real_df = RealDataLoader(gene_expression_reduced,
                                 coordinates,
                                 threshold_distance = threshold_distance,
                                 cell_type_probability = init_prob,
                                 cell_labels = cell_types,
                                 for_eval = False)
        save_loader(real_df,save_f+str(animal_id))