#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:51 2020

@author: heavens
"""
import numpy as np
import pandas as pd
from fict.utils import data_op as dop

class RealDataLoader(dop.DataLoader):
    """Data loader for the real FISH data.
    Args:
        gene_expression: A N-by-M matrix indicate the gene expression, where N
            is the number of samples, M is the number of genes.
        cell_coordinate: A N-by-2 matrix indicate the location of the cells in
            the 2D plane, N is the number of samples.
        threshold_distance: A float indicate the thershold distance of two cells
            being considered as neighbour.
        num_class:The number of class.
        cell_labels: A N-by-1 matrix indicate the true label of cell types.
    """
    def __init__(self,
                 gene_expression,
                 cell_coordinate,
                 threshold_distance,
                 num_class,
                 field = None,
                 cell_labels = None,
                 for_eval = False):
        self.gene_expression = gene_expression
        self.sample_n = gene_expression.shape[0]
        self.class_n = num_class
        self.coordinate = cell_coordinate
        self.field = field
        self.adjacency = dop.get_adjacency(self.coordinate,threshold_distance)
        self.exclude_adjacency = dop.get_adjacency(self.coordinate,threshold_distance,exclude_self = True)
        self.for_eval = for_eval
        self.cell_labels = cell_labels
        self.nb_count = np.empty((self.sample_n,self.class_n))
        super().__init__(xs = (self.gene_expression,self.nb_count),
                       y = self.cell_labels,
                       for_eval = self.for_eval)
        
    def renew_neighbourhood(self,
                            type_prob,
                            threshold_distance = None,
                            nearest_k = None,
                            exclude_self= False):
        if (nearest_k is None) and (threshold_distance is None):
            raise TypeError("renew_neighbourhood require at least input one of\
                            the following arguments:threshold_distance, nearest_k")
        if threshold_distance is not None:
            self.adjacency = dop.get_adjacency(self.coordinate,
                                               threshold_distance,
                                               exclude_self = exclude_self)
        elif nearest_k is not None:
            self.adjacency = dop.get_adjacency_knearest(self.coordinate,
                                              nearest_k,
                                              exclude_self = exclude_self)
        self.nb_count = dop.get_neighbourhood_count(self.adjacency,
                                                    type_prob,
                                                    exclude_self = exclude_self,
                                                    one_hot_label = True)
        self.xs = (self.xs[0],self.nb_count)
    
    def dim_reduce(self,dims = 10,method = "PCA"):
        if method == "PCA":
            self.reduced_gene_expression = dop.pca_reduce(self.gene_expression,dims = dims)
            self.xs = (self.reduced_gene_expression,self.nb_count)    
        elif method == "TSNE":
            self.reduced_gene_expression = dop.tsne_reduce(self.gene_expression,dims = dims)
            self.xs = (self.reduced_gene_expression,self.nb_count) 

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
    data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH2018/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
    save_f = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/df_"
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
        bregma = data['Bregma']
        gene_expression = data.iloc[:,gene_col]
        type_tags = np.unique(cell_types)
        coordinates = data.iloc[:,coor_col]
        coordinates = np.asarray(coordinates)
        gene_expression = np.asarray(gene_expression)
        gene_expression = gene_expression/np.sum(gene_expression,axis = 1,keepdims = True)
        real_df = RealDataLoader(gene_expression,
                                 coordinates,
                                 threshold_distance = threshold_distance,
                                 cell_labels = cell_types,
                                 num_class = n_c,
                                 field = bregma,
                                 for_eval = False)
        dop.save_loader(real_df,save_f+str(animal_id))