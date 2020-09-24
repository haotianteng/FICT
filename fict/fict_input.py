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
                 gene_list = None,
                 field = None,
                 cell_labels = None,
                 for_eval = False):
        self.gene_expression = gene_expression
        self.sample_n = gene_expression.shape[0]
        self.class_n = num_class
        self.gene_list = gene_list
        if field is None:
            field = np.zeros(self.sample_n)
        field = np.asarray(field)
        self.field = field.reshape((field.shape[0],1))
        if cell_coordinate.shape[1] == 2:
            self.coordinate = np.concatenate((cell_coordinate,1000*self.field),axis=1)
        else:
            self.coordinate = cell_coordinate
        self.field = field.reshape((field.shape[0]))
        self.adjacency = dop.get_adjacency(self.coordinate,threshold_distance)
        self.exclude_adjacency = dop.get_adjacency(self.coordinate,threshold_distance,exclude_self = True)
        self.for_eval = for_eval
        self.cell_labels = cell_labels
        self.type_prob = None
        self.nb_count = np.empty((self.sample_n,self.class_n))
        self.pca_components = None
        super().__init__(xs = (self.gene_expression,self.nb_count),
                       y = self.cell_labels,
                       for_eval = self.for_eval)
        
    def renew_neighbourhood(self,
                            type_prob,
                            threshold_distance = None,
                            nearest_k = None,
                            update_adj = False,
                            exclude_self= False,
                            partial_update = 1,
                            hard_update = False):
        if update_adj:
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
        if hard_update:
            predict = np.argmax(type_prob,axis = 1)
            type_prob = dop.one_hot_vector(predict,class_n = self.class_n)[0]
        if self.type_prob is None:
            self.type_prob = type_prob
            if partial_update<1:
                print("Warning, initial type probability is None, update partition is forced to 1.")
        else:
            choice_n = int(self.sample_n*partial_update)
            choice_idx = np.random.choice(self.sample_n,choice_n,replace = False)
            self.type_prob[choice_idx] = type_prob[choice_idx]
        self.nb_count = dop.get_neighbourhood_count(self.adjacency,
                                                    self.type_prob,
                                                    exclude_self = exclude_self,
                                                    one_hot_label = True)
        self.xs = (self.xs[0],self.nb_count)
        
    def dim_reduce(self,dims = 10,method = "PCA",embedding = None,pca_components = None):
        if method == "PCA":
            self.reduced_gene_expression,self.pca_components = dop.pca_reduce(self.gene_expression,dims = dims,pca = pca_components)
            self.xs = (self.reduced_gene_expression,self.nb_count)    
        elif method == "TSNE":
            self.reduced_gene_expression,_ = dop.tsne_reduce(self.gene_expression,dims = dims)
            self.xs = (self.reduced_gene_expression,self.nb_count)
        elif method == "Embedding":
            self.reduced_gene_expression,_ = dop.embedding_reduce(self.gene_expression,embedding = embedding)
            self.xs = (self.reduced_gene_expression,self.nb_count)
if __name__ == "__main__":
    pass