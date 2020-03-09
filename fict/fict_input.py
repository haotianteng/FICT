#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:51 2020

@author: heavens
"""
import numpy as np
from fict.utils.data_op import DataLoader,get_adjacency,get_neighbourhood_count

class RealDataLoader(DataLoader):
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
        self.adjacency = get_adjacency(self.coordinate,threshold_distance)
        self.for_eval = for_eval
        self.cell_labels = cell_labels
        self.renew_neighbourhood(cell_type_probability)
        super().__init__(xs = (self.gene_expression,self.nb_count),
                       y = self.cell_labels,
                       for_eval = self.for_eval)
        
    def renew_neighbourhood(self,type_prob):
        self.nb_count = get_neighbourhood_count(self.adjacency,
                                                type_prob,
                                                exclude_self = False,
                                                one_hot_label = True)
        self.xs = (self.gene_expression,self.nb_count)

if __name__ == "__main__":
    df = RealDataLoader(np.asarray([[0],[1],[2]]),
                        np.asarray([[0,0],[2,1],[3,2]]),
                        threshold_distance = 1.4,
                        cell_labels = np.asarray([[1],[2],[3]]),
                        cell_type_probability = np.asarray([[1.0/3]*3]*3))
    x_batch,y_batch = df.next_batch(4)
        