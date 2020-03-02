#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:47:51 2020

@author: heavens
"""

class Dataset(object):
    def __init__(self,gene_expression,neighbour_count):
        self.gene_expression = gene_expression
        self.neighbour_count = neighbour_count
        self.cell_n = self.gene_expression.shape[0]
        self.perm = np.arange(self.cell_n)
        
    def next(batch_size,)