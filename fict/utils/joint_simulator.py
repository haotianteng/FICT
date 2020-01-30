#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 03:01:35 2020

@author: heavens
"""
import numpy as np

sample_n = 1000 #Number of samples
n_g = 100 #Number of genes
n_c = 10 #Number of cell type
neighbours = np.arange(40,120,1)

class Simulator():
    def __init__(self,sample_n,gene_n,cell_n,neighbour_range,seed = 1992):
        self.sample_n = sample_n
        self.gene_n = gene_n
        self.cell_n = cell_n
        self.neighbour_range = neighbour_range
        self.seed = seed
        
    def gene_parameters(self,seed = None):
        self.all_types = np.arange(self.cell_n)
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        self.g_mean = np.random.rand(self.cell_n,self.gene_n)
        g_cov = np.random.rand(self.cell_n,self.gene_n,self.gene_n)
        self.g_cov = np.asarray([np.dot(x,x.transpose())/self.gene_n for x in g_cov])
        neighbour_p = np.random.rand(self.cell_n,self.cell_n)
        self.neighbour_p = neighbour_p/np.sum(neighbour_p,axis=1,keepdims=True)
        self.neighbour_n = np.random.choice(self.neighbour_range,size=sample_n)
        cell_prior = np.random.random(self.cell_n)
        self.cell_prior = cell_prior/np.sum(cell_prior)
    
    def gene_data(self,seed = None):
        gene_expression = []
        cell_type = []
        cell_neighbour = []
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        for i in range(sample_n):
            current_t = np.random.choice(self.all_types,p = self.cell_prior)
            cell_type.append(current_t)
            gene_expression.append(np.random.multivariate_normal(mean = self.g_mean[current_t],cov = self.g_cov[current_t]))
            cell_neighbour.append(np.random.multinomial(self.neighbour_n[i],pvals=self.neighbour_p[current_t]))
        return np.asarray(gene_expression),np.asarray(cell_type),np.asarray(cell_neighbour)

if __name__ == "__main__":
    sim = Simulator(sample_n,n_g,n_c,neighbours)
    sim.gene_parameters()
    gene_expression,cell_type,cell_neighbour = sim.gene_data()
