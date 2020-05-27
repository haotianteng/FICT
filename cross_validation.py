#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:25:16 2020

@author: haotian teng
"""
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from fict.fict_model import FICT_EM
from fict.fict_input import RealDataLoader
from fict.utils.data_op import pca_reduce
from fict.utils.data_op import one_hot_vector
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib import cm
from fict.utils.data_op import tag2int,load_loader
from fict.fict_train import alternative_train
from fict.fict_train import centroid_ellipse
import seaborn as sns

GENE_ROUND = 20
SPATIAL_ROUND = 20
BOTH_ROUND = 20
THRESHOLD_DISTANCE = 20

def load_train(data_loader,num_class = None):
    data_loader.dim_reduce(dims = 10,method = "PCA")
    int_y,tags = tag2int(data_loader.y)
    data_loader.y = int_y
    if num_class is None:
        one_hot_label,tags = one_hot_vector(int_y)
        data_loader.renew_neighbourhood(one_hot_label,nearest_k = 20)
        num_class = len(tags)
    else:
        arti_label = np.random.randint(low = 0, 
                                       high = num_class,
                                       size = data_loader.sample_n)
        one_hot_label,tags = one_hot_vector(arti_label)
        data_loader.renew_neighbourhood(one_hot_label,nearest_k = 20)
    num_gene = data_loader.xs[0].shape[1]
    model = FICT_EM(num_gene,
                    num_class)
    batch_n = data_loader.xs[0].shape[0]

    alternative_train(data_loader,
                      model,
                      gene_round = GENE_ROUND,
                      spatial_round = SPATIAL_ROUND,
                      both_round = BOTH_ROUND,
                      threshold_distance=THRESHOLD_DISTANCE,
                      batch_size = batch_n)
    return model
    
if __name__ == "__main__":
    data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/df_1"
    print("Load the data loader.")
    loader = load_loader(data_f)
    fields = list(set(loader.field))
    loaders = []
    models = []
    for f in fields:
        mask = loader.field == f
        l = RealDataLoader(loader.gene_expression[mask],
                           loader.coordinate[mask],
                           20,
                           5,
                           field = loader.field[mask],
                           cell_labels = loader.cell_labels[mask])
        loaders.append(l)
        m = load_train(l,num_class=10)
        models.append(m) 
    n=4
    cv_gene = np.zeros((n,n))
    cv_spatio = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            print((i,j))
            e1,_,_ = models[i].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 1)
            e2,_,_ = models[j].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 1)
            cv_gene[i,j] = adjusted_rand_score(np.argmax(e1,axis = 0),np.argmax(e2,axis = 0))
    sns.heatmap(cv_gene)
    for i in range(n):
        for j in range(n):
            print((i,j))
            e1,_,_ = models[i].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 1)
            loaders[i].renew_neighbourhood(e1.T,
                                           nearest_k =20)
            e1,_,_ = models[i].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 1)
            e2,_,_ = models[j].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 1)
            loaders[i].renew_neighbourhood(e2.T,
                                          nearest_k = 20)
            e2,_,_ = models[j].expectation(loaders[i].xs,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 1)
            cv_spatio[i,j] = adjusted_rand_score(np.argmax(e1,axis = 0),np.argmax(e2,axis = 0))
    sns.heatmap(cv_spatio)