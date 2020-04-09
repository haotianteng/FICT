#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:57:24 2020

@author: heavens
"""
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from fict.fict_model import FICT_EM
from fict.fict_input import RealDataLoader
from time import time
from sklearn.decomposition import PCA
from sklearn import manifold
from matplotlib import pyplot as plt

def pca_reduce(X, dims=2):
    """ Reduce the dimensions of X down to dims using PCA
    X has shape (n, d)
    Returns: The reduced X of shape (n, dims)
    		 The fitted PCA model used to reduce X
    """
    print("reducing dimensions using PCA")
    X = X - np.mean(X,axis = 0,keepdims = True)
    X = X/np.std(X,axis = 0, keepdims = True)
    pca = PCA(n_components = dims)
    pca.fit(X)
    X_reduced = pca.transform(X)
    return X_reduced

def tsne_reduce(X,dims = 5):
    tsne = manifold.TSNE(n_components=dims, init='pca', random_state=0)
    t_expression = tsne.fit_transform(X)
    return t_expression

def train(model,
          decay,
          train_rounds,
          data_loader,
          batch_size,
          spatio_factor = 0.5, 
          gene_factor = 0.5,
          prior_factor = 1.0,
          update_spatio = True,
          update_gene = True,
          verbose = True,
          report_per_rounds = 10,
          renew_per_rounds = 10):
    accur_record = []
    for i in range(train_rounds):
        x_batch,y = data_loader.next_batch(batch_size,shuffle = True)
        posterior = model.expectation(x_batch,
                                   spatio_factor = spatio_factor,
                                   gene_factor = gene_factor,
                                   prior_factor = prior_factor)
        model.maximization(x_batch,posterior,decay = decay,update_spatio_model = False)
        predict = np.argmax(posterior,axis=0) 
        accuracy = adjusted_rand_score(predict,y)
        accur_record.append(accuracy)
#        if i%renew_rounds == 0:
#            posterior_all = m2.expectation(data_loader.xs,spatio_factor = spatio_factor,gene_factor = gene_factor)
#            data_loader.renew_neighbourhood(np.transpose(posterior_all))
        if i%report_per_rounds == 0 and verbose:
            print("%d Round Accuracy:%f"%(i,accuracy))
    return accur_record

if __name__ == "__main__":
    from fict.utils.data_op import tag2int
    from fict.utils.data_op import one_hot_vector
    from fict.utils.data_op import load_loader
    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/df_test"
    real_loader = load_loader(data_f)
    nb_count = real_loader.xs[1]
    nb_freq = nb_count
#    nb_freq = nb_count/np.sum(nb_count,axis = 1,keepdims = True)
    nb_freq = nb_freq - np.mean(nb_freq,axis = 0)
    nb_reduced = manifold.TSNE().fit_transform(nb_freq)
    y = real_loader.y
    colors = ['Grey', 'Purple', 'Blue', 'Green', 'Orange', 'Red',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cells = np.unique(real_loader.y)
    for i,cell in enumerate(cells):
        plt.plot(nb_reduced[y==cell,0],nb_reduced[y==cell,1],'.')
    n_c = 10
    m2 = FICT_EM(real_loader.gene_expression.shape[1],n_c)
    renew_rounds = 10
    batch_n = 5000
    gene_round = 100
    spatial_round = 100
    both_round = 50
    for threshold_distance in np.arange(10,200,10):
        int_label,tags = tag2int(real_loader.y)
        one_hot_label = one_hot_vector(int_label)
        real_loader.renew_neighbourhood(one_hot_label,
                                        threshold_distance = threshold_distance,
                                        exclude_self = True)
    #    print("Begin training using gene expression and spatio information.")
    #    accur_record_both = train(m2,
    #          0.5,
    #          both_round,
    #          real_loader,
    #          batch_n,
    #          spatio_factor = 0)
        print("Train the spatio model only.")
    #    accur_record_spatio = train(m2,
    #          0.5,
    #          spatial_round,
    #          real_loader,
    #          batch_n,
    #          spatio_factor = 2.0,
    #          gene_factor = 0,
    #          prior_factor = 0.0,
    #          update_spatio = True,
    #          update_gene = False)
        nb_count = real_loader.xs[1]
        y = real_loader.y
        for i,tag in enumerate(tags):
            mean_nb_count = np.mean(nb_count[y==tag],axis = 0)
            m2.p['mn_p'][i] = mean_nb_count/np.sum(mean_nb_count)
        posterior = m2.expectation(real_loader.xs,
                       spatio_factor = 1,
                       gene_factor = 0,
                       prior_factor = 0)
        predict = np.argmax(posterior,axis=0) 
        accuracy = adjusted_rand_score(predict,y)
        print("Accuracy with spatio information only and %.2f threshold distance %.4f"%(threshold_distance,accuracy))
