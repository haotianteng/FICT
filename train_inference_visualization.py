#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:35:03 2020

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

GENE_ROUND = 20
SPATIAL_ROUND = 20
BOTH_ROUND = 20
THRESHOLD_DISTANCE = 20

def load_train(data_loader):
    data_loader.dim_reduce(dims = 10,method = "PCA")
    int_y,tags = tag2int(data_loader.y)
    data_loader.y = int_y
    one_hot_label,tags = one_hot_vector(int_y)
    data_loader.renew_neighbourhood(one_hot_label,nearest_k = 20)
    num_class = len(tags)
    num_gene = data_loader.xs[0].shape[1]
    print(num_gene)
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
    ## Load data and train the model
    data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/df_1"
    print("Load the data loader.")
    loader = load_loader(data_f)
    test_n = 500
    real_loader = RealDataLoader(loader.gene_expression[:test_n],
                                         loader.coordinate[:test_n],
                                         20,
                                         loader.class_n,
                                         field = loader.field[:test_n],
                                         cell_labels = loader.cell_labels[:test_n])
    m2 = load_train(real_loader)
    nb_count = real_loader.xs[1]
    nb_freq = nb_count/np.sum(nb_count,axis = 1,keepdims = True)
    print("Using PCA to get the dimensional-reduced gene expression matrix.")
    nb_reduced = pca_reduce(nb_freq,dims = 10)
    
    ## Visualize the result
    y = real_loader.y
    tags = list(set(y))
    colors = cm.get_cmap('Set2', len(tags))
    y_color = [tags.index(x) for x in y]
    print("Using TSNE to get the dimensional-reduced neighbourhood count matrix.")
    nb_tsne = manifold.TSNE(method = 'exact').fit_transform(nb_freq)
    gene_tsne = real_loader.reduced_gene_expression
    fig,axes = plt.subplots(nrows = 3,ncols = 2)
    
    ### Plot the GMM model
    ax = axes[0][0]
    centroid_ellipse(real_loader.xs[0],real_loader.y,m2,ax)
    ax.title.set_text("GMM model on gene expression data.")
    
    ### Visualize the original label in neighbourhood space.
    tags = list(set(y))
    int_c = [tags.index(x) for x in y]
    ax = axes[0][1]
    ax.scatter(nb_tsne[:,0],nb_tsne[:,1],cmap = colors,c=int_c,s = 2)
    ax.title.set_text("Neighborhood frequency visualization.")    

    ### Visualize the spatial model 
    ax = axes[1][0]
    int_label,tags = tag2int(real_loader.y)
    one_hot_label,tags = one_hot_vector(int_label)
    real_loader.renew_neighbourhood(one_hot_label,
                                    nearest_k = 20,
                                    exclude_self = True)
    posterior_spatio,_,_ = m2.expectation(real_loader.xs,
                   spatio_factor = 1,
                   gene_factor = 0,
                   prior_factor = 0)
    predict_spatio = np.argmax(posterior_spatio,axis = 0)
    ax.scatter(nb_tsne[:,0],nb_tsne[:,1],c=predict_spatio,cmap = colors,s = 2)
    ax.title.set_text("Spatial model prediction projection on neighbourhood space.")
    
    ax = axes[1][1]
    ax.scatter(gene_tsne[:,0],gene_tsne[:,1],c=predict_spatio,cmap = colors,s = 2)
    ax.title.set_text("Spatial model prediction projection on gene expression space.")
    accuracy = adjusted_rand_score(predict_spatio,y)
    print("Rand score with spatio information only %.4f"%(accuracy))
    
    ### Expect using Spatial+gene, maximize the spatial model
    nb_count = real_loader.xs[1]
    y = real_loader.y
    posterior,_,_ = m2.expectation(real_loader.xs,
                   spatio_factor = 1,
                   gene_factor = 1,
                   prior_factor = 1)
    predict_both = np.argmax(posterior,axis=0) 
    accuracy = adjusted_rand_score(predict_both,y)
    ax = axes[2][0]
    ax.scatter(nb_tsne[:,0],nb_tsne[:,1],c=predict_both,cmap = colors,s = 2)
    ax.title.set_text("Spatial+gene expection, spatial maximization training model.")
    ax = axes[2][1]
    ax.scatter(gene_tsne[:,0],gene_tsne[:,1],c=predict_both,cmap = colors,s = 2)
    ax.title.set_text("Spatial+gene expection, spatial maximization training model.")
