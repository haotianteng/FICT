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
from fict.utils.data_op import tsne_reduce
from fict.utils.data_op import pca_reduce
from fict.utils.data_op import one_hot_vector
from fict.utils.data_op import KL_divergence
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle
from fict.utils.data_op import tag2int,load_loader
from fict.fict_train import alternative_train
from fict.fict_train import centroid_ellipse
import seaborn as sns

TRAIN_CONFIG = {'gene_phase':{},'spatio_phase':{}}
TRAIN_CONFIG['gene_round'] = 20
TRAIN_CONFIG['spatio_round'] = 20
TRAIN_CONFIG['both_round'] = 20
TRAIN_CONFIG['gene_phase'] = {'gene_factor':1.0,
                              'spatio_factor':0.0,
                              'prior_factor':0.0}
TRAIN_CONFIG['spatio_phase'] = {'gene_factor':1.0,
                                'spatio_factor':1.0,
                                'prior_factor':0.0,
                                'nearest_k':20,
                                'threshold_distance':None,
                                'partial_update':0.1}

def load_train(data_loader,num_class = None):
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
    TRAIN_CONFIG['batch_size'] = data_loader.xs[0].shape[0]
    alternative_train(data_loader,
                      model,
                      train_config = TRAIN_CONFIG)
    return model

def gene_visualization(predict,loader,ax):
    class_n = len(set(predict))
    colors = cm.get_cmap('Set2', class_n)
    print("Reduce the dimension by T-SNE")
    gene_reduced = tsne_reduce(loader.xs[0],
                               method = 'barnes_hut')
    ax.scatter(gene_reduced[:,0],
               gene_reduced[:,1],
               c=predict,
               cmap = colors,
               s = 2)
    return ax

def index_match(p_ref,p,metrics = 'KLD'):
    n1 = len(p_ref)
    n2 = len(p)
    dists = np.zeros((n1,n2))
    for i,pi in enumerate(p):
        for j,pj in enumerate(p_ref):
            if metrics == 'KLD':
                dists[i,j] = KL_divergence(pj,pi)
            elif metrics == 'Eular':
                dists[i,j] = np.sqrt(np.sum((pi-pj)**2))
    perm = np.argmin(dists,axis = 1)
    return perm,dists
    
def heatmap(cv,ax,xticks= None,yticks = None):
    n,m = cv.shape
    im = ax.imshow(cv)
    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    if xticks is not None:
        ax.set_xticklabels(xticks[:n])
    if yticks is not None:
        ax.set_yticklabels(yticks[:m])
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(n):
        for j in range(m):
            text = ax.text(j, i, "%.2f"%(cv[i,j]),
                           ha="center", va="center", color="w")
    ax.set_title("Cross validation")
    return ax

if __name__ == "__main__":
    data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/df_1"
    print("Load the data loader.")
    loader = load_loader(data_f)
    fields = list(set(loader.field))
    fields = np.sort(fields)
    n_class = 10
    reduced_dim = 20

    ### Training the models
    loaders = []
    models = []
    for f in fields:
        mask = loader.field == f
        l = RealDataLoader(loader.gene_expression[mask],
                           loader.coordinate[mask],
                           20,
                           n_class,
                           field = loader.field[mask],
                           cell_labels = loader.cell_labels[mask])
        l.dim_reduce(dims = reduced_dim,method = "PCA")
        loaders.append(l)
        m = load_train(l,num_class=10)
        models.append(m) 
    with open("/home/heavens/CMU/FISH_Clustering/MERFISH_data/models/loaders.bn",'wb+') as f:
        pickle.dump(loaders,f)
    with open("/home/heavens/CMU/FISH_Clustering/MERFISH_data/models/trained_models.bn",'wb+') as f:
        pickle.dump(models,f)
    ###
    
    ### Load the models and loaders from previous record
    with open("/home/heavens/CMU/FISH_Clustering/MERFISH_data/models/loaders.bn",'rb') as f:
        loaders = pickle.load(f)
    with open("/home/heavens/CMU/FISH_Clustering/MERFISH_data/models/trained_models.bn",'rb') as f:
        models = pickle.load(f)
    ###
    n=5
    renew_round = 10
    cv_gene = np.zeros((n,n))
    cv_spatio = np.zeros((n,n))
    e_gene = np.empty((n,n,2),dtype = np.object)
    e_spatio = np.empty((n,n,2),dtype = np.object)
    pca_eigens = []
    for i in np.arange(n):
        _,components = pca_reduce(loaders[i].gene_expression,dims = reduced_dim)
        pca_eigens.append(components)
        
    for i in range(n):
        for j in range(n):
            print((i,j))
            batch_i = (pca_reduce(loaders[i].gene_expression,pca = pca_eigens[i])[0],loaders[i].xs[1])
            batch_j = (pca_reduce(loaders[i].gene_expression,pca = pca_eigens[j])[0],loaders[i].xs[1])
            e1,_,_ = models[i].expectation(batch_i,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            e2,_,_ = models[j].expectation(batch_j,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            e_gene[i,j,0] = e1
            e_gene[i,j,1] = e2
            cv_gene[i,j] = adjusted_rand_score(np.argmax(e1,axis = 0),np.argmax(e2,axis = 0))
    for i in range(n):
        for j in range(n):
            print((i,j))
            batch_i = (pca_reduce(loaders[i].gene_expression,pca = pca_eigens[i])[0],loaders[i].xs[1])
            batch_j = (pca_reduce(loaders[i].gene_expression,pca = pca_eigens[j])[0],loaders[i].xs[1])
            e1,_,_ = models[i].expectation(batch_i,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            loaders[i].renew_neighbourhood(e1.T,
                                           nearest_k =20)
            for k in np.arange(renew_round):
                e1,_,_ = models[i].expectation(batch_i,
                                               gene_factor = 1,
                                               spatio_factor = 1,
                                               prior_factor = 0)
                loaders[i].renew_neighbourhood(e1.T,
                                               nearest_k =20,
                                               partial_update = 0.1)
            e1,_,_ = models[i].expectation(batch_i,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 0)
            e2,_,_ = models[j].expectation(batch_j,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            loaders[i].renew_neighbourhood(e2.T,
                                          nearest_k = 20)
            for k in np.arange(renew_round):
                e2,_,_ = models[j].expectation(batch_j,
                                               gene_factor = 1,
                                               spatio_factor = 1,
                                               prior_factor = 0)
                loaders[i].renew_neighbourhood(e2.T,
                                               nearest_k =20,
                                               partial_update = 0.1)
            e2,_,_ = models[j].expectation(batch_j,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 0)
            cv_spatio[i,j] = adjusted_rand_score(np.argmax(e1,axis = 0),np.argmax(e2,axis = 0))
            e_spatio[i,j,0] = e1
            e_spatio[i,j,1] = e2
    ### Visualize the image
    figs,axs = plt.subplots(nrows=1,ncols=2)
    cvs = [cv_gene,cv_spatio]
    for i in np.arange(2):
        ax = axs[i]
        cv = cvs[i]
        heatmap(cv,ax,xticks = fields, yticks = fields)
    figs.tight_layout()
    plt.show()