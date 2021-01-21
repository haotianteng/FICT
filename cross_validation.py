#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:25:16 2020

@author: haotian teng
"""
import os
import json
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from fict.fict_model import FICT_EM
from fict.fict_input import RealDataLoader
from fict.utils.data_op import tsne_reduce
from fict.utils.data_op import pca_reduce
from fict.utils.data_op import embedding_reduce
from fict.utils.data_op import one_hot_vector
from fict.utils.data_op import KL_divergence
from fict.utils.data_op import get_knearest_distance
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle
from fict.utils.data_op import tag2int,load_loader
from fict.utils import embedding as emb
from fict.fict_train import alternative_train
from fict.fict_train import centroid_ellipse
from fict.utils import data_op as dop
import seaborn as sns
import argparse
import sys

TRAIN_CONFIG = {'gene_phase':{},'spatio_phase':{}}
TRAIN_CONFIG['gene_round'] = 20
TRAIN_CONFIG['spatio_round'] = 10
TRAIN_CONFIG['both_round'] = 10
TRAIN_CONFIG['verbose'] = 1
TRAIN_CONFIG['gene_phase'] = {'gene_factor':1.0,
                              'spatio_factor':0.0,
                              'prior_factor':0.0}
TRAIN_CONFIG['spatio_phase'] = {'gene_factor':1.0,
                                'spatio_factor':1.0,
                                'prior_factor':0.0,
                                'nearest_k':10,
                                'threshold_distance':None,
                                'renew_rounds':5,
                                'partial_update':1,
                                'equal_contribute':False}

def load_pickle(f):
    with open(f,'rb') as x:
        obj = pickle.load(x)
    return(obj)

def load_train(data_loader,num_class = None):
    int_y,tags = tag2int(data_loader.y)
    data_loader.y = int_y
    if num_class is None:
        one_hot_label,tags = one_hot_vector(int_y)
        data_loader.renew_neighbourhood(one_hot_label,
                                        nearest_k = TRAIN_CONFIG['spatio_phase']['nearest_k'],
                                        threshold_distance = TRAIN_CONFIG['spatio_phase']['threshold_distance'],
                                        update_adj = True)
        num_class = len(tags)
    else:
        arti_label = np.random.randint(low = 0, 
                                       high = num_class,
                                       size = data_loader.sample_n)
        one_hot_label,tags = one_hot_vector(arti_label)
        data_loader.renew_neighbourhood(one_hot_label,
                                        nearest_k = TRAIN_CONFIG['spatio_phase']['nearest_k'],
                                        threshold_distance = TRAIN_CONFIG['spatio_phase']['threshold_distance'],
                                        update_adj = True)
    num_gene = data_loader.xs[0].shape[1]
    model = FICT_EM(num_gene,
                    num_class)
    TRAIN_CONFIG['batch_size'] = data_loader.xs[0].shape[0]
    alternative_train(data_loader,
                      model,
                      train_config = TRAIN_CONFIG)
    return model

def cluster_visualization(posterior,loader,ax,mode = 'gene'):
    """Visualize the cluster
    Input:
        posterior: The posterior probability .
        loader: The dataloader.
        ax: The axes of the figure that is going to be printed on.
        mode: Can be one of the following mode:
            gene, neighbourhood, coordinate.
    """
    predict = np.argmax(posterior,axis = 0)
    class_n = len(set(predict))
    colors = cm.get_cmap('Set2', class_n)
    print("Reduce the dimension by T-SNE")
    if mode == 'gene':
        locs = tsne_reduce(loader.xs[0],
                                   method = 'barnes_hut')
    elif mode == 'coordinate':
        locs = loader.coordinate
    elif mode == 'neighbourhood':
        locs = tsne_reduce(loader.xs[1],method = 'barnes_hut')
    ax.scatter(locs[:,0],
               locs[:,1],
               c=predict,
               cmap = colors,
               s = 5)
    return ax

def compare_visual(e_gene,e_spatio,loaders,i,j):
    figs,axs = plt.subplots(nrows = 2,ncols = 2)
    figs.set_size_inches(24,h=12)
    loader = loaders[i]
    cluster_visualization(e_gene[i,j,0],loader,axs[0][0],mode = 'coordinate')
    cluster_visualization(e_gene[i,j,1],loader,axs[0][1],mode = 'coordinate')
    cluster_visualization(e_spatio[i,j,0],loader,axs[1][0],mode = 'coordinate')
    cluster_visualization(e_spatio[i,j,1],loader,axs[1][1],mode = 'coordinate')
    axs[0][0].set_title("Gene model %d on dataset %d"%(i,i))
    axs[0][1].set_title("Gene model %d on dataset %d"%(j,i))
    axs[1][0].set_title("Spatio model %d on dataset %d"%(i,i))
    axs[1][1].set_title("Spatio model %d on dataset %d"%(j,i))
    return figs,axs

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
    
def heatmap(cv,ax,xticks= None,yticks = None,title = ''):
    n,m = cv.shape
    _ = ax.imshow(cv)
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
    ax.set_title(title)
    return ax

def run(args):
    data_f = args.input
    result_f = args.output
    n_class = args.n_class
    reduced_dim = args.reduced_dim
    k_nearest = args.k_nearest
    thres_dist = args.threshold_distance
    renew_round = args.renew_round
    spatio_factor = args.spatio_factor
    equal_contribute = args.equal_contribute
    reduced_method = args.reduced_method
    embedding_file = args.embedding_file
    if (k_nearest is None) and (thres_dist is None):
        print("Either nearest_k or threshold_distance is not provided,"+\
              "default nearest_k is used %d."%(TRAIN_CONFIG['spatio_phase']['nearest_k']))
        k_nearest = TRAIN_CONFIG['spatio_phase']['nearest_k']
    elif (k_nearest is not None) and (thres_dist is not None):
        print("Warning, both nearest_k and threshold_distance are provided,"+
              "nearest_k argument will not be used.")
        k_nearest = None
        TRAIN_CONFIG['spatio_phase']['nearest_k'] = None
        TRAIN_CONFIG['spatio_phase']['threshold_distance'] = thres_dist
    else:
        TRAIN_CONFIG['spatio_phase']['nearest_k'] = k_nearest
        TRAIN_CONFIG['spatio_phase']['threshold_distance'] = thres_dist
    n=args.k_fold
    TRAIN_CONFIG['n_class'] = n_class
    TRAIN_CONFIG['reduced_dim'] = reduced_dim
    TRAIN_CONFIG['reduced_method'] = reduced_method
    TRAIN_CONFIG['embedding_file'] = embedding_file
    TRAIN_CONFIG['data_file'] = data_f
    TRAIN_CONFIG['spatio_phase']['renew_rounds'] =  renew_round
    TRAIN_CONFIG['spatio_phase']['spatio_factor'] = spatio_factor
    TRAIN_CONFIG['spatio_phase']['equal_contribute'] = equal_contribute
    config_f = os.path.join(result_f,"config")
    embedding = emb.load_embedding(embedding_file)
    if not os.path.isdir(result_f):
        os.mkdir(result_f)
    with open(config_f,'w+') as f:
        json.dump(TRAIN_CONFIG,f)
    print("Load the data loader.")
    data_fs = data_f.split(',')
    if args.mode == 'multi':
        if len(data_fs) == 1:
            raise ValueError("Multiple datasets are required for multi cross validation mode.")
        loaders = []
        for f in data_fs:
            loaders.append(load_loader(f))
    else:
        loader = load_loader(data_fs[0])
### Get the relationship between threshold neighbourhood distance and knearest neighbour.
#    k_max = 30
#    knearest_dist = np.zeros(k_max)
#    for k in np.arange(k_max):
#        print("Calculate average distance for %d nearest"%(k))
#        dists = get_knearest_distance(loader.coordinate,
#                                      nearest_k = k+1)
#        knearest_dist[k] = np.mean(dists)
#    knearest_dist = np.asarray(knearest_dist)
#    with open(os.path.join(result_f,"average_k_distance.bn"),'wb+') as f:
#        pickle.dump(knearest_dist,f)
### 
    if args.load:
        print("Load the models.")
        with open(os.path.join(args.output,"loaders.bn"),'rb') as f:
            loaders = pickle.load(f)
            fields = np.arange(len(loaders))
        if n> len(fields) or n==0:
            print("Warning, the maximum k for k-fold cross-validation is %d"%(len(fields)))
            print("Use the number of fields %d instead of input %d."%(len(fields),n))
            n = len(fields)
        with open(os.path.join(args.output,"trained_models.bn"),'rb') as f:
            models = pickle.load(f)
    else:
        if args.mode != 'multi':
            fields = list(set(loader.field))
            fields = np.sort(fields)
            if args.mode == 'bregma':
                if n> len(fields) or n==0:
                    print("Warning, the maximum k for k-fold cross-validation is %d"%(len(fields)))
                    print("Use the number of fields %d instead of input %d."%(len(fields),n))
                    n = len(fields)
                def data_iterator():
                    for f in fields[:n]:
                        yield loader.field==f
            elif args.mode == 'random':
                split_group = np.random.randint(0,high=n,size = loader.sample_n)
                def data_iterator():
                    for i in np.arange(n):
                        yield split_group==i
            print("Model training begin.")
            loaders = []
            loader.dim_reduce(dims = reduced_dim,
                              method = reduced_method,
                              embedding = embedding)
            for mask in data_iterator():
                l = RealDataLoader(loader.gene_expression[mask],
                                   loader.coordinate[mask],
                                   20,
                                   n_class,
                                   field = loader.field[mask],
                                   cell_labels = loader.cell_labels[mask])
                l.dim_reduce(dims = reduced_dim,
                             method = reduced_method,
                             embedding = embedding)
                loaders.append(l)
        else:
            fields = np.arange(len(loaders))
            if n> len(fields) or n==0:
                print("Warning, the maximum k for k-fold cross-validation is %d"%(len(fields)))
                print("Use the number of fields %d instead of input %d."%(len(fields),n))
                n = len(fields)
            for i,l in enumerate(loaders):
                l = RealDataLoader(l.gene_expression,
                                   l.coordinate,
                                   20,
                                   n_class,
                                   field = np.asarray(l.field),
                                   cell_labels = l.cell_labels)
                l.dim_reduce(dims = reduced_dim,
                             method = reduced_method,
                             embedding = embedding)
                loaders[i] = l
        models = []
        for l in loaders:
            m = load_train(l,num_class = n_class)
            models.append(m)
        with open(os.path.join(result_f,"loaders.bn"),'wb+') as f:
            pickle.dump(loaders,f)
        with open(os.path.join(result_f,"trained_models.bn"),'wb+') as f:
            pickle.dump(models,f)
    ###
    loaders_bk = np.copy(loaders)
    renew_round = args.renew_round
    cv_gene = np.zeros((n,n))
    cv_spatio = np.zeros((n,n))
    e_gene = np.empty((n,n,2),dtype = np.object)
    e_spatio = np.empty((n,n,2),dtype = np.object)
    proj = []
    for i in np.arange(n):
        if reduced_method == 'PCA':
            _,components = pca_reduce(loaders[i].gene_expression,dims = reduced_dim)
            reduced_func = lambda x,y:pca_reduce(x,pca = y)
        elif reduced_method == 'TSNE':
            _,components = tsne_reduce(loaders[i].gene_expression,dims = reduced_dim)
            reduced_func = lambda x,y:tsne_reduce(x,tsne = y)
        elif reduced_method == 'Embedding':
            _,components = embedding_reduce(loaders[i].gene_expression,embedding = embedding)
            reduced_func = lambda x,y:embedding_reduce(x,embedding = y)
        proj.append(components)
    print("Begin Cross Validation")
    for i in range(n):
        for j in range(n):
            loaders = np.copy(loaders_bk)
            print((i,j))
            batch_i = (reduced_func(loaders[i].gene_expression,proj[i])[0],loaders[i].xs[1])
            batch_j = (reduced_func(loaders[i].gene_expression,proj[j])[0],loaders[i].xs[1])
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
            loaders = np.copy(loaders_bk)
            print((i,j))
            batch_i = (reduced_func(loaders[i].gene_expression,proj[i])[0],loaders[i].xs[1])
            batch_j = (reduced_func(loaders[i].gene_expression,proj[j])[0],loaders[i].xs[1])
            e1,_,_ = models[i].expectation(batch_i,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            loaders[i].renew_neighbourhood(e1.T,
                                           nearest_k =k_nearest,
                                           threshold_distance = thres_dist)
            for k in np.arange(renew_round):
                e1,_,_ = models[i].expectation(batch_i,
                                               gene_factor = 1,
                                               spatio_factor = 1,
                                               prior_factor = 0)
                loaders[i].renew_neighbourhood(e1.T,
                                               nearest_k =k_nearest,
                                               threshold_distance = thres_dist,
                                               partial_update = 0.1)
            e1,_,_ = models[i].expectation(batch_i,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 0)
            e2,_,_ = models[j].expectation(batch_j,
                                           gene_factor = 1,
                                           spatio_factor = 0,
                                           prior_factor = 0)
            loaders = np.copy(loaders_bk)
            loaders[i].renew_neighbourhood(e2.T,
                                           nearest_k = k_nearest,
                                           threshold_distance = thres_dist)
            for k in np.arange(renew_round):
                e2,_,_ = models[j].expectation(batch_j,
                                               gene_factor = 1,
                                               spatio_factor = 1,
                                               prior_factor = 0)
                loaders[i].renew_neighbourhood(e2.T,
                                               nearest_k =k_nearest,
                                               threshold_distance = thres_dist,
                                               partial_update = 0.1)
            e2,_,_ = models[j].expectation(batch_j,
                                           gene_factor = 1,
                                           spatio_factor = 1,
                                           prior_factor = 0)
            cv_spatio[i,j] = adjusted_rand_score(np.argmax(e1,axis = 0),np.argmax(e2,axis = 0))
            e_spatio[i,j,0] = e1
            e_spatio[i,j,1] = e2
    ### Visualize the result and save
    figs,axs = plt.subplots(nrows=1,ncols=2)
    figs.set_size_inches(12,h=6)
    cvs = [cv_gene,cv_spatio]
    titles = ['Cross validation of Gene','Cross validation of Gene+Spatio']
    for i in np.arange(2):
        ax = axs[i]
        cv = cvs[i]
        heatmap(cv,ax,xticks = fields, yticks = fields,title = titles[i])
    figs.savefig(os.path.join(result_f,'cv.png'),bbox_inches='tight')
    with open(os.path.join(result_f,"cv_result.bn"),'wb+') as f:
        pickle.dump([e_gene,e_spatio,cv_gene,cv_spatio],f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FICT',
                                     description='A cell type clsuter for FISH data.')
    parser.add_argument('-i', '--input', required = True,
                        help="The input data loader pickle file, multiple input file separated by comma.")
    parser.add_argument('-o','--output', required = True,
                        help="The output folder")
    parser.add_argument('--renew_round', default = 30, type = int,
                        help="The renew neighbourhood rounds.")
    parser.add_argument('--n_class', default = 7, type = int,
                        help="The number of output class.")
    parser.add_argument('-d','--reduced_dim', default = 20, type = int,
                        help="The reduced dimension of gene expression data.")
    parser.add_argument('--k_fold',default = 12, type = int,
                        help="The number of fold cross validation.")
    parser.add_argument('--k_nearest',default = None, type = int,
                        help="The number of nearest neighbourhood.")
    parser.add_argument('--threshold_distance',default = None, type = float,
                        help="The threshold distance of neighbourhood.")
    parser.add_argument('--load', action='store_true',
                        help="If the models has been trained already.")
    parser.add_argument('--spatio_factor',type = float, default = 1,
                        help="The spatio factor used in spatio model.")
    parser.add_argument('--reduced_method', default = 'PCA',
                        help="The method used to do dimension reduction, can be PCA, TSNE and Embedding.")
    parser.add_argument('--embedding_file', default = None,
                        help="The path of the embedding file if embedding is chosen to do dimension reduction.")
    parser.add_argument('--equal_contribute',action = "store_true", 
                        help="If normalize the probability of gene and spatio.")
    parser.add_argument('--mode', default='bregma',
                        help="How to divide the dataset for cross validation,can be one of the following: random, bregma.")
    args = parser.parse_args(sys.argv[1:])
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    run(args)
