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
from itertools import permutations
import matplotlib.transforms as transforms
from matplotlib import cm
import matplotlib 
import scipy
def permute_accuracy(predict,y):
    """Find the best accuracy among all the possible label permutation.
    Input args:
        predict: the clustering result.
        y: the true label.
    Return:
        best_accur: return the best accuracy.
        perm: return the permutation given the best accuracy.
    """
    label_tag = np.unique(y)
    sample_n = len(y)
    perms = list(permutations(label_tag))
    hits = []
    for perm in perms:
        hit = np.sum([(predict == p) * (y == i) for i,p in enumerate(perm)])
        hits.append(hit)
    return np.max(hits)/sample_n,perms[np.argmax(hits)]

def centroid_ellipse(x,y,m,axs):
    """Visualize the centroid ellipse of the GMM model m.
    Args:
        x: A N-by-2 dimensional-reduced matrix of the training data.
        y: The true label.
        m: The GMM model, it should contain the parameter dict p as its attribute
            p['g_mean']: A c-by-2 matrix given the mean of the GMM model for class c.
            p['g_cov']: A c-by-2-by-2 tensor given the covariance of the GMM model for class c.
    """
    tags = list(set(y))
    std_multiple = 2
    colors = cm.get_cmap('Set2', len(tags))
    y_color = [tags.index(x) for x in y]
    axs.scatter(x[:,0],x[:,1],c = y_color,cmap = colors,s = 2)
    for c in np.arange(m.p['class_n']):
        value,vectors = scipy.linalg.eig(m.p['g_cov'][c][0:2,0:2])
        value = abs(value)
        max_v = np.argmax(value)
        ellipse = matplotlib.patches.Ellipse((m.p['g_mean'][c,0],m.p['g_mean'][c,1]),
                                   width = np.sqrt(np.abs(value[max_v]))*std_multiple*2,
                                   height = np.sqrt(np.abs(value[1-max_v]))*std_multiple*2,
                                   angle = np.rad2deg(np.arctan(vectors[1][max_v]/vectors[0][max_v])),
                                   facecolor = 'none',
                                   edgecolor = colors(c))
        ellipse.set_transform(axs.transData)
        axs.add_patch(ellipse)
    return ellipse

def train(model,
          decay,
          train_rounds,
          data_loader,
          batch_size,
          tol = 1e-6,
          spatio_factor = 0.5, 
          gene_factor = 0.5,
          prior_factor = 1.0,
          update_spatio = True,
          update_gene = True,
          verbose = 0,
          stochastic_update = False,
          renew_neighbourhood = 10,
          report_per_rounds = 10,
          renew_per_rounds = 10,
          nearest_k = None,
          threshold_distance = None,
          partial_update = 1,
          equal_contrib = False):
    accur_record = []
    log_likelihood = []
    if nearest_k is None and threshold_distance is None:
        raise ValueError("nearest_k and threshold_distance can't be both None.")
    if nearest_k is not None and threshold_distance is not None:
        raise ValueError("nearest_k and threshold_distance can't be assigned at the same time.")
    if verbose>1:
        fig,axs = plt.subplots()
        centroid_ellipse(data_loader.xs[0],data_loader.y,model,axs)
        plt.show()
    for i in range(train_rounds):
        x_batch,y = data_loader.next_batch(batch_size,shuffle = True)
        posterior,ll,_ = model.expectation(x_batch,
                                   spatio_factor = spatio_factor,
                                   gene_factor = gene_factor,
                                   prior_factor = prior_factor,
                                   equal_contrib = equal_contrib)
        model.maximization(x_batch,posterior,
                           decay = decay,
                           update_gene_model = update_gene,
                           update_spatio_model = update_spatio,
                           stochastic_update = stochastic_update)
        predict = np.argmax(posterior,axis=0) 
        if y is not None:
            accuracy = adjusted_rand_score(predict,y)
            accur_record.append(accuracy)
        log_likelihood.append(ll)
        if i%renew_per_rounds == 0 and renew_neighbourhood:
            for renew_i in np.arange(renew_neighbourhood):
                posterior_all,ll_renew,_ = model.expectation(data_loader.xs,
                                                     spatio_factor = spatio_factor,
                                                     gene_factor = gene_factor,
                                                     prior_factor = prior_factor)
                data_loader.renew_neighbourhood(np.transpose(posterior_all),
                                                nearest_k = nearest_k,
                                                threshold_distance = threshold_distance,
                                                partial_update = partial_update)
                if verbose>0:
                    if renew_i==0:
                        print("\tRenew neighbourhood %d, likelihood change:%f"%(renew_i,ll_renew-ll))
                        ll_b = ll_renew
                    else:
                        print("\tRenew neighbourhood %d, likelihood change:%f"%(renew_i,ll_renew-ll_b))
        if i%report_per_rounds == 0:
            if verbose>1:
                fig,axs = plt.subplots()
                centroid_ellipse(x_batch[0],y,model,axs)
                plt.show()
            if verbose>0:
                if y is not None:
                    print("%d Round Match:%f"%(i,accuracy))
                if i >0:
                    ll_change = ll-log_likelihood[-2]
                    print("%d Round likelihood change:%f"%(i,ll_change))
                    if abs(ll_change)<tol:
                        return accur_record,log_likelihood
                else:
                    print("0 Round likelihood:%f"%(ll))
    return accur_record,log_likelihood

def alternative_train(data_loader,
                      model,
                      train_config):
    """The training pipeline for alternatively training spatial aware model.
    Args:
        data_loader: The data loader of the gene expression and negibhourhood
            frequency.
        model: The FICT_EM model class.
        train_config: The config file of the training parameters.
    """
    print("Initialize the gaussian model with kmeans++")
    model.gaussain_initialize(data_loader.xs[0])
    print("Begin training using gene expression only.")
    gene_round = train_config['gene_round']
    spatial_round = train_config['spatio_round']
    both_round = train_config['both_round']
    verbose = train_config['verbose']
    threshold_distance = train_config['spatio_phase']['threshold_distance']
    nearest_k = train_config['spatio_phase']['nearest_k']
    batch_size = train_config['batch_size']
    gene_factor_g = train_config['gene_phase']['gene_factor']
    spatio_factor_g = train_config['gene_phase']['spatio_factor']
    prior_factor_g = train_config['gene_phase']['prior_factor']
    gene_factor_s = train_config['spatio_phase']['gene_factor']
    spatio_factor_s = train_config['spatio_phase']['spatio_factor']
    prior_factor_s = train_config['spatio_phase']['prior_factor']
    renew_rounds = train_config['spatio_phase']['renew_rounds']
    partial_update = train_config['spatio_phase']['partial_update']
    equal_contribute = train_config['spatio_phase']['equal_contribute']
    accur_record_gene = train(model,
                              0.5,
                              gene_round,
                              data_loader,
                              batch_size,
                              update_gene = True,
                              update_spatio = False,
                              renew_neighbourhood = 0,
                              spatio_factor = spatio_factor_g,
                              prior_factor = prior_factor_g,
                              gene_factor = gene_factor_g,
                              report_per_rounds=1,
                              verbose = verbose,
                              stochastic_update = False,
                              nearest_k = nearest_k,
                              threshold_distance = threshold_distance)
    print("Train the spatio model while hold the gene model.")
    predict_gene,ll,_ = model.expectation(data_loader.xs,
                                  spatio_factor=0,
                                  gene_factor=1,
                                  prior_factor = 0)
    data_loader.renew_neighbourhood(predict_gene.transpose(),
                                    threshold_distance = threshold_distance,
                                    nearest_k = nearest_k,
                                    exclude_self = True)
    ### Initial spatio model with gene model.
    accur_record_spatio = train(model,
          0.5,
          spatial_round,
          data_loader,
          batch_size,
          spatio_factor = spatio_factor_g,
          gene_factor = gene_factor_g,
          prior_factor = prior_factor_g,
          update_spatio = True,
          update_gene = False,
          report_per_rounds=1,
          renew_per_rounds = 1, 
          renew_neighbourhood = 0,#No neighbourhood frequency renew during training.
          verbose = 1,
          nearest_k = nearest_k,
          threshold_distance = threshold_distance)
    
    ### Train the saptio+gene model.
    accur_record_both = train(model,
          0.5,
          both_round,
          data_loader,
          batch_size,
          spatio_factor = spatio_factor_s,
          gene_factor = gene_factor_s,
          prior_factor = prior_factor_s,
          update_spatio = True,
          update_gene = False,
          report_per_rounds=1,
          renew_per_rounds = 1, 
          renew_neighbourhood = renew_rounds,
          verbose = 1,
          nearest_k = nearest_k,
          threshold_distance = threshold_distance,
          partial_update = partial_update,
          equal_contrib = equal_contribute)
    return model,(accur_record_gene,accur_record_spatio,accur_record_both)

if __name__ == "__main__":
    pass
    ##Implement the command line entry.