#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:37:49 2020

@author: heavens
"""

import numpy as np
import pandas as pd
from fict.utils import joint_simulator
from fict.utils.joint_simulator import Simulator
from fict.utils.joint_simulator import SimDataLoader
from fict.utils.joint_simulator import get_gene_prior
from fict.utils.joint_simulator import get_nf_prior
from fict.utils.opt import valid_neighbourhood_frequency
from sklearn.metrics.cluster import adjusted_rand_score
from fict.fict_input import RealDataLoader
from matplotlib import pyplot as plt
from matplotlib import cm
from importlib import reload
from sklearn.decomposition import PCA
from sklearn import manifold
from fict.utils import joint_simulator
import seaborn as sns
import pickle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from fict.fict_model import FICT_EM


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

def make_error_boxes(ax, xdata, ydata, xerror = None, yerror = None, facecolor='r',
                     edgecolor='None', alpha=0.5):
    if (xerror is None) and (yerror is None):
        raise ValueError("Either X error or Y error should be given.")
    # Loop over data points; create box from errors at each point
    if xerror is None:
        xerror = np.zeros(yerror.shape)
        xerror[0,:] = xerror[0,:] - 0.5
        xerror[1,:] = xerror[1,:] + 0.5
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)
    
    return None

def plot_freq(neighbour,axes,color,cell_tag):
    sample_n = neighbour.shape[1]
    neighbour = neighbour/np.sum(neighbour,axis = 1,keepdims = True)
    std = np.std(neighbour, axis = 0)/np.sqrt(sample_n)
    mean = np.mean(neighbour, axis = 0)
    x = np.arange(sample_n)
    yerror = np.asarray([-std,std])
#    make_error_boxes(axes, x, mean, yerror = yerror)
    patches = axes.boxplot(neighbour,
                        vert=True,  # vertical box alignment
                        patch_artist=True,
                        notch=True,
                        usermedians = mean) # fill with color
    for patch in patches['boxes']:
        patch.set_facecolor(color)
        patch.set_color(color)
        patch.set_alpha(0.5)
    for patch in patches['fliers']:
        patch.set_markeredgecolor(color)
        patch.set_color(color)
    for patch in patches['whiskers']:
        patch.set_color(color)
    for patch in patches['caps']:
        patch.set_color(color)
    axes.errorbar(x+1,mean,color = color,label = cell_tag)
    return mean,yerror

if __name__ == "__main__":
    type_n = 3
    test_cell = [0,1,2]
    freq = [[0.2,0.1,0.7],[0.2,0.6,0.2],[0.3,0.2,0.5]]
    nb_n0  = 20
    em_round =20
    nb_n_noise = 5
    
    with open("simulator.bin",'rb') as f:
        sim = pickle.load(f)
    sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None)
    mask = np.zeros(sim_cell_type.shape)
    for cell_idx in test_cell:
        mask = np.logical_or(mask,sim_cell_type == cell_idx)
    partial_cell_type = sim_cell_type[mask]
    partial_neighbour = sim_cell_neighbour[mask]
    partial_gene_expression = sim_gene_expression[mask]
    fig,axs = plt.subplots()
    colors = ['green', 'blue','red']
    for i,cell_idx in enumerate(test_cell):
        freq_true,yerror = plot_freq(partial_neighbour[partial_cell_type == cell_idx],
                                     axes = axs,
                                     color = colors[i],
                                     cell_tag = test_cell[i])
        print(yerror)
    nb_freqs = np.zeros((type_n,type_n))
    for i in np.arange(type_n):
        parital_nb = sim_cell_neighbour[sim_cell_type==i]
        freq = parital_nb/np.sum(parital_nb,axis = 1,keepdims = True)
        nb_freqs[i,:] = np.mean(freq,axis = 0)
    plt.title("Generated neighbourhood frequency of cell %d and %d."%(test_cell[0],test_cell[1]))
    plt.xlabel("Cell type")
    plt.ylabel("Frequency")
    plt.show()
    
    dummy_cell_neighbour = np.zeros(sim_cell_neighbour.shape)
    masks = []
    for i in np.arange(type_n):
        masks.append(partial_cell_type==test_cell[i])
    
    ### EM on variable neighbour count number
    print("Training on variable neighbour count number")
    for i in np.arange(type_n):
        dummy_cell_neighbour[masks[i]] = np.random.multinomial(nb_n0,freq[i],size = sum(masks[i]))
    for i in np.arange(5):
        cell_index = np.arange(len(masks[0]))
        for j in np.arange(type_n):
            update_index = np.random.choice(cell_index[masks[j]],replace = False,size = 100)
            nb_n = nb_n0 + np.random.choice(np.arange(-nb_n_noise,nb_n_noise+1))
            dummy_cell_neighbour[update_index] = np.random.multinomial(nb_n,freq[j],size = len(update_index))
    
    nb_freqs = np.asarray(nb_freqs)
    model = FICT_EM(partial_gene_expression.shape[1],sim_cell_neighbour.shape[1])
    Accrs = []
    for i in range(em_round):
        batch = (sim_gene_expression,dummy_cell_neighbour)
        posterior = model.expectation(batch,spatio_factor=1.0,gene_factor=0,prior_factor=0)
        model.maximization(batch,posterior,decay = 0.5,update_gene_model = False,stochastic_update=False)
        predict = np.argmax(posterior,axis=0)
        partial_predict = predict[mask]
        Accuracy = adjusted_rand_score(partial_predict,partial_cell_type)
        Accrs.append(Accuracy)
        print("Accuracy:%f"%(Accuracy))
    
    fig,axs = plt.subplots()
    colors = ['green', 'blue','red']
    for i,cell_idx in enumerate(test_cell):
        freq_true,yerror = plot_freq(dummy_cell_neighbour[partial_cell_type == cell_idx],
                                     axes = axs,
                                     color = colors[i],
                                     cell_tag = test_cell[i]+1)
    plt.title("Generated neighbourhood frequency of cell %d and %d."%(test_cell[0],test_cell[1]))
    plt.xlabel("Cell type")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(np.arange(len(Accrs)),Accrs)
    plt.title("Training Accuracy")
    plt.xlabel("Train epoches.")
    plt.ylabel("Accuracy")
    
    ### EM on variable neighbour count number
    repeat = 40
    final_accr = []
    for i in np.arange(repeat):
        type_n = 2
        test_cell = [0,1]
        freq = [[0.2,0.8],[0.4,0.6]]
        nb_n0  = 20
        em_round =20
        nb_n_noise = 5
        sample_n = 1000
        
        mask = np.zeros(sim_cell_type.shape)
        for cell_idx in test_cell:
            mask = np.logical_or(mask,sim_cell_type == cell_idx)
        partial_cell_type = sim_cell_type[mask]
        partial_neighbour = sim_cell_neighbour[mask]
        partial_gene_expression = sim_gene_expression[mask]
        sample_n = partial_neighbour.shape[0]
        dummy_cell_neighbour = np.zeros((sample_n,type_n))
        masks = []
        for i in np.arange(type_n):
            masks.append(partial_cell_type==test_cell[i])
        print("Training on variable neighbour count number")
        for i in np.arange(type_n):
            dummy_cell_neighbour[masks[i]] = np.random.multinomial(nb_n0,freq[i],size = sum(masks[i]))
        for i in np.arange(5):
            cell_index = np.arange(len(masks[0]))
            for j in np.arange(type_n):
                update_index = np.random.choice(cell_index[masks[j]],replace = False,size = 100)
                nb_n = nb_n0 + np.random.choice(np.arange(-nb_n_noise,nb_n_noise+1))
                dummy_cell_neighbour[update_index] = np.random.multinomial(nb_n,freq[j],size = len(update_index))
        
        nb_freqs = np.asarray(nb_freqs)
        model = FICT_EM(partial_gene_expression.shape[1],dummy_cell_neighbour.shape[1])
        Accrs = []
        for i in range(em_round):
            batch = (partial_gene_expression,dummy_cell_neighbour)
            posterior = model.expectation(batch,spatio_factor=1.0,gene_factor=0,prior_factor=0)
            model.maximization(batch,posterior,decay = 0.5,update_gene_model = False,stochastic_update=False)
            predict = np.argmax(posterior,axis=0)
            partial_predict = predict
            Accuracy = adjusted_rand_score(partial_predict,partial_cell_type)
            Accrs.append(Accuracy)
        print("Final Accuracy:%f"%(Accuracy))
        final_accr.append(max(Accrs))
    
        ### Plot the training curve
        fig,axs = plt.subplots()
        colors = ['green', 'blue','red']
        for i,cell_idx in enumerate(test_cell):
            freq_true,yerror = plot_freq(dummy_cell_neighbour[partial_cell_type == cell_idx],
                                         axes = axs,
                                         color = colors[i],
                                         cell_tag = test_cell[i]+1)
        plt.title("Generated neighbourhood frequency of cell %d and %d."%(test_cell[0],test_cell[1]))
        plt.xlabel("Cell type")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
    plt.figure()
    plt.plot(np.arange(len(Accrs)),Accrs)
    plt.title("Training Accuracy")
    plt.xlabel("Train epoches.")
    plt.ylabel("Accuracy")
    
    final_accr = np.asarray(final_accr)
    np.mean(final_accr)
    np.std(final_accr)
    
    
