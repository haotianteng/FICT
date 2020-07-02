#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:16:54 2020

@author: haotian teng
"""
import os 
import pickle
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from fict.utils.data_op import tsne_reduce
from matplotlib.patches import Rectangle
def load(f):
    with open(f,'rb') as f:
        obj = pickle.load(f)
    return obj

def heatmap(cv,ax,xticks= None,yticks = None,title = '',highlight_cells = None):
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
    if highlight_cells is not None:
        for coor in highlight_cells:
            ax.add_patch(Rectangle((coor[0]-0.5,coor[1]-0.5), 1, 1, fill=False, edgecolor='blue', lw=3))
    ax.set_title(title)
    return ax

def cluster_visualization(posterior,loader,ax,mode = 'gene',mask = None):
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
    if mask is not None:
        locs = locs[mask,:]
        predict = predict[mask]
    ax.scatter(locs[:,0],
               locs[:,1],
               c=predict,
               cmap = colors,
               s = 5)
    return ax,locs,predict

def compare_visual(e_gene,e_spatio,loaders,i,j,field = None,mode = 'coordinate'):
    figs,axs = plt.subplots(nrows = 2,ncols = 2)
    figs.set_size_inches(24,h=12)
    loader = loaders[i]
    if field is not None:
        mask = loader.field==field
        mask = mask[:,0]
    else:
        mask = None
        
    cluster_visualization(e_gene[i,j,0],loader,axs[0][0],mode = mode,mask = mask)
    cluster_visualization(e_gene[i,j,1],loader,axs[0][1],mode = mode,mask = mask)
    cluster_visualization(e_spatio[i,j,0],loader,axs[1][0],mode = mode,mask = mask)
    cluster_visualization(e_spatio[i,j,1],loader,axs[1][1],mode = mode,mask = mask)
    axs[0][0].set_title("Gene model %d on dataset %d"%(i,i))
    axs[0][1].set_title("Gene model %d on dataset %d"%(j,i))
    axs[1][0].set_title("Spatio model %d on dataset %d"%(i,i))
    axs[1][1].set_title("Spatio model %d on dataset %d"%(j,i))
    return figs,axs

def confusion_matrix(e1,e2,field_mask = None):
    class_n = e1.shape[0]
    predict1 = np.argmax(e1,axis = 0)
    predict2 = np.argmax(e2,axis = 0)
    if field_mask is not None:
        predict1 = predict1[field_mask]
        predict2 = predict2[field_mask]
    cf_matrix = np.zeros((class_n,class_n))
    for i in np.arange(class_n):
        for j in np.arange(class_n):
            cf_matrix[i,j] = np.sum(np.logical_and(predict1==i,predict2==j))
    return cf_matrix

def greedy_match(confusion):
    confusion = np.copy(confusion)
    class_n = confusion.shape[0]
    perm = np.arange(class_n)
    overlap = 0
    for i in np.arange(class_n):
        ind = np.unravel_index(np.argmax(confusion, axis=None), confusion.shape)
        overlap += confusion[ind]
        perm[ind[0]] = ind[1]
        confusion[ind[0],:] = -1
        confusion[:,ind[1]] = -1
    return perm,overlap
    
def cluster_plot(e1,e2,loader,cell1,cell2,field = None,title = ['','']):
    predict1 = np.argmax(e1,axis = 0)
    predict2 = np.argmax(e2,axis = 0)
    if type(cell1) is np.int_ or type(cell1) is int:
        cell1 = np.asarray([cell1])
    if type(cell2) is np.int_ or type(cell2) is int:
        cell2 = np.asarray([cell2])
    assert len(cell1)==len(cell2)
    mask1 = predict1 == cell1[0]
    mask2 = predict2 == cell2[0]
    for idx,c in enumerate(cell1):
        mask1 = np.logical_or(mask1,predict1 == c)
        mask2 = np.logical_or(mask2,predict2 == cell2[idx])
    if field is not None:
        field_mask = loader.field==field
        field_mask = field_mask[:,0]
        mask1 = np.logical_and(mask1,field_mask)
        mask2 = np.logical_and(mask2,field_mask)
    locs = loader.coordinate
    locs1 = locs[mask1,:]
    locs2 = locs[mask2,:]
    figs,axs = plt.subplots(nrows = 1,ncols = 2)
    figs.set_size_inches(24,h=8)
    predict1 = predict1[mask1]
    predict2 = predict2[mask2]
    p1 = np.copy(predict1)
    p2 = np.copy(predict2)
    colors = cm.get_cmap('Set2', len(cell1))
    for idx,c in enumerate(cell1):
        p1[predict1==c] = idx
    for idx,c in enumerate(cell2):
        p2[predict2==c] = idx
    axs[0].scatter(locs1[:,0],
                   locs1[:,1],
                   c=p1,
                   cmap = colors,
                   s=5)
    axs[0].set_title(title[0])
    axs[1].scatter(locs2[:,0],
                   locs2[:,1],
                   c=p2,
                   cmap = colors,
                   s=5)
    axs[1].set_title(title[1])
    return axs


plt.close('all')
base_f = "/home/heavens/lanec1/data/MERFISH_data/cv_result_multi/animal1-4"
model_f = os.path.join(base_f,'trained_models.bn')
cv_f = os.path.join(base_f,'cv_result.bn')
loader_f = os.path.join(base_f,'loaders.bn')
field = 0.01
animal1 = 0
animal2 = 1
models = load(model_f)
e_gene,e_spatio,cv_gene,cv_spatio = load(cv_f)
loaders = load(loader_f)
figs,axs = compare_visual(e_gene,e_spatio,loaders,animal1,animal2,field = field,mode = 'coordinate')
field_mask = loaders[animal1].field==field
field_mask = field_mask[:,0]
cm_gene = confusion_matrix(e_gene[animal1,animal2,0],
                      e_gene[animal1,animal2,1],
                      field_mask)
cm_sp = confusion_matrix(e_spatio[animal1,animal2,0],
                         e_spatio[animal1,animal2,1],
                         field_mask)
cm_gs = confusion_matrix(e_gene[animal1,animal2,0],
                         e_spatio[animal1,animal2,0],
                         field_mask)
fig,axs = plt.subplots(ncols = 3, nrows = 1)
perm_gene,overlap_gene = greedy_match(cm_gene)
perm_spatio,overlap_spatio = greedy_match(cm_sp)
perm_gs,overlap_gs = greedy_match(cm_gs)

heatmap(cm_gene,
        axs[0],
        title = 'Gene confusion matrix.',
        highlight_cells = list(zip(perm_gene,np.arange(len(perm_gene)))))
heatmap(cm_sp,
        axs[1],
        title = "Spatio confusion matrix.",
        highlight_cells = list(zip(perm_spatio,np.arange(len(perm_spatio)))))
heatmap(cm_gs,
        axs[2],
        title = "Gene-Spatio confusion matrix.",
        highlight_cells = list(zip(perm_gs,np.arange(len(perm_gs)))))

cluster_plot(e_gene[animal1,animal2,0],
             e_gene[animal1,animal2,1],
             loaders[animal1],
             np.arange(len(perm_gene)),
             perm_gene,
             field = field,
             title = ['Gene model %d on animal %d'%(animal1,animal1),'Gene model %d on animal %d'%(animal2,animal1)])


cluster_plot(e_spatio[animal1,animal2,0],
             e_spatio[animal1,animal2,1],
             loaders[animal1],
             np.arange(len(perm_spatio)),
             perm_spatio,
             field = field,
             title = ['Spatio model %d on animal %d'%(animal1,animal1),'Spatio model %d on animal %d'%(animal2,animal1)])


cluster_plot(e_gene[animal1,animal2,0],
             e_spatio[animal1,animal2,0],
             loaders[animal1],
             np.arange(len(perm_gs)),
             perm_gs,
             field = field,
             title = ['Gene model %d on animal %d bregma %.2f'%(animal1,animal1,field),'Spatio model %d on animal %d bregma %.2f'%(animal1,animal1,field)])