#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:41:29 2020

@author: heavens
"""
from fict.utils.joint_simulator import Simulator
from fict.utils.joint_simulator import SimDataLoader
from fict.utils.joint_simulator import get_gene_prior
from fict.utils.joint_simulator import get_nf_prior
from fict.utils.opt import valid_neighbourhood_frequency
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from fict.fict_input import RealDataLoader
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt

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


### Hyper parameter setting
print("Setting hyper parameter")
sample_n = 1000 #Number of samples
n_g = 100 #Number of genes
n_c = 3 #Number of cell type
density = 20 #The number of neighbour for each cells.
threshold_distance = 1 # The threshold distance of neighbourhood.
gene_col = np.arange(9,164)
coor_col = [5,6]
header = 1
data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/aau5324_Moffitt_Table-S7.xlsx"

### Data preprocessing
print("Reading data from %s"%(data_f))
data = pd.read_excel(data_f,header = header)
cell_types = data['Cell_class']
data = data[cell_types!= 'Ambiguous']
cell_types = data['Cell_class']
gene_expression = data.iloc[:,gene_col]
type_tags = np.unique(cell_types)
coordinates = data.iloc[:,coor_col]

### Choose only the n_c type cells
print("Choose the subdataset of %d cell types"%(n_c))
if len(type_tags)<n_c:
    raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
mask = np.asarray([False]*len(cell_types))
for tag in type_tags[:n_c]:
    mask = np.logical_or(mask,cell_types==tag)
gene_expression = gene_expression[mask]
cell_types = np.asarray(cell_types[mask])
coordinates = np.asarray(coordinates[mask])

### Generate prior from the given dataset.
gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
neighbour_freq_prior,tags,type_count = get_nf_prior(coordinates,cell_types)
type_prior = type_count/np.sum(type_count)
#target_freq = neighbour_freq_prior + 2.0*np.random.random(neighbour_freq_prior.shape)
target_freq = np.asarray([[7.2,3,2],[2,6,3],[2.5,1,7]])
target_freq = (target_freq)/np.sum(target_freq,axis=1,keepdims=True)
result = valid_neighbourhood_frequency(target_freq)
target_freq = result[0]

### Assign cell types by neighbourhood assignment
sim = Simulator(sample_n,n_g,n_c,density)
sim.gen_parameters(gene_mean_prior = gene_mean[:,:n_g])
sim.gen_coordinate(density = density)
#sim.assign_cell_type(target_neighbourhood_frequency=target_freq, 
#                     method = "assign-neighbour",
#                     max_iter = 30000)
#plt.scatter(sim.coor[:,0],sim.coor[:,1], c = sim.cell_type_assignment,s = 20)
#plt.title("Cell type assignment after assign_neighbour")
#plt.xlabel("X")
#plt.ylabel("Y")

### Assign cell types by Gibbs sampling and load
alternate_n = 20
sim.assign_cell_type(target_neighbourhood_frequency=target_freq, 
                     method = "Gibbs-sampling",
                     max_iter = 30,
                     use_exist_assignment = False)
sim.assign_cell_type(target_neighbourhood_frequency=target_freq, 
                     method = "Metropolis-swap",
                     max_iter = 30000,
                     use_exist_assignment = True,
                     annealing = False)
plt.scatter(sim.coor[:,0],sim.coor[:,1], c = sim.cell_type_assignment,s = 20)
plt.title("Cell type assignment after assign_neighbour")
plt.xlabel("X")
plt.ylabel("Y")


sim._get_neighbourhood_frequency()
nb_freq = sim.neighbour_frequency

### Save the simulator to the file
with open("simulator.bin",'wb+') as f:
    pickle.dump(sim,f)

### Show the neighbourhood frequency of generated dataset
with open("simulator.bin",'rb') as f:
    sim = pickle.load(f)
sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None)
mask = np.zeros(sim_cell_type.shape)
test_cell = np.arange(n_c)
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
nb_freqs = np.zeros((n_c,n_c))
for i in np.arange(n_c):
    parital_nb = sim_cell_neighbour[sim_cell_type==i]
    freq = parital_nb/np.sum(parital_nb,axis = 1,keepdims = True)
    nb_freqs[i,:] = np.mean(freq,axis = 0)
plt.title("Generated neighbourhood frequency of cell %d %d and %d."%(test_cell[0],test_cell[1],test_cell[2]))
plt.xlabel("Cell type")
plt.ylabel("Frequency")
plt.show()
print("Target neighbourhood frequency:")
print(target_freq)
print("Generated neighbourhood frequency:")
print(nb_freqs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sim_cell_neighbour[:,0],sim_cell_neighbour[:,1],sim_cell_neighbour[:,2],c=sim_cell_type)
nb_reduced = manifold.TSNE().fit_transform(sim_cell_neighbour)
colors = ['red','green','blue']
plt.figure()
for i in np.arange(3):
    plt.plot(nb_reduced[sim_cell_type==i,0],nb_reduced[sim_cell_type==i,1],'.',c = colors[i])