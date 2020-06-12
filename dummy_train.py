#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 04:16:13 2020

@author: heavens
"""
import numpy as np
import pickle
import seaborn as sns
from fict.fict_model import FICT_EM
from fict.utils.data_op import pca_reduce
from fict.utils.data_op import tag2int
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from matplotlib import pyplot as plt
from fict.utils.data_op import one_hot_vector
from mpl_toolkits.mplot3d import Axes3D
from fict.fict_input import RealDataLoader
from fict.fict_train import permute_accuracy
from fict.fict_train import train

def mix_gene_profile(simulator, indexs,gene_proportion=1,cell_proportion = 0.5):
    """mix the gene expression profile of cell types in indexs
    Args:
        simulator: A Simualator instance.
        indexs: The indexs of the cell type want to mixd to.
        proportion: The proportion of cells that being mixed.
    """
    mix_mean = np.mean(sim.g_mean[indexs,:],axis = 0)
    mix_cov = np.mean(sim.g_cov[indexs,:,:],axis = 0)
    perm = np.arange(sim.gene_n)
    np.random.shuffle(perm)
    mix_gene_idx = perm[:int(sim.gene_n*gene_proportion)]
    mix_cov_idx = tuple(np.meshgrid(mix_gene_idx,mix_gene_idx))
    sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None)
    mix_cells = []
    for i in indexs:
        current_mix_mean = np.copy(sim.g_mean[i])
        current_mix_mean[mix_gene_idx] = mix_mean[mix_gene_idx]
        current_mix_cov = np.copy(sim.g_cov[i])
        current_mix_cov[mix_cov_idx] = mix_cov[mix_cov_idx]
        perm = [x for x,c in enumerate(sim_cell_type) if c==i]
        np.random.shuffle(perm)
        mix_cell_index = perm[:int(len(perm)*cell_proportion)]
        mix_cells += mix_cell_index
        sim_gene_expression[mix_cell_index,:] = np.random.multivariate_normal(mean = current_mix_mean,
                           cov = current_mix_cov,
                           size = len(mix_cell_index))
    return sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,np.asarray(mix_cells)

def _plot_freq(neighbour,axes,color,cell_tag):
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

def plot_freq(nb_count,cell_label,plot_class):
    type_n = len(plot_class)
    fig,axs = plt.subplots()
    colors = ['green', 'blue','red']
    for i,cell_idx in enumerate(plot_class):
        freq_true,yerror = _plot_freq(nb_count[cell_label == cell_idx],
                                     axes = axs,
                                     color = colors[i],
                                     cell_tag = plot_class[i])
    nb_freqs = np.zeros((type_n,type_n))
    for i in np.arange(type_n):
        parital_nb = sim_cell_neighbour[sim_cell_type==i]
        freq = parital_nb/np.sum(parital_nb,axis = 1,keepdims = True)
        nb_freqs[i,:] = np.mean(freq,axis = 0)
    title_str = "Generated neighbourhood frequency of cell"+ ",".join([str(x) for x in plot_class])
    plt.title(title_str)
    plt.xlabel("Cell type")
    plt.ylabel("Frequency")
    plt.show()

def train_dummy(model,
          epoches,
          data_loader,
          gene_expression,
          nb_count):
    pass

with open("/home/heavens/CMU/FISH_Clustering/FICT/simulator.bin",'rb') as f:
    sim = pickle.load(f)
    
### mix the gene profile of 2nd and 3rd cell type and generate gene expression
mix_cell_t = [1,2]
sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells = mix_gene_profile(sim,mix_cell_t)
mask = np.zeros(len(sim_cell_type),dtype = np.bool)
mask[mix_cells] = True
reduced_d = 4
k_n = 5
reduced_expression,_ = pca_reduce(sim_gene_expression,dims = reduced_d)
class_n,gene_n = sim.g_mean.shape
plot_freq(sim_cell_neighbour,sim_cell_type,[0,1,2])
arti_posterior = one_hot_vector(sim_cell_type)[0]
int_type,tags = tag2int(sim_cell_type)
np.random.shuffle(arti_posterior)
data_loader = RealDataLoader(reduced_expression,
                             sim.coor,
                             threshold_distance = 1,
                             num_class = class_n,
                             cell_labels = sim_cell_type)
plt.figure()
plt.scatter(sim.coor[:,0],sim.coor[:,1],c = sim_cell_type)
plt.title("Cell scatter plot.")

## Create a model and train using only gene expression.
print("#####################################")
print("Training with gene expression only.")
model_gene = FICT_EM(reduced_d,class_n)
em_epoches = 5
Accrs_gene = []
Accrs_gene_same = []
thres_dist = 1
data_loader.renew_neighbourhood(arti_posterior,
                                threshold_distance=thres_dist)
batch = data_loader.xs
model_gene.gaussain_initialize(batch[0])
for i in np.arange(em_epoches):
    posterior,ll,_ = model_gene.expectation(batch,
                                       spatio_factor=0,
                                       gene_factor=1,
                                       prior_factor = 0.0)
    model_gene.maximization(batch,
                            posterior,
                            decay = 0.5,
                            update_gene_model = True,
                            update_spatio_model = False,
                            stochastic_update=False)
    predict = np.argmax(posterior,axis=0)
    partial_predict = predict[mask]
    partial_cell_type = sim_cell_type[mask]
    Accuracy = permute_accuracy(predict,sim_cell_type)[0]
    rand_score = adjusted_rand_score(predict,sim_cell_type)
    Accuracy_same_gene = permute_accuracy(partial_predict,partial_cell_type)[0]
    rand_score_same_gene = adjusted_rand_score(partial_predict,partial_cell_type)
    Accrs_gene.append(Accuracy)
    Accrs_gene_same.append(Accuracy_same_gene)
    print("Permutation accuracy of mixd cells:%f"%(Accuracy_same_gene))
    print("Permutation accuracy of all cells:%f"%(Accuracy))
predict_gene = predict
gene_p = np.copy(model_gene.p['g_mean'])
print("#####################################")
print("\n")


## Train spatio model 
print("#####################################")
print("Train a spatial model based on true nieghbourhood.")
model = model_gene
em_epoches = 10
Accrs_spatial = []
Accrs_spatial_same = []
#### Update neighbourhood count with true label, to get the best possible
#### accuracy of the spatio model
arti_posterior = one_hot_vector(sim_cell_type)[0]
data_loader.renew_neighbourhood(arti_posterior,threshold_distance=thres_dist)
batch = data_loader.xs
for i in np.arange(em_epoches):
    posterior,ll,_ = model.expectation(batch,
                              spatio_factor=1.0,
                              gene_factor=0.0,
                              prior_factor = 0,
                              equal_contrib = True)
    predict = np.argmax(posterior,axis=0)
    model.maximization(batch,
                       posterior,
                       decay = None,
                       update_gene_model = False,
                       update_spatio_model = True,
                       stochastic_update=False)
    partial_predict = predict[mask]
    partial_cell_type = sim_cell_type[mask]
    Accuracy = permute_accuracy(predict,sim_cell_type)[0]
    Accuracy_same = permute_accuracy(partial_predict,partial_cell_type)[0]
    Accrs_spatial.append(Accuracy)
    Accrs_spatial_same.append(Accuracy_same)
    print("Permute accuracy from true neighbourhood of mixd cell:%f"%(Accuracy_same))
    print("Permute accuracy for all:%f"%(Accuracy))
print("#####################################")
print("\n")
####
    

### Initialize using gene model
### Update with the prediction of gene model
print("#####################################")
print("Training with spatio and gene expression.")
Accrs_both = []
Accrs_both_same = []

batch = data_loader.xs
posterior,ll,_ = model.expectation(batch,
                              spatio_factor=0,
                              gene_factor=1,
                              prior_factor = 0.0)
data_loader.renew_neighbourhood(posterior.transpose(),
                                threshold_distance=thres_dist)
batch = data_loader.xs
model.maximization(batch,
                   posterior,
                   update_gene_model = False,
                   update_spatio_model = True,
                   stochastic_update=False)
icm_steps = 30
both_rounds = 10
for i in np.arange(both_rounds):
    batch,_ = data_loader.next_batch(1000,shuffle= False)
    for j in np.arange(icm_steps):
        posterior,ll,_ = model.expectation(batch,
                                  spatio_factor=1,
                                  gene_factor=1,
                                  prior_factor = 0,
                                  equal_contrib = True)
        data_loader.renew_neighbourhood(posterior.transpose(),
                                        partial_update = 0.1,
                                        threshold_distance=thres_dist)
    predict = np.argmax(posterior,axis=0)
    model.maximization(batch,
                       posterior,
                       decay = 0.5,
                       update_gene_model = False,
                       update_spatio_model = True,
                       stochastic_update=False)
    partial_predict = predict[mask]
    partial_cell_type = sim_cell_type[mask]
    Accuracy = permute_accuracy(predict,sim_cell_type)[0]
    Accuracy_same = permute_accuracy(partial_predict,partial_cell_type)[0]
    Accrs_both.append(Accuracy)
    Accrs_both_same.append(Accuracy_same)
    print("Permute accuracy of mixd cell:%f"%(Accuracy_same))
    print("Permute accuracy for all:%f"%(Accuracy))
    print("Likelihood %.2f"%(ll))
print("#####################################")
print("\n")

### Begin the plot
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nb_reduced = manifold.TSNE().fit_transform(sim_cell_neighbour)
color_map = np.asarray(['r','g','b'])
hit_map = np.asarray(['red','green'])

### True label plot
ax.scatter(sim_cell_neighbour[:,0],
           sim_cell_neighbour[:,1],
           sim_cell_neighbour[:,2],
           c=color_map[sim_cell_type])
colors = ['red','green','blue']
figs,axs = plt.subplots(nrows = 2,ncols =2)
figs2,axs2 = plt.subplots(nrows = 2,ncols = 2)
scatter = axs[0][0].scatter(nb_reduced[:,0],nb_reduced[:,1],c = sim_cell_type,s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
axs[0][0].set_title("True label")
scatter = axs2[0][0].scatter(nb_reduced[:,0],nb_reduced[:,1],c = sim_cell_type,s=10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
axs2[0][0].set_title("True label")
    
### Spatio model plot
for i in np.arange(10):
    posterior_spatio,ll,_ = model.expectation(batch,
                                  spatio_factor=1,
                                  gene_factor=1,
                                  prior_factor = 1.0,
                                  equal_contrib = True)
    data_loader.renew_neighbourhood(posterior_spatio.transpose(),
                                    threshold_distance=thres_dist)
posterior_spatio,ll,_ = model.expectation(batch,
                              spatio_factor=1,
                              gene_factor=0,
                              prior_factor = 0.0)
predict_spatio = np.argmax(posterior_spatio,axis=0)
print("Adjusted rand score of spatio model only %.3f"%(adjusted_rand_score(predict_spatio,sim_cell_type)))
perm_accur_spatio,perm_spatio = permute_accuracy(predict_spatio,sim_cell_type)
print("Best accuracy of spatio model only %.3f"%(perm_accur_spatio))
ax = axs[0][1]
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = color_map[predict_spatio],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by spatio model")
#Plot the hit
ax = axs2[0][1]
predict_spatio,_ = tag2int(predict_spatio)
hit_spatio = np.zeros(len(predict_spatio))
for i,p in enumerate(perm_spatio):
    hit_spatio = np.logical_or(hit_spatio,(predict_spatio==p)*(int_type==i))
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = hit_map[hit_spatio.astype(int)],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Hit by spatio model")

### Gene model plot
accur,perm_gene = permute_accuracy(predict_gene,sim_cell_type)
print("Adjusted rand score of gene model only %.3f"%(adjusted_rand_score(predict_gene,sim_cell_type)))
print("Best accuracy of gene model only %.3f"%(accur))
ax = axs[1][0]
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = color_map[predict_gene],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by gene model")
ax = axs2[1][0]
predict_gene,_ = tag2int(predict_gene)
hit_gene = np.zeros(len(predict_gene))
for i,p in enumerate(perm_gene):
    hit_gene = np.logical_or(hit_gene,(predict_gene==p)*(int_type==i))
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = hit_map[hit_gene.astype(int)],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Hit by gene model")


###Gene+spatio model plot
posterior_sg,ll,expectations = model.expectation(batch,
                              spatio_factor=1,
                              gene_factor=1,
                              prior_factor = 0.0,
                              equal_contrib = True)
predict_sg = np.argmax(posterior_sg,axis=0)
print("Adjusted rand score of gene+spatio model %.3f"%(adjusted_rand_score(predict_sg,sim_cell_type)))
accr_sg,perm_sg = permute_accuracy(predict_sg,sim_cell_type)
print("Best accuracy of gene+spatio model %.3f"%(accr_sg))
ax = axs[1][1]
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = color_map[predict_sg],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by gene+spatio model")
ax = axs2[1][1]
predict_sg,_ = tag2int(predict_sg)
hit_sg = np.zeros(len(predict_sg))
for i,p in enumerate(perm_sg):
    hit_sg = np.logical_or(hit_sg,(predict_sg==p)*(int_type==i))
scatter = ax.scatter(nb_reduced[:,0],
                     nb_reduced[:,1],
                     c = hit_map[hit_sg.astype(int)],
                     s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Hit by gene+spatio model")

###Check different factor setting.
accurs = []
spatio_factors = []
lls = []
for factor in np.arange(0,1,0.01):
    posterior_sg,ll,_ = model.expectation(batch,
                              spatio_factor=factor,
                              gene_factor=1,
                              prior_factor = 0.0,
                              equal_contrib = False)
    predict_sg = np.argmax(posterior_sg,axis=0)
    spatio_factors.append(factor)
    accurs.append(permute_accuracy(predict_sg,sim_cell_type)[0])
    lls.append(ll)
idx = np.argmax(accurs)
plt.figure()
plt.plot(spatio_factors,accurs)
plt.xlabel("The spatio factor(gene factor is 1)")
plt.ylabel("The permute accuracy.")
plt.title("The permute accuracy across different spatio factor.")
print("Best accuracy of gene+spatio model %.3f, with spatio factor %.3f"%(accurs[idx],spatio_factors[idx]))