#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 04:16:13 2020

@author: heavens
"""
import numpy as np
import pickle
from fict.fict_model import FICT_EM
from fict.utils.data_op import pca_reduce
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from matplotlib import pyplot as plt
from fict.utils.data_op import one_hot_vector
from mpl_toolkits.mplot3d import Axes3D
from fict.fict_input import RealDataLoader
from fict.fict_train import permute_accuracy

def mix_gene_profile(simulator, indexs,gene_proportion=1,cell_proportion = 0.7):
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
        mix_cells.append(mix_cell_index)
        sim_gene_expression[mix_cell_index,:] = np.random.multivariate_normal(mean = current_mix_mean,
                           cov = current_mix_cov)
    return sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells

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
        print(yerror)
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

with open("simulator.bin",'rb') as f:
    sim = pickle.load(f)
    
### mix the gene profile of 2nd and 3rd cell type and generate gene expression
sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells = mix_gene_profile(sim,[0,1])
reduced_d = 10
reduced_expression = pca_reduce(sim_gene_expression,dims = reduced_d)
class_n,gene_n = sim.g_mean.shape
plot_freq(sim_cell_neighbour,sim_cell_type,[0,1,2])
arti_posterior = one_hot_vector(sim_cell_type)
data_loader = RealDataLoader(reduced_expression,
                             sim.coor,
                             threshold_distance = 1,
                             num_class = class_n,
                             cell_labels = sim_cell_type)
plt.figure()
plt.scatter(sim.coor[:,0],sim.coor[:,1],c = sim_cell_type)
plt.title("Cell scatter plot.")

## Create a model and train using only gene expression.
print("Training with gene expression only.")
model_gene = FICT_EM(reduced_d,class_n)
em_epoches = 5
Accrs_gene = []
Accrs_gene_same = []
batch = data_loader.xs
model_gene.gaussain_initialize(batch[0])
for i in np.arange(em_epoches):
    posterior,ll = model_gene.expectation(batch,
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
    mask = np.logical_or(sim_cell_type==0,sim_cell_type==1)
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
mask = np.logical_or(predict==0,predict==1)
gene_p = np.copy(model_gene.p['g_mean'])

## Train spatio model 
print("Train a spatial model based on true nieghbourhood.")
model = model_gene
em_epoches = 3
Accrs_spatial = []
Accrs_spatial_same = []

#### Update neighbourhood count with true label, to get the best possible
#### accuracy of the spatio model
data_loader.renew_neighbourhood(arti_posterior)
batch = data_loader.xs
for i in np.arange(em_epoches):
    posterior,ll = model.expectation(batch,
                              spatio_factor=1.0,
                              gene_factor=0.0,
                              prior_factor = 0,
                              equal_contrib = True)
    predict = np.argmax(posterior,axis=0)
    model.maximization(batch,
                       posterior,
                       decay = 0.5,
                       update_gene_model = False,
                       update_spatio_model = True,
                       stochastic_update=False)
    mask = np.logical_or(sim_cell_type==0,sim_cell_type==1)
    partial_predict = predict[mask]
    partial_cell_type = sim_cell_type[mask]
    Accuracy = permute_accuracy(predict,sim_cell_type)[0]
    Accuracy_same = permute_accuracy(partial_predict,partial_cell_type)[0]
    Accrs_spatial.append(Accuracy)
    Accrs_spatial_same.append(Accuracy_same)
    print("Permute accuracy from true neighbourhood of mixd cell:%f"%(Accuracy_same))
    print("Permute accuracy for all:%f"%(Accuracy))
####

### Initialize using gene model
### Update with the prediction of gene model
print("Training with spatio and gene expression.")
Accrs_both = []
Accrs_both_same = []

batch = data_loader.xs
posterior,ll = model.expectation(batch,
                              spatio_factor=0,
                              gene_factor=1,
                              prior_factor = 0.0)
data_loader.renew_neighbourhood(posterior.transpose())
batch = data_loader.xs
model.maximization(batch,
                   posterior,
                   update_gene_model = False,
                   update_spatio_model = True,
                   stochastic_update=False)
icm_steps = 10
for i in np.arange(em_epoches):
    batch,_ = data_loader.next_batch(1000,shuffle= False)
    for j in np.arange(icm_steps):
        posterior,ll = model.expectation(batch,
                                  spatio_factor=0.5,
                                  gene_factor=0.5,
                                  prior_factor = 0,
                                  equal_contrib = True)
        data_loader.renew_neighbourhood(posterior.transpose())
    predict = np.argmax(posterior,axis=0)
    model.maximization(batch,
                       posterior,
                       decay = 0.5,
                       update_gene_model = False,
                       update_spatio_model = True,
                       stochastic_update=False)
    mask = np.logical_or(sim_cell_type==0,sim_cell_type==1)
    partial_predict = predict[mask]
    partial_cell_type = sim_cell_type[mask]
    Accuracy = permute_accuracy(predict,sim_cell_type)[0]
    Accuracy_same = permute_accuracy(partial_predict,partial_cell_type)[0]
    Accrs_both.append(Accuracy)
    Accrs_both_same.append(Accuracy_same)
    print("Permute accuracy of mixd cell:%f"%(Accuracy_same))
    print("Permute accuracy for all:%f"%(Accuracy))


### Begin the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nb_reduced = manifold.TSNE().fit_transform(sim_cell_neighbour)
ax.scatter(sim_cell_neighbour[:,0],sim_cell_neighbour[:,1],sim_cell_neighbour[:,2],c=sim_cell_type)
colors = ['red','green','blue']
figs,axs = plt.subplots(nrows = 2,ncols =2)
for i in np.arange(3):
    axs[0][0].plot(nb_reduced[sim_cell_type==i,0],nb_reduced[sim_cell_type==i,1],'.',c = colors[i],label = i)
    axs[0][0].legend()
    axs[0][0].set_title("True label")
for i in np.arange(10):
    posterior_spatio,ll = model.expectation(batch,
                                  spatio_factor=0.5,
                                  gene_factor=0.5,
                                  prior_factor = 1.0,
                                  equal_contrib = True)
    data_loader.renew_neighbourhood(posterior_spatio.transpose())
posterior_spatio,ll = model.expectation(batch,
                              spatio_factor=1,
                              gene_factor=0,
                              prior_factor = 0.0)
predict_spatio = np.argmax(posterior_spatio,axis=0)
print("Adjusted rand score of spatio model only %.3f"%(adjusted_rand_score(predict_spatio,sim_cell_type)))
print("Best accuracy of spatio model only %.3f"%(permute_accuracy(predict_spatio,sim_cell_type)[0]))
ax = axs[0][1]
scatter = ax.scatter(nb_reduced[:,0],nb_reduced[:,1],c = predict_spatio,s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by spatio model")

print("Adjusted rand score of gene model only %.3f"%(adjusted_rand_score(predict_gene,sim_cell_type)))
print("Best accuracy of gene model only %.3f"%(permute_accuracy(predict_gene,sim_cell_type)[0]))

ax = axs[1][0]
scatter = ax.scatter(nb_reduced[:,0],nb_reduced[:,1],c = predict_gene,s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by gene model")


posterior_sg,ll = model.expectation(batch,
                              spatio_factor=0.5,
                              gene_factor=0.5,
                              prior_factor = 0.0,
                              equal_contrib = True)
predict_sg = np.argmax(posterior_sg,axis=0)
print("Adjusted rand score of gene+spatio model %.3f"%(adjusted_rand_score(predict_sg,sim_cell_type)))
print("Best accuracy of gene+spatio model %.3f"%(permute_accuracy(predict_sg,sim_cell_type)[0]))

ax = axs[1][1]
scatter = ax.scatter(nb_reduced[:,0],nb_reduced[:,1],c = predict_sg,s = 10)
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend)
ax.set_title("Predict by gene+spatio model")

for factor in np.arange(0,1,0.1):
    posterior_sg,ll = model.expectation(batch,
                              spatio_factor=factor,
                              gene_factor=1-factor,
                              prior_factor = 0.0,
                              equal_contrib = True)
    predict_sg = np.argmax(posterior_sg,axis=0)
    print("Spatio factor:%.2f"%(factor))
    print("Best accuracy of gene+spatio model %.3f"%(permute_accuracy(predict_sg,sim_cell_type)[0]))
    print("Likelihood:%.2f"%(ll))