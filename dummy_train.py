#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 04:16:13 2020

@author: heavens
"""
import numpy as np
import pickle
import seaborn as sns
import json
from fict.fict_model import FICT_EM
from fict.utils.data_op import pca_reduce
from fict.utils.data_op import embedding_reduce
from fict.utils.data_op import save_smfish
from fict.utils.data_op import save_loader
from fict.utils import embedding as emb
from fict.utils.data_op import tag2int
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from matplotlib import pyplot as plt
from fict.utils.data_op import one_hot_vector
from mpl_toolkits.mplot3d import Axes3D
from fict.fict_input import RealDataLoader
from fict.fict_train import permute_accuracy
from fict.fict_train import train
from gect.gect_train_embedding import train_wrapper
import sys
import os
import argparse
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
plt.rcParams["font.size"] = "25"
def mix_gene_profile(simulator, indexs,gene_proportion=1,cell_proportion = 0.7,seed = None):
    """mix the gene expression profile of cell types in indexs
    Args:
        simulator: A Simualator instance.
        indexs: The indexs of the cell type want to mixd to.
        gene_proportion: The proportion of genes that being mixed.
        cell_proportion: The proportion of cells that being mixed.
        seed: The seed used to generate result
    """
    mix_mean = np.mean(sim.g_mean[indexs,:],axis = 0)
    mix_cov = np.mean(sim.g_cov[indexs,:,:],axis = 0)
    perm = np.arange(sim.gene_n)
    np.random.shuffle(perm)
    mix_gene_idx = perm[:int(sim.gene_n*gene_proportion)]
    mix_cov_idx = tuple(np.meshgrid(mix_gene_idx,mix_gene_idx))
    sim_gene_expression,sim_cell_type,sim_cell_neighbour = sim.gen_expression(drop_rate = None,seed = seed)
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

def accuracy_with_perm(predict,label,perm):
    n = len(predict)
    accur = np.sum([(predict == p) * (label == i) for i,p in enumerate(perm)])
    return accur/n

def main(sim_data,base_f):
    sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells = sim_data
    mask = np.zeros(len(sim_cell_type),dtype = np.bool)
    mask[mix_cells] = True
    reduced_d = 10
    k_n = 5
    ### train a embedding model from the simulated gene expression
    print("Begin training the embedding model.")
    np.savez(os.path.join(base_f,'sim_gene.npz'),
             feature = sim_gene_expression,
             labels = sim_cell_type)
    class Args:
        pass
    args = Args()
    args.train_data = os.path.join(base_f,'sim_gene.npz')
    args.eval_data = os.path.join(base_f,'sim_gene.npz')
    args.log_dir = base_f
    args.model_name = "simulate_embedding"
    args.embedding_size = reduced_d
    args.batch_size = 1000
    args.step_rate=4e-3
    args.drop_out = 0.9
    args.epoches = 10
    args.retrain = False
    args.device = None
    train_wrapper(args)
    embedding_file = os.path.join(base_f,'simulate_embedding/')
    embedding = emb.load_embedding(embedding_file)
    
    ### Dimensional reduction of simulated gene expression using PCA or embedding
    class_n,gene_n = sim.g_mean.shape
    plot_freq(sim_cell_neighbour,sim_cell_type,[0,1,2])
    arti_posterior = one_hot_vector(sim_cell_type)[0]
    int_type,tags = tag2int(sim_cell_type)
    np.random.shuffle(arti_posterior)
    data_loader = RealDataLoader(sim_gene_expression,
                                 sim.coor,
                                 threshold_distance = 1,
                                 num_class = class_n,
                                 cell_labels = sim_cell_type,
                                 gene_list = np.arange(sim_gene_expression.shape[1]))
    data_loader.dim_reduce(method = "Embedding",embedding = embedding)
    data_folder = os.path.join(base_f,'data/')
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    save_smfish(data_loader,data_folder+str(run_i),is_labeled = True)
    save_loader(data_loader,data_folder+str(run_i))
    fict_folder = os.path.join(base_f,'FICT_result/')
    if not os.path.isdir(fict_folder):
        os.mkdir(fict_folder)
    result_f = fict_folder+str(run_i)
    if not os.path.isdir(result_f):
        os.mkdir(result_f)
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
        Accuracy,perm = permute_accuracy(predict,sim_cell_type)
        rand_score = adjusted_rand_score(predict,sim_cell_type)
        Accuracy_same_gene = accuracy_with_perm(partial_predict,
                                                partial_cell_type,
                                                perm)
        rand_score_same_gene = adjusted_rand_score(partial_predict,partial_cell_type)
        Accrs_gene.append(Accuracy)
        Accrs_gene_same.append(Accuracy_same_gene)
        print("Permutation accuracy of mixd cells:%f"%(Accuracy_same_gene))
        print("Permutation accuracy of all cells:%f"%(Accuracy))
    predict_gene = predict
    gene_p = np.copy(model_gene.p['g_mean'])
    print("#####################################")
    print("\n")
    with open(os.path.join(result_f,"gene_model.bn"),'wb+') as f:
        pickle.dump(model_gene,f)
    
    ## Train a spatio model with true neighbourhood
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
        Accuracy,perm = permute_accuracy(predict,sim_cell_type)
        Accuracy_same = accuracy_with_perm(partial_predict,
                                           partial_cell_type,
                                           perm)
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
                                      spatio_factor=1.2,
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
        Accuracy,perm = permute_accuracy(predict,sim_cell_type)
        Accuracy_same = accuracy_with_perm(partial_predict,
                                           partial_cell_type,
                                           perm)
        Accrs_both.append(Accuracy)
        Accrs_both_same.append(Accuracy_same)
        print("Permute accuracy of mixd cell:%f"%(Accuracy_same))
        print("Permute accuracy for all:%f"%(Accuracy))
        print("Likelihood %.2f"%(ll))
    print("#####################################")
    print("\n")
    with open(os.path.join(result_f,"sg_model.bn"),'wb+') as f:
        pickle.dump(model,f)
    ### Begin the plot
    plt.close('all')
    fig = plt.figure(figsize = (20,10))
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
    figs,axs = plt.subplots(nrows = 2,ncols =2,figsize = (20,10))
    figs2,axs2 = plt.subplots(nrows = 2,ncols = 2,figsize=(20,10))
    ax = axs[0][0]
    scatter = axs[0][0].scatter(nb_reduced[:,0],
                                nb_reduced[:,1],
                                c = color_map[sim_cell_type],s = 10)
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
    axs[0][0].set_title("True label")
    scatter = axs2[0][0].scatter(nb_reduced[:,0],nb_reduced[:,1],c = sim_cell_type,s=10)
    ax = axs2[0][0]
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
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
    ari_spatio = adjusted_rand_score(predict_spatio,sim_cell_type)
    print("Adjusted rand score of spatio model only %.3f"%(ari_spatio))
    perm_accur_spatio,perm_spatio = permute_accuracy(predict_spatio,sim_cell_type)
    print("Best accuracy of spatio model only %.3f"%(perm_accur_spatio))
    ax = axs[0][1]
    scatter = ax.scatter(nb_reduced[:,0],
                         nb_reduced[:,1],
                         c = color_map[predict_spatio],
                         s = 10)
#    legend = ax.legend(*scatter.legend_elements(),
#                       loc="lower left", 
#                       title="Classes")
#    ax.add_artist(legend)
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
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", 
#                        title="Classes")
#    ax.add_artist(legend)
    ax.set_title("Hit by spatio model")
    
    ### Gene model plot
    accur,perm_gene = permute_accuracy(predict_gene,sim_cell_type)
    ari_gene = adjusted_rand_score(predict_gene,sim_cell_type)
    print("Adjusted rand score of gene model only %.3f"%(ari_gene))
    print("Best accuracy of gene model only %.3f"%(accur))
    ax = axs[1][0]
    scatter = ax.scatter(nb_reduced[:,0],
                         nb_reduced[:,1],
                         c = color_map[predict_gene],
                         s = 10)
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
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
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
    ax.set_title("Hit by gene model")
    
    
    ###Gene+spatio model plot
    posterior_sg,ll,expectations = model.expectation(batch,
                                  spatio_factor=1,
                                  gene_factor=1,
                                  prior_factor = 0.0,
                                  equal_contrib = True)
    predict_sg = np.argmax(posterior_sg,axis=0)
    ari_sg = adjusted_rand_score(predict_sg,sim_cell_type)
    print("Adjusted rand score of gene+spatio model %.3f"%(ari_sg))
    accr_sg,perm_sg = permute_accuracy(predict_sg,sim_cell_type)
    print("Best accuracy of gene+spatio model %.3f"%(accr_sg))
    ax = axs[1][1]
    scatter = ax.scatter(nb_reduced[:,0],
                         nb_reduced[:,1],
                         c = color_map[predict_sg],
                         s = 10)
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
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
#    legend = ax.legend(*scatter.legend_elements(),
#                        loc="lower left", title="Classes")
#    ax.add_artist(legend)
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
    return (ari_gene,ari_spatio,ari_sg),(accur,perm_accur_spatio,accr_sg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dummy_train',
                                     description='Train on simuulation data.')
    parser.add_argument('-p', '--prefix', required = True,
                        help="The prefix of the input dataset.")
    parser.add_argument('-n',default = 50,
                        help="The resolution factor of Leiden clustering.")
    args = parser.parse_args(sys.argv[1:])
    RUN_TIME = args.n
    aris_gene = []
    aris_spatio = []
    aris_sg = []
    accs_gene = []
    accs_spatio = []
    accs_sg = []
    base_f = args.prefix
    print("Begin simulation %d times."%(RUN_TIME))
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    with open(os.path.join(base_f,'simulator.bin'),'rb') as f:
        sim = pickle.load(f)
        
    ### mix the gene profile of 2nd and 3rd cell type and generate gene expression
    mix_cell_t = [1,2]
    seed_list = [random.randrange(2**32 - 1) for _ in np.arange(RUN_TIME)]
    for run_i in np.arange(RUN_TIME):
        mix = mix_gene_profile(sim,mix_cell_t,seed = seed_list[run_i])
        sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells=mix
        print(sim_gene_expression.shape)
        sim_data = (sim_gene_expression,sim_cell_type,sim_cell_neighbour,mix_mean,mix_cov,mix_cells)
        print("Start the %d simulation"%(run_i))
        aris,accrs = main(sim_data,base_f)
        aris_gene.append(aris[0])
        aris_spatio.append(aris[1])
        aris_sg.append(aris[2])
        accs_gene.append(accrs[0])
        accs_spatio.append(accrs[1])
        accs_sg.append(accrs[2])
    record = {}
    record['aris_gene'] = aris_gene
    record['aris_spatio'] = aris_spatio
    record['aris_sg'] = aris_sg
    record['accs_gene'] = accs_gene
    record['accs_spatio'] = accs_spatio
    record['accs_sg'] = accs_sg
    df = pd.DataFrame(record, columns = list(record.keys()))
    with open(os.path.join(base_f,'FICT_result/record.json') , 'w+') as f:
        json.dump(record,f)
    df_long = pd.melt(df,value_vars = ['accs_gene','accs_sg'],
                      var_name = "Model",
                      value_name = "Accuracy")
    sns.set(style="whitegrid")
    plt.figure()
    ax = sns.boxplot(data = df_long,x = "Model",y="Accuracy")
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = df_long['Accuracy'].max() + 0.01, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    p_val = ttest_ind(accs_gene,accs_sg,equal_var = False).pvalue
    p_val = "{:.2e}".format(p_val)
    plt.text((x1+x2)*.5, y+h, "p=%s"%(p_val), ha='center', va='bottom', color=col)
    plt.show()