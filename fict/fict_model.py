#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 04:05:14 2020

@author: heavens
"""

import numpy as np
from scipy.stats import multivariate_normal
from fict.utils.random_generator import dirichlet_wrapper
from fict.utils.joint_simulator import Simulator
from fict.utils.joint_simulator import SimDataLoader
from fict.utils.joint_simulator import get_gene_prior
from fict.utils.joint_simulator import get_nf_prior
from fict.utils.opt import valid_neighbourhood_frequency
from scipy.special import softmax
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
from fict.utils.em import EM
from fict.fict_input import RealDataLoader

class FICT_EM(EM):
    def __init__(self,
                 gene_n,
                 class_n):
        g_mean = np.random.rand(class_n,gene_n)
        tol = 1e-7
        g_cov = []
        for i in range(class_n):
            while True:
                #This is to ensure the generated matrix is invertible.
                temp = np.random.rand(gene_n,gene_n)
                cov = np.dot(temp,np.transpose(temp))
                try:
                    singularity = np.abs(np.matmul(cov,np.linalg.inv(cov)) - np.eye(gene_n))
                    if np.all(singularity[:]<tol):
                        g_cov.append(cov)
                        break
                except np.linalg.LinAlgError:
                    pass
        g_cov = np.asarray(g_cov)
        mn_p = np.random.rand(class_n,class_n)
        mn_p = mn_p/np.sum(mn_p,axis = 1,keepdims = True)
        temp = np.random.rand(class_n)
        prior = temp/sum(temp)
        EM.__init__(self,parameter = {'gene_n':gene_n,
                     'class_n':class_n,
                     'g_mean':g_mean,
                     'g_cov':g_cov,
                     'mn_p':mn_p,
                     'prior':prior})
        
    @property
    def class_n(self):
        return self.p['class_n']
            
    def expectation(self,batch,no_spatio = False):
        gene_batch,neighbour_batch = batch
        self.Gs = []
        self.MNs = []
        for i in range(self.p['class_n']):
            self.Gs.append(multivariate_normal(self.p['g_mean'][i],self.p['g_cov'][i],allow_singular=True))
            self.MNs.append(dirichlet_wrapper(self.p['mn_p'][i]))
        self.Prior = self.p['prior']
        batch_n = gene_batch.shape[0]
        assert neighbour_batch.shape[0] == batch_n
        posterior = np.zeros((self.class_n,batch_n))
        for i in range(self.class_n):
            lpx1z = self.Gs[i].logpdf(gene_batch)
            lpy1z = self.MNs[i].logpmf(neighbour_batch)
            lpz = np.log(self.Prior[i])
            if no_spatio:
                posterior[i,:] = lpx1z + lpz
            else:
                posterior[i,:] = lpx1z + lpy1z + lpz
        return softmax(posterior,axis = 0)
    
    def maximization(self,
                     batch,posterior,
                     decay = 0.9,
                     pseudo = 1e-5):
        """Maximization step for the EM algorithm
        Input Args:
            batch: A tuple contain (gene_batch, nieghbour_batch)
                gene_batch [cell_n,gene_n]:The input batch of the gene expression data.
                neighbour_batch [cell_n,class_n]:The batch contain the count of neighbourhood.
            posterior [class_n,cell_n]: The posterior probability from the expectation step.
        """
        gene_batch,neighbour_batch = batch
        batch_n = gene_batch.shape[0]
        assert neighbour_batch.shape[0] == batch_n
        post_sum = np.sum(posterior,axis = 1,keepdims = True)
        post_sum = post_sum + pseudo
        self.p['g_mean'] = self._ema(self.p['g_mean'], 
                                     np.matmul(posterior,gene_batch)/post_sum,
                                     decay = decay)
        for i in range(self.class_n):
            batch_norm = gene_batch - self.p['g_mean'][i]
            cov = np.matmul(posterior[i]*np.transpose(batch_norm),batch_norm)
            self.p['g_cov'][i] = self._ema(self.p['g_cov'][i],
                                           cov/post_sum[i],
                                           decay = decay)
        self.p['prior'] = self._ema(self.p['prior'],
                                    post_sum[:,0]/batch_n,
                                    decay = decay)
        self.p['prior'] = self._renormalize(self.p['prior'],axis = 0)
        temp_mn = np.matmul(posterior,neighbour_batch)
        mn_sum = np.sum(temp_mn,axis = 1,keepdims = True)
        mn_sum = mn_sum + pseudo
        self.p['mn_p'] = self._ema(self.p['mn_p'],
                                   temp_mn/mn_sum,
                                   decay = decay)
        self.p['mn_p'] = self._renormalize(self.p['mn_p'])
    def _ema(self,old_v,new_v,decay = 0.8):
        """The Exponential Moving Average update of a variable.
        """
        return decay*old_v + (1-decay)*new_v
    def _renormalize(self,p,axis = 1):
        """Renormalize the given probability distribution p along axis
        """
        return p/np.sum(p,axis = axis,keepdims = True)
    def get_neighbour_count(posterior,adjacency_matrix):
        """Obtain the count of neighbourhood from the given posterior matrix and
        the adjacency matrix.
        Args:
            posterior: A N-by-M matrix, where N is the cell type number, and M
                is the batch number.
            adjacency_matrix: A M-by-M matrix where M is the batch number.
        Return:
            neighbour_count: A M-by-N matrix where M is the batch number and N
                is the number of cell type.
        """
        return np.transpose(np.matmul(posterior,adjacency_matrix))

if __name__ == "__main__":
    ## Script test and running example.
    ### Hyper parameter setting
#    sample_n = 1000 #Number of samples
#    n_g = 100 #Number of genes
#    n_c = 10 #Number of cell type
#    density = 20 #The average number of neighbour for each cells.
#    threshold_distance = 1 # The threshold distance of neighbourhood.
#    gene_col = np.arange(9,164)
#    coor_col = [5,6]
#    header = 1
#    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/aau5324_Moffitt_Table-S7.xlsx"
#    
#    ### Data preprocessing
#    data = pd.read_excel(data_f,header = header)
#    gene_expression = data.iloc[:,gene_col]
#    cell_types = data['Cell_class']
#    type_tags = np.unique(cell_types)
#    coordinates = data.iloc[:,coor_col]
#    
#    ### Choose only the n_c type cells
#    if len(type_tags)<n_c:
#        raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
#    mask = np.asarray([False]*len(cell_types))
#    for tag in type_tags[:n_c]:
#        mask = np.logical_or(mask,cell_types==tag)
#    gene_expression = gene_expression[mask]
#    cell_types = np.asarray(cell_types[mask])
#    coordinates = np.asarray(coordinates[mask])
# 
#    ## Training a classifier from the simulation dataset and validation
#    ### Generate prior from the given dataset.
#    gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
#    neighbour_freq_prior,tags,type_count = get_nf_prior(coordinates,cell_types)
#    type_prior = type_count/np.sum(type_count)
#    target_freq = (neighbour_freq_prior+0.1)/np.sum(neighbour_freq_prior+0.1,axis=1,keepdims=True)
#    result = valid_neighbourhood_frequency(target_freq)
#    target_freq = result[0]
#    
#    ### Generate simulation dataset and load
#    sim = Simulator(sample_n,n_g,n_c,density)
#    sim.gen_parameters(gene_mean_prior = gene_mean[:,:n_g])
#    sim.gen_coordinate(density = density)
#    sim.assign_cell_type(target_neighbourhood_frequency=target_freq, method = "assign-neighbour")
#    gene_expression,cell_type,cell_neighbour = sim.gen_expression()
#    df = SimDataloader(gene_expression,
#                    cell_neighbour,
#                    for_eval = False,
#                    cell_type_assignment = cell_type)
#    gene_n = n_g
#    cell_type = n_c
#    neighbour_n = 20
#    em_round = 500
##    decays = [0.8,0.9,0.99]
##    step_each_decay = 10 #Maximization step each EM round.
#    cell_n = sample_n
#    batch_n = 400
#    threshold_distance = 1
#    m = FICT_EM(gene_n,cell_type)
#    for i in range(em_round):
#        batch_all = df.next_batch(batch_n,shuffle = True)
#        batch = (batch_all[0],batch_all[2])
#        label = batch_all[1]
#        posterior = m.expectation(batch)
#        m.maximization(batch,posterior,decay = 0.9)
#        predict = np.argmax(posterior,axis=0)
#        Accuracy = adjusted_rand_score(predict,label)
#        if i%10 == 0:
#            print("%d Round Accuracy:%f"%(i,Accuracy))
    
    ## Training the classifier from the real dataset and validation
    m2 = FICT_EM(gene_expression.shape[1],n_c)
    renew_rounds = 10
    batch_n = 2000
    em_round = 5000
    threshold_distance = 20
    gene_expression = np.asarray(gene_expression)
    init_prob = np.ones((gene_expression.shape[0],n_c))*1.0/n_c
    real_df = RealDataLoader(gene_expression,
                             coordinates,
                             threshold_distance = threshold_distance,
                             cell_type_probability = init_prob,
                             cell_labels = cell_types,
                             for_eval = False)
    for i in range(em_round):
        x_batch,y = real_df.next_batch(batch_n,shuffle = True)
        posterior = m2.expectation(x_batch)
        m2.maximization(x_batch,posterior,decay = 0.5)
        predict = np.argmax(posterior,axis=0)
        Accuracy = adjusted_rand_score(predict,y)
        if i%renew_rounds == 0:
            posterior_all = m2.expectation(real_df.xs)
            real_df.renew_neighbourhood(np.transpose(posterior_all))
        if i%10 == 0:
            print("%d Round Accuracy:%f"%(i,Accuracy))
