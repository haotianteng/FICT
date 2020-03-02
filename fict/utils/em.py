#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:28:03 2020

@author: heavens
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
from joint_simulator import Simulator
from joint_simulator import Dataloader
from scipy.special import softmax
from sklearn.metrics.cluster import adjusted_rand_score

#np.random.seed(2020)
class EM():
    """
    General Expectation-Maximization class
    """
    def __init__(self, parameter):
        self.p = parameter
        
    def expectation(self,batch):
        #The expectation step for the EM algorithm implemented here.
        pass
    
    def maximization(self):
        #The update of the parameters of the maximization step for the EM algorithm
        #implemented here.
        pass
        
class MM(object):
    """
    MM class for Mixture Model.
    Input Args:
        observable: A observable dataset with N sample(first size).
        model: A model that has expectation and maximization method implemented.        
    """
    def __init__(self, observable, model):
        self.model = model
        
    def em_step(self, batch):
        self.class_assignment,hidden = self.model.expectation(batch)
        self.model.maximize(self.class_assignment,batch,hidden)
    
    def train(self, 
              training_step, 
              batch_size=None ):
        if batch_size is None:
            # Using all the dataset instead of batch
            shuffle = False
        else:
            shuffle = True
        for i in range(training_step):
            batch = self.observable.next(batch_size = batch_size,
                                         shuffle = shuffle)
            self.em_step(batch)

class fict_model(EM):
    def __init__(self,
                 gene_n,
                 class_n,
                 neighbour_n):
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
                     'neighbour_n':neighbour_n,
                     'g_mean':g_mean,
                     'g_cov':g_cov,
                     'mn_p':mn_p,
                     'prior':prior})
        
    @property
    def class_n(self):
        return self.p['class_n']
            
    @property
    def neighbour_n(self):
        return self.p['neighbour_n']
    
    def expectation(self,batch):
        gene_batch,neighbour_batch = batch
        self.Gs = []
        self.MNs = []
        for i in range(self.p['class_n']):
            self.Gs.append(multivariate_normal(self.p['g_mean'][i],self.p['g_cov'][i],allow_singular=True))
            self.MNs.append(multinomial(self.neighbour_n,self.p['mn_p'][i]))
        self.Prior = self.p['prior']
        batch_n = gene_batch.shape[0]
        assert neighbour_batch.shape[0] == batch_n
        posterior = np.zeros((self.class_n,batch_n))
        for i in range(self.class_n):
            lpx1z = self.Gs[i].logpdf(gene_batch)
            lpy1z = self.MNs[i].logpmf(neighbour_batch)
            lpz = np.log(self.Prior[i])
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
        temp_mn = np.matmul(posterior,neighbour_batch)
        mn_sum = np.sum(temp_mn,axis = 1,keepdims = True)
        mn_sum = mn_sum + pseudo
        self.p['mn_p'] = self._ema(self.p['mn_p'],
                                   temp_mn/mn_sum,
                                   decay = decay)
    
    def _ema(self,old_v,new_v,decay = 0.8):
        """The Exponential Moving Average update of a variable.
        """
        return decay*old_v + (1-decay)*new_v
    
    def get_neighbour_count(posterior,adjacency_matrix):
        #TODO Implemeted the neighbour count matrix form the posterior probability
        #and adjacency matrix.
        pass

if __name__ == "__main__":
    gene_n = 100
    cell_type = 10
    neighbour_n = 20
    em_round = 1000
#    decays = [0.8,0.9,0.99]
#    step_each_decay = 10 #Maximization step each EM round.
    cell_n = 10000
    batch_n = 400
    threshold_distance = 1
    m = fict_model(gene_n,cell_type,neighbour_n)
    sim = Simulator(cell_n,gene_n,cell_type,neighbour_n,threshold_distance)
    dataloader = Dataloader(sim)
    for i in range(em_round):
        batch_all = dataloader.next_batch(batch_n,shuffle = True)
        batch = (batch_all[0],batch_all[2])
        label = batch_all[1]
        posterior = m.expectation(batch)
        m.maximization(batch,posterior,decay = 0.9)
        predict = np.argmax(posterior,axis=0)
        Accuracy = adjusted_rand_score(predict,label)
        if i%10 == 0:
            print("%d Round Accuracy:%f"%(i,Accuracy))