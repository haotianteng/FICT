#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 04:05:14 2020

@author: heavens
"""

import numpy as np
from scipy.stats import multivariate_normal
from fict.utils.random_generator import continuous_multinomial
from scipy.special import softmax
from fict.utils.em import EM
from sklearn.decomposition import PCA
from sklearn import manifold
from time import time

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

#def select_gene(X,n_genes = 25):
#    """Select the significant genes by the coefficient of variation.
#    Args:
#        X: A N-by-M gene expression matrix, where N is the number of sample,
#            M is the number of genes.
#    """
#    mean = np.mean(X,axis = )
#    cv = 


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
    
    @property
    def gene_n(self):
        return self.p['gene_n']
            
    def expectation(self,batch,spatio_factor = 0.2,gene_factor=1.0,prior_factor = 1.0):
        gene_batch,neighbour_batch = batch
        self.Gs = []
        self.MNs = []
        for i in range(self.p['class_n']):
            self.Gs.append(multivariate_normal(self.p['g_mean'][i],self.p['g_cov'][i],allow_singular=True))
            self.MNs.append(continuous_multinomial(self.p['mn_p'][i]))
        self.Prior = self.p['prior']
        batch_n = gene_batch.shape[0]
        assert neighbour_batch.shape[0] == batch_n
        posterior = np.zeros((self.class_n,batch_n))
        for i in range(self.class_n):
            lpx1z = self.Gs[i].logpdf(gene_batch)
            lpy1z = self.MNs[i].logpmf(neighbour_batch)
            lpx1z = lpx1z / np.mean(lpx1z) * np.mean(lpy1z)
            lpz = np.log(self.Prior[i])
            posterior[i,:] = gene_factor*lpx1z + spatio_factor*lpy1z + prior_factor*lpz
        
        return softmax(posterior,axis = 0)
    
    def maximization(self,
                     batch,posterior,
                     decay = 0.9,
                     pseudo = 1e-5,
                     update_spatio_model = True,
                     update_gene_model = True,
                     stochastic_update = False):
        """Maximization step for the EM algorithm
        Input Args:
            batch: A tuple contain (gene_batch, nieghbour_batch)
                gene_batch [batch_n,gene_n]:The input batch of the gene expression data.
                neighbour_batch [cell_n,class_n]:The batch contain the count of neighbourhood.
            posterior [class_n,cell_n]: The posterior probability from the expectation step.
            update_saptio_model: Boolean variable indicate if we want to update
                the parameters of the multinomial spatial model.
            update_gene_model: Boolean variable indicate if we want to update
                the parameters of the gene expression model.
        """
        gene_batch,neighbour_batch = batch
        batch_n = gene_batch.shape[0]
        assert neighbour_batch.shape[0] == batch_n
        post_sum = np.sum(posterior,axis = 1,keepdims = True)
        post_sum = post_sum + pseudo
        if update_gene_model:
            mean_estimate = np.matmul(posterior,gene_batch)/post_sum
            for i in range(self.class_n):
                if stochastic_update:
                    self.p['g_mean'][i] = self._rescaling_gradient(self.p['g_mean'][i], 
                                                 mean_estimate[i],
                                                 np.linalg.inv(self.p['g_cov'][i]),
                                                 step_size = 1-decay)
                else:
                    self.p['g_mean'][i] = mean_estimate[i]
                
            for i in range(self.class_n):
                batch_norm = gene_batch - self.p['g_mean'][i]
                cov = np.matmul(posterior[i]*np.transpose(batch_norm),batch_norm)
                if stochastic_update:
                    self.p['g_cov'][i] = self._ema(self.p['g_cov'][i],
                                                   cov/post_sum[i],
                                                   decay = decay)
                else:
                    self.p['g_cov'][i] = cov/post_sum[i]
        if stochastic_update:
            new_prior = self._entropic_descent(np.reshape(self.p['prior'],(1,self.class_n)),
                                        np.reshape(post_sum[:,0]/batch_n,(1,self.class_n)),
                                        step_size = 1-decay)
            self.p['prior'] = np.reshape(new_prior,self.class_n)
        else:
            self.p['prior'] = (post_sum[:,0]+1/self.class_n)/batch_n
        if update_spatio_model:
            temp_mn = np.matmul(posterior,neighbour_batch)
            mn_sum = np.sum(temp_mn,axis = 1,keepdims = True)
            mn_sum = mn_sum + pseudo*self.class_n
            if stochastic_update:
                self.p['mn_p'] = self._entropic_descent(self.p['mn_p'],
                                          temp_mn/mn_sum,
                                          step_size = 1 - decay)
            else:
                self.p['mn_p'] = (temp_mn+pseudo)/mn_sum

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

