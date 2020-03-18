#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:28:03 2020

@author: heavens
"""
#np.random.seed(2020)
import numpy as np
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
    def _ema(self,old_v,new_v,decay = 0.8):
        """The Exponential Moving Average update of a variable.
        """
        return decay*old_v + (1-decay)*new_v
    
    def _entropic_descent(self,old_v,new_v,step_size = 0.1):
        """The Implementation of Entropic Descent for proability variable
        http://www.princeton.edu/~yc5/ele522_optimization/lectures/mirror_descent.pdf
        """
        gradient = new_v - old_v
        result = old_v * np.exp(step_size*gradient)
        return self._normalize(result)
    
    def _rescaling_gradient(self,old_v,new_v,inv_covariance_matrix,step_size = 0.1):
        """Update the variable according to covariance matrix
        """
        gradient = new_v - old_v
        update = np.matmul(inv_covariance_matrix,gradient)
        return old_v + step_size * update
    
    def _normalize(self,p,axis = 1):
        """Normalize the given probability distribution p along axis
        """
        return p/np.sum(p,axis = axis,keepdims = True)
    
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

