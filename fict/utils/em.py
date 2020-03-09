#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:28:03 2020

@author: heavens
"""
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

