#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:44:40 2020

@author: heavens
"""

from scipy.stats import multinomial
import numpy as np

class multinomial_wrapper(object):
    """A wrapper for the scipy multinomial distrbution allow a dynmic assigned n.
    """
    def __init__(self,p):
        self.pdf = multinomial
        self.p = p
    def pmf(self,count_batch):
        batch_shape = count_batch.shape
        if len(batch_shape) == 1:
            n = np.sum(count_batch)
            f = self.pdf(n=n, p = self.p)
            return f.pmf(count_batch)
        results = []
        for count in count_batch:
            n = np.sum(count)
            f = self.pdf(n=n,p=self.p)
            results.append(f.pmf(count))
        return np.asarray(results)
    
    def logpmf(self,count_batch):
        batch_shape = count_batch.shape
        if len(batch_shape) == 1:
            n = np.sum(count_batch)
            f = self.pdf(n=n, p = self.p)
            return f.logpmf(count_batch)
        results = []
        for count in count_batch:
            n = np.sum(count)
            f = self.pdf(n=n,p=self.p)
            results.append(f.logpmf(count))
        return np.asarray(results)