#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:44:40 2020

@author: heavens
"""

from scipy.stats import multinomial
import numpy as np

def multinomial_probability(count,p,return_log = True):
    prob = multinomial(np.sum(count),p)
    if return_log:
        return prob.logpmf(count)
    else:
        return prob.pmf(count)