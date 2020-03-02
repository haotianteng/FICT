#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:03:44 2020

@author: heavens
"""

from scipy.optimize import minimize
import numpy as np
from numpy.random import uniform

def symmetric_loss(D,M,l=0):
    loss = np.linalg.norm(M*D - np.transpose(M*D)) + l*np.abs(np.sum(D)-1)
    return loss

def normalize(M,axis = 1):
    M = M/np.sum(M,axis = axis,keepdims = True)
    return M

def constrain(D):
    return np.sum(D)-1

def reconstruct(init_M,Diagonal):
    count = init_M*Diagonal
    count = (count + np.transpose(count))/2
    return count

def valid_neighbourhood_frequency(init_frequency,norm_axis = 1):
    init_frequency = normalize(init_frequency,axis = norm_axis)
    n = init_frequency.shape[0]
    D = [1/n]*n
    if norm_axis == 1:
        init_frequency = np.transpose(init_frequency)
    eq_cons = {'type': 'eq',
               'fun':constrain}
    cons = [eq_cons]
    for i in range(n):
        cons.append({'type':'ineq',
                 'fun':lambda x:x[i]})
    opt = minimize(symmetric_loss,
                   D,
                   method = 'trust-constr',
                   args = init_frequency,
                   constraints = cons,
                   tol = 1e-5)
    new_count = reconstruct(init_frequency,opt.x)
    if norm_axis == 1:
        init_frequency = np.transpose(init_frequency)
    return normalize(new_count,axis = norm_axis),opt
    
if __name__ == "__main__":
#    count_matrix = np.asarray([[2,4,8],[4,3,7],[8,6,5]])
#    test_matrix = normalize(count_matrix)
#    D0 = [0.5,0.2,0.3]
#    result = minimize(symmetric_loss,
#                      D0,
#                      args = test_matrix,
#                      method='BFGS',
#                      tol = 1e-3)
#    print(result.fun)
#    print("Count matrix:")
#    print(test_matrix*result.x)
#    print("Given prob matrix:")
#    print(test_matrix)
#    print("Recorvered matrix:")
#    print(reconstruct(test_matrix,result.x))
    text_f = "/home/heavens/CMU/FISH_Clustering/FICT/out.txt"
    test_matrix2 = np.random.rand(10,10)
    print("Given prob matrix:")
    print(normalize(test_matrix2))
    print("Optimized nearest valid matrix:")
    result,opt = valid_neighbourhood_frequency(test_matrix2)
    print(normalize(result))
    with open(text_f,'w') as f:
        for i in range(10):
            f.write(' '.join([str(x) for x in result[i,:]]))
            f.write("\n")
            f.write(' '.join([str(x) for x in normalize(test_matrix2)[i,:]]))
            f.write("\n")
            f.write("%"*10)
            f.write("\n")