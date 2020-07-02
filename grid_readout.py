#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:32:50 2020

@author: haotian teng
"""
import numpy as np
import os
import pickle
from cross_validation import compare_visual
from matplotlib import pyplot as plt
### Read out the grid search for reduced dimensions, threshold distance and renew rounds.
#reduced_dim = np.arange(5,20,5)
#thres_dist = np.arange(20,60,10)
#renew_rounds = np.arange(10,30,10)
#root_f = "/home/heavens/lanec1/data/MERFISH_data/grid_test"
#average_cv = np.zeros((len(reduced_dim),len(thres_dist),len(renew_rounds)))
#for i,rd in enumerate(reduced_dim):
#    for j,td in enumerate(thres_dist):
#        for k,rr in enumerate(renew_rounds):
#            f_name = "r%dt%drenew%d"%(rd,td,rr)
#            folder = os.path.join(root_f,f_name)
#            with open(os.path.join(folder,'cv_result.bn'),'rb') as f:
#                cv = pickle.load(f)
#                e_gene,e_spatio,cv_gene,cv_spatio = cv
#                cv_n = cv_spatio.shape[0]
#                score = (np.sum(cv_spatio) - cv_n)/(cv_n**2-cv_n)
#                print("Dim %d Thres %d renew %d score:%.3f"%(rd,td,rr,score))
#                average_cv[i,j,k] = score

### Read out the grid search for spatio factors.
plt.close('all')
root_f = "/home/heavens/lanec1/data/MERFISH_data/grid_search_spatio_factor"
sp_factor = [1,0.5,0.2,0.1,0.05,0.01,0.001]
cv_sp = np.zeros(7)
cv_sp_gene = np.zeros(7)
for i,sp in enumerate(sp_factor):
    f_name = "r%dt%drenew%dsf"%(10,40,15)+str(sp)
    print(f_name)
    folder = os.path.join(root_f,f_name)
    with open(os.path.join(folder,'cv_result.bn'),'rb') as f:
        cv = pickle.load(f)
        e_gene,e_spatio,cv_gene,cv_spatio = cv
        cv_n = cv_spatio.shape[0]
        score = (np.sum(cv_spatio) - cv_n)/(cv_n**2-cv_n)
        score_gene = (np.sum(cv_gene) - cv_n)/(cv_n**2-cv_n)
        print("Spatio factor %f score:%.3f"%(sp,score))
        print("Spatio factor %f gene model score:%.3f"%(sp,score_gene))
        cv_sp[i] = score
        cv_sp_gene[i] = score_gene
    with open(os.path.join(folder,'loaders.bn'),'rb') as f:
        loaders = pickle.load(f)
    figs,axs = compare_visual(e_gene,e_spatio,loaders,4,7)
    figs.suptitle("Spatio factor %.3f"%(sp))