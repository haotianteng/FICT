#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:57:24 2020

@author: heavens
"""
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from importlib import reload
from fict.fict_model import FICT_EM
from fict.fict_input import RealDataLoader
from fict.fict_input import load_loader
from time import time

def train(model,
          decay,
          train_rounds,
          data_loader,
          batch_size,
          spatio_factor = 0.5, 
          gene_factor = 0.5,
          prior_factor = 1.0,
          update_spatio = True,
          update_gene = True,
          verbose = True,
          report_per_rounds = 10,
          renew_per_rounds = 10):
    accur_record = []
    for i in range(train_rounds):
        x_batch,y = data_loader.next_batch(batch_size,shuffle = True)
        posterior = model.expectation(x_batch,
                                   spatio_factor = spatio_factor,
                                   gene_factor = gene_factor,
                                   prior_factor = prior_factor)
        model.maximization(x_batch,posterior,decay = decay,update_spatio_model = False)
        predict = np.argmax(posterior,axis=0) 
        accuracy = adjusted_rand_score(predict,y)
        accur_record.append(accuracy)
#        if i%renew_rounds == 0:
#            posterior_all = m2.expectation(data_loader.xs,spatio_factor = spatio_factor,gene_factor = gene_factor)
#            data_loader.renew_neighbourhood(np.transpose(posterior_all))
        if i%report_per_rounds == 0 and verbose:
            print("%d Round Accuracy:%f"%(i,accuracy))
    return accur_record

if __name__ == "__main__":
    from fict.utils.data_op import tag2int
    from fict.utils.data_op import one_hot_vector
    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/df_test"
    real_loader = load_loader(data_f)
    n_c = 10
    m2 = FICT_EM(real_loader.gene_expression.shape[1],n_c)
    renew_rounds = 10
    batch_n = 5000
    gene_round = 100
    spatial_round = 100
    both_round = 50
    for threshold_distance in np.arange(10,200,10):
        int_label,tags = tag2int(real_loader.y)
        one_hot_label = one_hot_vector(int_label)
        real_loader.renew_neighbourhood(one_hot_label,
                                        threshold_distance = threshold_distance,
                                        exclude_self = True)
    #    print("Begin training using gene expression and spatio information.")
    #    accur_record_both = train(m2,
    #          0.5,
    #          both_round,
    #          real_loader,
    #          batch_n,
    #          spatio_factor = 0)
        print("Train the spatio model only.")
    #    accur_record_spatio = train(m2,
    #          0.5,
    #          spatial_round,
    #          real_loader,
    #          batch_n,
    #          spatio_factor = 2.0,
    #          gene_factor = 0,
    #          prior_factor = 0.0,
    #          update_spatio = True,
    #          update_gene = False)
        nb_count = real_loader.xs[1]
        y = real_loader.y
        for i,tag in enumerate(tags):
            mean_nb_count = np.mean(nb_count[y==tag],axis = 0)
            m2.p['mn_p'][i] = mean_nb_count/np.sum(mean_nb_count)
        posterior = m2.expectation(real_loader.xs,
                       spatio_factor = 1,
                       gene_factor = 0,
                       prior_factor = 0)
        predict = np.argmax(posterior,axis=0) 
        accuracy = adjusted_rand_score(predict,y)
        print("Accuracy with spatio information only and %.2f threshold distance %.4f"%(threshold_distance,accuracy))
