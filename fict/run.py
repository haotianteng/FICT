#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:34:50 2021

@author: haotian teng
"""
import pickle
import os
import sys
import argparse
import numpy as np
import pandas as pd
from fict.fict_model import FICT_EM
from fict.utils.data_op import save_smfish
from fict.utils.data_op import save_loader
from fict.utils import embedding as emb
from fict.utils.data_op import tag2int
from fict.utils.data_op import one_hot_vector
from fict.fict_train import alternative_train
from sklearn.model_selection import train_test_split
from gect.gect_train_embedding import train_wrapper
from fict.fict_input import RealDataLoader

TRAIN_CONFIG = {'gene_phase':{},'spatio_phase':{}}
TRAIN_CONFIG['gene_round'] = 20
TRAIN_CONFIG['spatio_round'] = 5
TRAIN_CONFIG['both_round'] = 10
TRAIN_CONFIG['verbose'] = 1
TRAIN_CONFIG['gene_phase'] = {'gene_factor':1.0,
                              'spatio_factor':0.0,
                              'prior_factor':0.0}
TRAIN_CONFIG['spatio_phase'] = {'gene_factor':1.0,
                                'spatio_factor':1.0,
                                'prior_factor':0.0,
                                'nearest_k':None,
                                'threshold_distance':1.0,
                                'renew_rounds':10,
                                'partial_update':1.0,
                                'equal_contribute':False}

def load_train(data_loader,num_class = None):
    if data_loader.y:
        int_y,tags = tag2int(data_loader.y)
        data_loader.y = int_y
    if num_class is None:
        one_hot_label,tags = one_hot_vector(int_y)
        data_loader.renew_neighbourhood(one_hot_label,
                                        nearest_k = TRAIN_CONFIG['spatio_phase']['nearest_k'],
                                        threshold_distance = TRAIN_CONFIG['spatio_phase']['threshold_distance'],
                                        update_adj = True)
        num_class = len(tags)
    else:
        arti_label = np.random.randint(low = 0, 
                                       high = num_class,
                                       size = data_loader.sample_n)
        one_hot_label,tags = one_hot_vector(arti_label)
        data_loader.renew_neighbourhood(one_hot_label,
                                        nearest_k = TRAIN_CONFIG['spatio_phase']['nearest_k'],
                                        threshold_distance = TRAIN_CONFIG['spatio_phase']['threshold_distance'],
                                        update_adj = True)
    num_gene = data_loader.xs[0].shape[1]
    model = FICT_EM(num_gene,
                    num_class)
    TRAIN_CONFIG['batch_size'] = data_loader.xs[0].shape[0]
    alternative_train(data_loader,
                      model,
                      train_config = TRAIN_CONFIG)
    return model

def train_embedding(gene_expression, reduced_d,result_f):
    ### train a embedding model from the simulated gene expression
    print("Begin training the embedding model.")
    gene_train,gene_test = train_test_split(
            gene_expression,test_size=0.2)
    np.savez(os.path.join(result_f,'gene_all.npz'),
             feature = gene_expression,
             labels = None)
    np.savez(os.path.join(result_f,'gene_train.npz'),
             feature = gene_train,
             labels = None)
    np.savez(os.path.join(result_f,'gene_test.npz'),
             feature = gene_test,
             labels = None)
    class Args:
        pass
    args = Args()
    args.train_data = os.path.join(result_f,'gene_all.npz')#Use all the samples to train the embeddings.
    args.eval_data = os.path.join(result_f,'gene_test.npz')#Use a smaller size to evaluation to save time.
    args.log_dir = result_f
    args.model_name = "simulate_embedding"
    args.embedding_size = reduced_d
    args.batch_size = gene_expression.shape[0]
    args.step_rate=4e-3
    args.drop_out = 0.9
    args.epoches = 300
    args.threads = 5
    args.retrain = False
    args.device = None
    fig_collection = {}
    train_wrapper(args)
    embedding_file = os.path.join(result_f,'simulate_embedding/')
    embedding = emb.load_embedding(embedding_file)
    return embedding

def save_model(model,output,model_name = "sg_model_best.bn"):
    with open(os.path.join(output,"sg_model_best.bn"),'wb+') as f:
        pickle.dump(model,f)

def run(args):
    ## Load data using data loader
    try:
        with open(args.prefix,'rb') as f:    
            data_loader = pickle.load(f)
            ge = data_loader.gene_expression
            coor = data_loader.coordinate
    except:
        try:
            ge = pd.read_csv(args.prefix + '.expression',
                             index_col = 0, 
                             delimiter = ' ',
                             header = None)
            coor = pd.read_csv(args.prefix + '.coordinates',
                               index_col = 0,
                               delimiter = ' ',
                               header = None) 
        except:
            print("Can't find data in the given location, either a data loader\
                  or .exrpession and .coordinates files can be found.")
    embedding = train_embedding(ge,args.hidden,args.output)
    data_loader = RealDataLoader(ge,
                                 coor,
                                 threshold_distance = 1,
                                 num_class = args.n_type,
                                 cell_labels = None,
                                 gene_list = np.arange(ge.shape[1]),
                                 for_eval = True)
    model = load_train(data_loader,num_class=args.n_type)
    data_loader.dim_reduce(method = "Embedding",embedding = embedding)
    repeat = args.restart if args.restart else 1
    best_model = None
    mll = None
    lls = []
    for i in np.arange(repeat):
        model = load_train(data_loader,num_class = args.n_type)
        ###Gene+spatio model plot
        posterior_sg,_,_ = model.expectation(data_loader.xs,
                                             gene_factor = 1,
                                             spatio_factor = 0,
                                             prior_factor = 0)
        data_loader.renew_neighbourhood(posterior_sg.T,
                                        nearest_k =None,
                                        threshold_distance = 1)
        batch = data_loader.xs
        for k in np.arange(30):
            posterior_sg,_,_ = model.expectation(batch,
                                                 gene_factor = 1,
                                                 spatio_factor = 1,
                                                 prior_factor = 0,
                                                 equal_contrib = False)
            data_loader.renew_neighbourhood(posterior_sg.T,
                                            nearest_k =None,
                                            threshold_distance = 1,
                                            partial_update = 0.1)
            batch = data_loader.xs
        posterior_sg,_,ps = model.expectation(batch,
                                              gene_factor = 1,
                                              spatio_factor = 1,
                                              prior_factor = 0,
                                              equal_contrib = False)
        ll = sum([np.sum(x) for x in ps])
        lls.append(ll)
        if not mll or ll>mll:
            best_model = model
            mll = ll
            save_model(best_model,args.output)
            predict_sg = np.argmax(posterior_sg,axis=0)
            np.savetxt(os.path.join(args.output,'cluster_result.csv'),predict_sg.astype(int))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FICT-SAMPLE',
                                     description='Train on simuulation data.')
    parser.add_argument('-p', '--prefix', required = True,
                        help="The prefix of the input dataset, for simulation data, it's the folder contains simulator.bin file.")
    parser.add_argument('-o', '--output', required = True,
                        help="The output folder of the model.")
    parser.add_argument('-c', '--config', default = None,
                        help="The configure file, if None then use default configuration.")
    parser.add_argument('--n_type',default = 3, type = int,
                        help="Number of cell types.")
    parser.add_argument('--hidden',default = 10,type = int,
                        help="Hidden size of the denoise auto-encoder.")
    parser.add_argument('--restart', default = None, type = int,
                        help="How many times we wanna restart the model.")
    args = parser.parse_args(sys.argv[1:])
    
    
    # ## Debugging code ##
    # class Args:
    #     pass
    # args = Args()
    # args.prefix = "/home/heavens/Tools/FICT-SAMPLE/datasets/MERFISH/0"
    # args.output = "/home/heavens/Tools/FICT-SAMPLE/datasets/MERFISH/output"
    # args.config = None
    # args.n_type = 3
    # args.hidden = 10
    # args.plot = True
    # args.restart = 30
    # ##

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
