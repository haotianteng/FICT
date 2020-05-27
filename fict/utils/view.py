#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:49:41 2020

@author: heavens
"""
import numpy as np
import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from fict.fict_input import RealDataLoader
from fict.utils import data_op as dop
import sys

def get_meta(meta_file = None):
    if meta_file is not None:
        meta = json.load(meta_file)
    else:
        meta = {'header':0,
                'tag':{'exclude_label':'Ambiguous',
                        'label':'Cell_class',
                'sub_label':'Neuron_cluster_ID',
                'replication':'Animal_ID'},
                'loc':{'gene':np.arange(9,164),
                       'coordinate':[5,6]}
                }
    return meta
    

def extract(args):
    ### Hyper parameter setting
    print("Setting hyper parameter")
    meta = get_meta(args.meta_file)
    threshold_distance = 100 # The threshold distance of neighbourhood.
    gene_col = meta['loc']['gene']
    coor_col = meta['loc']['coordinate']
    header = meta['header']
    label_tag = meta['tag']['label']
    rep_tag = meta['tag']['replication']
    exclude_label = meta['tag']['exclude_label']
    data_f = args.input
    save_prefix = os.path.join(args.output_dir,'dataloader_')
    ### Data preprocessing
    print("Reading data from %s"%(data_f))
    if data_f.endswith('.xlsx'):
        data_all = pd.read_excel(data_f,header = header)
    elif data_f.endswith('.csv'):
        data_all = pd.read_csv(data_f,header = header)
    repeatations = np.unique(data_all[rep_tag])
    gene_expression_all = data_all.iloc[:,gene_col]
    nan_cols = np.unique(np.where(np.isnan(gene_expression_all))[1])
    for nan_col in nan_cols:
        gene_col = np.delete(gene_col,nan_col)
    for repeat in tqdm(repeatations,desc = "Extract repeatation from input file."):
        data = data_all[data_all[rep_tag]==repeat]
        cell_types = data[label_tag]
        data = data[cell_types!= exclude_label]
        cell_types = data[label_tag]
        gene_expression = data.iloc[:,gene_col]
        coordinates = data.iloc[:,coor_col]
        coordinates = np.asarray(coordinates)
        gene_expression = np.asarray(gene_expression)
        real_df = RealDataLoader(gene_expression,
                                 coordinates,
                                 threshold_distance = threshold_distance,
                                 cell_labels = cell_types,
                                 for_eval = False)
        dop.save_loader(real_df,save_prefix+str(repeat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the gene expression and cell classification information.')
    parser.add_argument('-i', 
                        '--input', 
                        required = True,
                        help="Input merfish file.")
    parser.add_argument('-o', 
                        '--output_dir', 
                        required = True,
                        help="Directory that output the signal and reference sequence.")
    parser.add_argument('-m',
                        '--meta_file',
                        default = None,
                        help="The file contain meta information of the FISH data.")
    
    
    FLAGS = parser.parse_args(sys.argv[1:])
#    extract(FLAGS)
    data_f = FLAGS.input
    save_prefix = os.path.join(FLAGS.output_dir,'dataloader_')
    ### Data preprocessing
    print("Reading data from %s"%(data_f))
    if data_f.endswith('.xlsx'):
        data_all = pd.read_excel(data_f,header = 0)
    elif data_f.endswith('.csv'):
        data_all = pd.read_csv(data_f,header = 0)