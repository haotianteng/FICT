#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 07:17:18 2020

@author: haotian teng
"""
from fict.fict_input import RealDataLoader
import numpy as np
import pandas as pd
from fict.utils import data_op as dop

### Hyper parameter setting
print("Setting hyper parameter")
sample_n = 1000 #Number of samples
n_g = 15 #Number of genes
n_c = 10 #Number of cell type
density = 20 #The average number of neighbour for each cells.
threshold_distance = 100 # The threshold distance of neighbourhood.
gene_col = np.arange(9,164)
coor_col = [5,6]
header = 0
data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH2018/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
save_f = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/df_"
smfishHmrf_save = "/home/heavens/CMU/FISH_Clustering/Benchmark/Merfish/data/"
### Data preprocessing
print("Reading data from %s"%(data_f))
if data_f.endswith('.xlsx'):
    data_all = pd.read_excel(data_f,header = header)
elif data_f.endswith('.csv'):
    data_all = pd.read_csv(data_f,header = header)
animal_idxs = np.unique(data_all['Animal_ID'])
gene_expression_all = data_all.iloc[:,gene_col]
nan_cols = np.unique(np.where(np.isnan(gene_expression_all))[1])
for nan_col in nan_cols:
    gene_col = np.delete(gene_col,nan_col)
gene_name = data_all.columns[gene_col]
bregmas_smfish = [0.01]*3 + [-0.04]+[0.01]*3+[-0.04]*4+[0.11]*25
for animal_id in animal_idxs:
    print("Extract the data for animal %d"%(animal_id))
    data = data_all[data_all['Animal_ID']==animal_id]
    cell_types = data['Cell_class']
    data = data[cell_types!= 'Ambiguous']
    cell_types = data['Cell_class']
    try:
        bregma = data['Bregma']
    except:
        bregma = data['Field of View']
    gene_expression = data.iloc[:,gene_col]
    type_tags = np.unique(cell_types)
    coordinates = data.iloc[:,coor_col]
    coordinates = np.asarray(coordinates)
    gene_expression = np.asarray(gene_expression)
    gene_expression = gene_expression/np.sum(gene_expression,axis = 1,keepdims = True)
    gene_names = data
    real_df = RealDataLoader(gene_expression,
                             coordinates,
                             threshold_distance = threshold_distance,
                             gene_list = gene_name,
                             cell_labels = cell_types,
                             num_class = n_c,
                             field = bregma,
                             for_eval = False)
    mask = bregma == bregmas_smfish[animal_id-1]
    smfish_loader = RealDataLoader(gene_expression[mask,:],
								   coordinates[mask,:],
								   threshold_distance = threshold_distance,
                                   gene_list = gene_name,
                                   cell_labels = cell_types[mask],
                                   num_class = n_c,
                                   field = [bregmas_smfish[animal_id-1]]*sum(mask),
                                   for_eval = False)
    dop.save_smfish(smfish_loader,smfishHmrf_save+str(animal_id))
    dop.save_loader(real_df,save_f+str(animal_id))