#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 07:18:45 2020

@author: haotian teng
"""

from fict.fict_input import RealDataLoader
import numpy as np
import pandas as pd
from fict.utils import data_op as dop


file1 = "/home/heavens/twilight/data/Seqfish+data/cortex_svz_cellcentroids.csv"
file2 = "/home/heavens/twilight/data/Seqfish+data/cortex_svz_cell_type_annotations.csv"
file3 = "/home/heavens/twilight/data/Seqfish+data/cortex_svz_counts.csv"
merge_f = "/home/heavens/twilight/data/Seqfish+data/cortex_svz.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df = pd.concat([df1,df2,df3],axis = 1,sort = False)
df = df.rename({'louvain':'Cell_class'},axis = 1)
df['Animal_ID'] = 0
df.to_csv(merge_f)
### Hyper parameter setting
print("Setting hyper parameter")
n_c = len(np.unique(df['Cell_class'])) #Number of cell type
threshold_distance = 100 # The threshold distance of neighbourhood.
gene_col = np.arange(7,10007)
coor_col = [2,3]
header = 0
save_f = "/home/heavens/twilight/data/Seqfish+data/df_"
### Data preprocessing
animal_idxs = np.unique(df['Animal_ID'])
gene_expression_all = df.iloc[:,gene_col]
nan_cols = np.unique(np.where(np.isnan(gene_expression_all))[1])
for nan_col in nan_cols:
    gene_col = np.delete(gene_col,nan_col)
for animal_id in animal_idxs:
    print("Extract the data for animal %d"%(animal_id))
    data = df[df['Animal_ID']==animal_id]
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
    real_df = RealDataLoader(gene_expression,
                             coordinates,
                             threshold_distance = threshold_distance,
                             cell_labels = cell_types,
                             num_class = n_c,
                             field = bregma,
                             for_eval = False)
    dop.save_loader(real_df,save_f+str(animal_id))