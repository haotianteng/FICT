#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:20:37 2020

@author: heavens
"""

import pandas
import numpy as np
import h5py
path = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/140genesData.xlsx"
out = "/home/heavens/CMU/FISH_Clustering/MERFISH_data/140genesCount.hdf5"
root = h5py.File(out,'w')
gene_data = pandas.read_excel(path)
gene_name_list = np.unique(gene_data['geneName'])
cell_list = np.unique(gene_data['cellID'])
cell_n = len(cell_list)
gene_n = len(gene_name_list)
dset = root.create_dataset("GeneCounts",(cell_n,gene_n),data = 'f4')
gene_entry = root.create_dataset("EntryAttribute",(gene_n,),data = bytes)
gene_entry[:] = gene_name_list
for cell_i, cell in enumerate(cell_list):
    cell_entry = gene_data[gene_data['cellID']==cell_i]
    entry,counts = np.unique(cell_entry['geneName'],return_counts = True)
    x,y = cell_entry.iloc[0]['CellPositionX'],cell_entry.iloc[0]['CellPositionY']
    dset[ce]