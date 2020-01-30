#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:18:28 2020

@author: heavens
"""
import pandas as pd
import os

base_folder = "/home/heavens/CMU/FISH_Clustering/seqfish_data/Collection/"
cortex_cell_loc = os.path.join(base_folder,"cortex_svz_cellcentroids.csv")
cortex_cell_type = os.path.join(base_folder,"cortex_svz_cell_type_annotations.csv")
cortex_gene_count = os.path.join(base_folder,"cortex_svz_counts.csv")
ob_cell_loc = os.path.join(base_folder,"ob_cellcentroids.csv")
ob_cell_type = os.path.join(base_folder,"ob_cell_type_annotations.csv")
ob_gene_count = os.path.join(base_folder,"ob_counts.csv")

def read_count(type_index,gene_count,cell_loc):
    type_index = pd.read_csv(type_index)
    gene_count = pd.read_csv(gene_count)
    cell_loc = pd.read_csv(cell_loc)
    cell_loc.insert(5,'louvain',type_index['louvain'])