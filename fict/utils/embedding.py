#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:42:52 2020

@author: haotian teng
"""
import torch
import os

def load_embedding(model_folder):
    """Loading function for embedding trained by GECT:
    https://github.com/haotianteng/GECT
    """
    if model_folder is None:
        return None
    ckpt_file = os.path.join(model_folder,'checkpoint') 
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    state_dict = torch.load(os.path.join(model_folder,latest_ckpt))
    embedding_matrix = state_dict['linear1.weight'].detach().cpu().numpy()
    return embedding_matrix.transpose()