#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:43:29 2020

@author: heavens
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def tag2int(label):
    tags = np.unique(label)
    label_n = len(label)
    int_label = np.empty(label_n,dtype = np.int)
    for tag_idx,tag in enumerate(tags):
        int_label[label==tag] = tag_idx
    return int_label,tags

def one_hot_vector(label):
    label_n = len(label)
    if label.dtype != np.int:
        print("Input vector has a dtype %s"%(label.dtype))
        print("However a int type is required, transfer the data type.")
        label = label.astype(int)
    tags = np.unique(label)
    class_n = len(tags)
    one_hot = np.zeros((label_n,class_n))
    for tag_idx,tag in enumerate(tags):
        one_hot[label==tag,tag_idx] = 1
    return one_hot
    
def pca_reduce(X, dims=2):
    """ Reduce the dimensions of X down to dims using PCA
    X has shape (n, d)
    Returns: The reduced X of shape (n, dims)
    		 The fitted PCA model used to reduce X
    """
    print("reducing dimensions using PCA")
    X = X - np.mean(X,axis = 0,keepdims = True)
    X = X/np.std(X,axis = 0, keepdims = True)
    pca = PCA(n_components = dims)
    pca.fit(X)
    X_reduced = pca.transform(X)
    return X_reduced

def KL_divergence(prob1,prob2,pesudo=1e-6):
    if 0 in prob1:
        prob1 = prob1+pesudo
        prob1 = prob1/np.sum(prob1)
        print("Zero element in the first distribution, add pesudo count %.2f."%(pesudo))
    if 0 in prob2:
        prob2 = prob2+pesudo
        prob2 = prob2/np.sum(prob2)
        print("Zero element in the second distribution, add pesudo count %.2f."%(pesudo))
    kld = np.sum([x*np.log(x/y) for x,y in zip(prob1,prob2)])
    return kld

def get_adjacency(coordinate,
                  threshold_distance):
    distance_matrix = cdist(coordinate,coordinate,'euclidean')
    adjacency = distance_matrix<threshold_distance
    return adjacency

def get_neighbourhood_count(adjacency, 
                            label,
                            exclude_self = True,
                            one_hot_label = True):
    """Get the neighbourhood type count given a adjacency matrix and label
    Args:
        adjacency: A N-by-N 0-1 adjacency matrix, N is the number of samples.
        label: A length N vector indicate the label tags.
        exclude_self: A boolean variable, if true the self-count is excluded.
    Return:
        nb_count: A N-by-M matrix given the neighbourhood count, where N is the
            number of sample, and M is the number of types of label.
    """
    sample_n = adjacency.shape[0]
    assert label.shape[0] == sample_n
    tags = np.unique(label)
    if one_hot_label:
        one_hot = label
    else:
        label_n = len(tags)
        one_hot = np.zeros((sample_n,label_n))
        one_hot[np.arange(sample_n),label] = 1
    if exclude_self:
        diag_mask = np.eye(sample_n).astype(bool)
        adjacency[diag_mask] = 0
    nb_count = np.matmul(adjacency,one_hot)
    return nb_count

class DataLoader():
    def __init__(self,
                 xs,
                 y = None,
                 for_eval = False):
        """Class for loading the data.
        Input Args:
            xs: The input data, can be a tuple of arrays with the same length in
                0 dimension.
            y: The label, need to have same length as x.
            for_eval: If the dataset is for evaluation.
        """
        if type(xs) is tuple:
            self.xs = xs
        else:
            self.xs = tuple(xs)
        self.y = y
        self.epochs_completed = 0
        self._index_in_epoch = 0
        self.sample_n = self.xs[0].shape[0]
        for x in self.xs:
            assert x.shape[0] == self.sample_n
        self._perm = np.arange(self.sample_n)
        self.for_eval = for_eval
        
    def read_into_memory(self, index):
        if self.for_eval:
            batch_y = None
        else:
            batch_y = self.y[index]
        batch_x = tuple([x[index] for x in self.xs])
        return batch_x,batch_y
    def _multi_concatenate(self,input_tuple,axis=0):
        result = [np.concatenate(x,axis = axis) for x in zip(*input_tuple)]
        return tuple(result)
    def next_batch(self, batch_size, shuffle=True):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:A scalar indicate the batch size.
                shuffle: boolean, indicate if the data should be shuffled after each epoch.
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)
        """
        if self.epochs_completed>=1 and self.for_eval:
            print("Warning, evaluation dataset already finish one iteration.")
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0:
            if shuffle:
                np.random.shuffle(self._perm)
        # Go to the next epoch
        if start + batch_size >= self.sample_n:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest samples in this epoch
            rest_sample_n = self.sample_n - start
            x_rest_part, y_rest_part = self.read_into_memory(self._perm[start:self.sample_n])
            start = 0
            if self.for_eval:
                x_batch = x_rest_part
                y_batch = y_rest_part
                self._index_in_epoch = 0
                end = 0
            # Shuffle the data
            else:
                if shuffle:
                    np.random.shuffle(self._perm)
                # Start next epoch
                self._index_in_epoch = batch_size - rest_sample_n
                end = self._index_in_epoch
                x_new_part, y_new_part = self.read_into_memory(
                    self._perm[start:end])
                if x_rest_part[0].shape[0] == 0:
                    x_batch = x_new_part
                    y_batch = y_new_part
                elif x_new_part[0].shape[0] == 0:
                    x_batch = x_rest_part
                    y_batch = y_rest_part
                else:
                    x_batch = self._multi_concatenate((x_rest_part, x_new_part))
                    y_batch = np.concatenate((y_rest_part, y_new_part),axis = 0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            x_batch,y_batch = self.read_into_memory(
                self._perm[start:end])
        return x_batch,y_batch