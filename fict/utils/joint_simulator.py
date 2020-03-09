#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 03:01:35 2020

@author: heavens
"""
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import multinomial
import pandas as pd
from numpy.random import uniform
from numpy.random import choice
from fict.utils.opt import valid_neighbourhood_frequency
from math import factorial
from fict.utils.random_generator import multinomial_wrapper as mp
from fict.utils.data_op import DataLoader
import pickle

def save_simulation(sim,file):
    """Save a Simulator instance to file.
    Args:
        sim: A instance of the Simulator.
        file: The file to save.
    """
    with open(file,'wb+') as f:
        pickle.dump(sim,f)

def load_simulation(file):
    """Load a Simulator instance from file
    Args:
        file: A file path to load the simulator.
    """
    with open(file,'rb') as f:
        instance = pickle.load(f)
    return instance

def get_gene_prior(gene_expression,cell_types,header = 0,):
    """Get the prior parameter (mean and std) of the given gene expression
    matrix and cell type assignment.
    """
    gene_expression = np.asarray(gene_expression)
    gene_mean = []
    gene_std = []
    tags = np.unique(cell_types)
    for tag in tags:
        sub_data = gene_expression[cell_types == tag]
        gene_mean.append(np.mean(sub_data,axis = 0))
        gene_std.append(np.std(sub_data,axis = 0))
    return np.asarray(gene_mean),np.asarray(gene_std)

def get_nf_prior(coordinates,cell_types,threshold = 100):
    distance_matrix = cdist(coordinates,coordinates,'euclidean')
    adjacency = distance_matrix<threshold
    type_tages_all,type_count = np.unique(cell_types,return_counts = True)
    type_indexs = {}
    for tag_id,tag in enumerate(type_tages_all):
        type_indexs[tag] = tag_id
    n_types = len(type_tages_all)
    nb_freq = np.zeros((n_types,n_types))
    n_sample = coordinates.shape[0]
    for i in range(n_sample):
        cell_bags = cell_types[adjacency[i,:]]
        type_tags,counts = np.unique(cell_bags,return_counts = True)
        for j,tag in enumerate(type_tags):
            nb_freq[type_indexs[cell_types[i]],type_indexs[tag]] += counts[j]
    nb_freq += 1e-3 #Add pseudocount
    nb_freq = nb_freq/np.sum(nb_freq,axis = 1,keepdims = True)
    return nb_freq,type_tages_all,type_count

class Simulator():
    """A simulator for generating gene expression and spatio location profile.
    Args:
        sample_n: The number of sample(cell) need to be generated.
        gene_n: The number of genes for each cell.
        cell_type_n: The number of cell types.
        density: The number o fcell in the unit cycle(also it's the average neighbourhood)
        seed: Random seed, default is 1992.
    """
    def __init__(self,
                 sample_n,
                 gene_n,
                 cell_type_n,
                 density,
                 seed = 1992):
        self.sample_n = sample_n
        self.gene_n = gene_n
        self.cell_n = cell_type_n
        self.seed = seed
    def gen_data_from_real(self,gene_expression,cell_type_list,neighbour_freq_prior):
        gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
        target_freq = (neighbour_freq_prior+0.1)/np.sum(neighbour_freq_prior+0.1,axis=1,keepdims=True)
        result = valid_neighbourhood_frequency(target_freq)
        self.target_freq = result[0]
        self.gen_parameters(gene_mean_prior = gene_mean[:,:n_g])
        self.gen_coordinate(density = density)
        sim.assign_cell_type(target_neighbourhood_frequency=self.target_freq)
        return sim
    def gen_parameters(self,
                        gene_mean_prior = None,
                        seed = None):
        self.all_types = np.arange(self.cell_n)
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        self.g_mean = np.exp(np.random.rand(self.cell_n,self.gene_n)+1)
        if gene_mean_prior is not None:
            self.g_mean += gene_mean_prior
        g_cov = np.random.rand(self.cell_n,self.gene_n,self.gene_n)
        self.g_cov = np.asarray([np.dot(x,x.transpose())/self.gene_n for x in g_cov])
        cell_prior = np.random.random(self.cell_n)
        self.cell_prior = softmax(cell_prior+1)
        self._neighbourhood_count = None
        self._neighbourhood_frequency = None
        
    def gen_coordinate(self,density):
        """Random assign the coordinate
        Args:
            density: How many samples in the unit circle.
            
        """
        self.density = density
        self.xrange = np.sqrt(np.pi*self.sample_n/self.density)
        self.coor = uniform(high = self.xrange,size = (self.sample_n,2))
        self.distance_matrix = cdist(self.coor,self.coor,'euclidean')
        self.adjacency = self.distance_matrix<1
        
    @property
    def neighbour_frequency(self):
        if self._neighbourhood_frequency is None:
            self._get_neighbourhood_frequency()
        return self._neighbourhood_frequency
    
    @property
    def neighbour_count(self):
        if self._neighbourhood_frequency is None:
            self._get_neighbourhood_count()
        return self._neighbourhood_count
    
    def assign_cell_type(self,
                         target_neighbourhood_frequency,
                         tol = 1e-1,
                         max_iter = 1e2,
                         method = None):
        """Generate cell type assignment iteratively from a given neighbourhood
        frequency, require the coordinates being generated first by calling
        self.gen_coordinate.
        Args:
            target_neighbourhood_frequency: A n-by-n target neighbourhood freuency matrix,
                n is the number of cell type, and ith row is the neighbourhood 
                frequency of ith cell, target_neighbourhood_frequency[i].
            tol: The error tolerance.
            max_iter: Max iteraton.
            method: Default is 'assign-cell', can be 'assign-neighbour'
        """
        self.cell_type_assignment = choice(np.arange(self.cell_n),
                                size = self.sample_n,
                                p = self.cell_prior)
        self._get_neighbourhood_frequency()
        error = np.linalg.norm(self._neighbourhood_frequency-target_neighbourhood_frequency)
        print("0 iteration, error %.2f"%(error))
        iter_n = 0
        perm = np.arange(self.sample_n)
        cell_types = np.arange(self.cell_n)
        error_record = []
        mps = []
        for i in np.arange(self.cell_n):
            mps.append(mp(target_neighbourhood_frequency[i]))
        while error>tol and iter_n<max_iter:
            np.random.shuffle(perm)
            for i in perm:
                if method is None or (method == "assign-cell"):
                    mask = np.copy(self.adjacency[i])
                    neighbour_matrix = self._neighbourhood_count[mask]
                    neighbour_celltype = self.cell_type_assignment[mask]
                    assign_prob = self._assign_probability(self._neighbourhood_count[i],
                                                           neighbour_matrix,
                                                           neighbour_celltype,
                                                           mps,
                                                           self.cell_prior)
                    self.cell_type_assignment[i] = np.random.choice(cell_types,
                                                                    size = 1,
                                                                    p = assign_prob)[0]
                elif method=='assign-neighbour':
                    i_type = self.cell_type_assignment[i]
                    mask = np.copy(self.adjacency[i])
                    neighbour_n = np.sum(mask)-1
                    mask[i] = False #Exclude the self count.
                    reasign_type = np.random.choice(cell_types,
                                                    size = neighbour_n,
                                                    p = target_neighbourhood_frequency[i_type])
                    self.cell_type_assignment[mask] = reasign_type
            self._get_neighbourhood_frequency()
            iter_n+=1
            print(np.unique(self.cell_type_assignment,return_counts = True))
            if iter_n%1 == 0:
                error = np.linalg.norm(self._neighbourhood_frequency-target_neighbourhood_frequency)
                error_record.append(error)
                print("%d iteration, error %.2f"%(iter_n,error))
        return error_record
    
    def _assign_probability(self,
                            neighbourhood_count,
                            neighbourhood_matrix,
                            neighbourhood_cell_type,
                            target_neighbourhood_frequency_pdf,
                            prior):
        """Calculate the posterior probability given the neighbourhood cell type.
        Args:
            neighbourhood_count: A length N vector indicate the count of N cell 
                types of the neighbourhood of given cell.
            neighbourhood_matrix:A X-by-N matrix rows contain the neighbourhood count
                of the X neighbourhood cells of given cell.
            neighbourhood_cell_type: A X length vector indicate the cell type of the neighbourhood.
            target_neighbourhood_frequency_pdf:A length N list contain the
                multinomial distribution of target frequency.
            prior: A length N vector indicate the prior probability.
        """
        alpha=2.5
        posterior = np.zeros(self.cell_n)
        for i in range(self.cell_n):
            posterior[i] = target_neighbourhood_frequency_pdf[i].logpmf(neighbourhood_count)/np.sum(neighbourhood_count)*alpha
            for j,count in enumerate(neighbourhood_matrix):
                count = np.copy(count)
                count[i] += 1
                posterior[i] += target_neighbourhood_frequency_pdf[neighbourhood_cell_type[j]].logpmf(count)/np.sum(count)*alpha

        assign_prob = posterior + np.log(prior)
        assign_prob = softmax(assign_prob)
        return assign_prob
        
    def _get_neighbourhood_frequency(self):
        """Get the neighbourhood frequency from the cell type assignment or from
        a given neighbourhood count.
        """
        if self._neighbourhood_frequency is None:
            self._neighbourhood_frequency = np.zeros((self.cell_n,self.cell_n))
        self._get_neighbourhood_count()
        count = self._neighbourhood_count
        for i in range(self.cell_n):
                self._neighbourhood_frequency[i]=np.sum(count[self.cell_type_assignment==i],axis = 0)
        self._neighbourhood_frequency = self._neighbourhood_frequency/np.sum(self._neighbourhood_frequency,axis = 1,keepdims = True)
    
    def _get_neighbourhood_count(self):
        one_hot = np.zeros((self.sample_n,self.cell_n))
        one_hot[np.arange(self.sample_n),self.cell_type_assignment] = 1
        self._neighbourhood_count = np.matmul(self.adjacency,one_hot)
    
    def gen_expression(self,seed = None):
        """Generate gene expression, need to call assign_cell_type first.
        """
        gene_expression = []
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        for i in range(self.sample_n):
            current_t = self.cell_type_assignment[i]
            gene_expression.append(np.random.multivariate_normal(mean = self.g_mean[current_t],cov = self.g_cov[current_t]))
        return np.asarray(gene_expression),self.cell_type_assignment,self._neighbourhood_count

class SimDataLoader(DataLoader):
    def __init__(self,
                 gene_expression,
                 cell_neighbour,
                 cell_type_assignment):
        """Class for loading the data.
        Input Args:
            gene_expression: A N-by-M matrix contain the gene expression for N samples and M genes.
            cell_neighbour: A N-by-C matrix contain the neighbourhood count for N samples and C cell types.
            cell_type_assignment: If the dataset is not for evaluation, then a length N vector indicate the
                cell_type_assignment need to be provided.
        """
        super().__init__(xs = (gene_expression,cell_neighbour),
                         y = cell_type_assignment,
                         for_eval = False)


if __name__ == "__main__":
    ### Hyper parameter setting
    sample_n = 1000 #Number of samples
    n_g = 100 #Number of genes
    n_c = 10 #Number of cell type
    density = 20 #The average number of neighbour for each cells.
    threshold_distance = 1 # The threshold distance of neighbourhood.
    gene_col = np.arange(9,164)
    coor_col = [5,6]
    header = 1
    data_f = "/home/heavens/CMU/FISH_Clustering/FICT/example_data2/aau5324_Moffitt_Table-S7.xlsx"
    model_f = "/home/heavens/CMU/FISH_Clustering/test_sim1"
    
    ### Data preprocessing
    data = pd.read_excel(data_f,header = header)
    gene_expression = data.iloc[:,gene_col]
    cell_types = data['Cell_class']
    type_tags = np.unique(cell_types)
    coordinates = data.iloc[:,coor_col]
    
    ### Choose only the n_c type cells
    if len(type_tags)<n_c:
        raise ValueError("Only %d cell types presented in the dataset, but require %d, reduce the number of cell type assigned."%(len(type_tags),n_c))
    mask = np.asarray([False]*len(cell_types))
    for tag in type_tags[:n_c]:
        mask = np.logical_or(mask,cell_types==tag)
    gene_expression = gene_expression[mask]
    cell_types = np.asarray(cell_types[mask])
    coordinates = np.asarray(coordinates[mask])
    
    ### Generate prior from the given dataset.
    gene_mean,gene_std = get_gene_prior(gene_expression,cell_types)
    neighbour_freq_prior,tags,type_count = get_nf_prior(coordinates,cell_types)
    type_prior = type_count/np.sum(type_count)
    target_freq = (neighbour_freq_prior+0.1)/np.sum(neighbour_freq_prior+0.1,axis=1,keepdims=True)
    result = valid_neighbourhood_frequency(target_freq)
    target_freq = result[0]
    
    ### Generate simulation dataset and load
    sim = Simulator(sample_n,n_g,n_c,density)
    sim.gen_parameters(gene_mean_prior = gene_mean[:,:n_g])
    sim.gen_coordinate(density = density)
    sim.assign_cell_type(target_neighbourhood_frequency=target_freq, method = "assign-neighbour")
    gene_expression,cell_type,cell_neighbour = sim.gen_expression()
    save_simulation(sim,model_f)
    sim2 = load_simulation(model_f)