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
from fict.utils.data_op import KL_divergence
from math import factorial
from fict.utils.random_generator import multinomial_wrapper as mp
from fict.utils.data_op import DataLoader
from fict.utils.scsim import Scsim
import time
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
        seed: Random seed, default None.
    """
    def __init__(self,
                 sample_n,
                 gene_n,
                 cell_type_n,
                 density,
                 seed = None):
        self.sample_n = sample_n
        self.gene_n = gene_n
        self.cell_n = cell_type_n
        self.seed = seed
        self.target_freq = None
        self.densty = None
        self.xrange = None
        self.coor = None
        self.distance_matrix = None
        self.adjacency = None
        self._neighbourhood_frequency = None
        self._neighbourhood_count = None
        self.cell_type_assignment = None
        self.all_types = None
        self.cell_prior = None
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
                        cell_prior = None,
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
        if cell_prior is None:
            cell_prior = np.random.random(self.cell_n)
            self.cell_prior = softmax(cell_prior+1)
        else:
            self.cell_prior = cell_prior/np.sum(cell_prior)
        self._neighbourhood_count = None
        self._neighbourhood_frequency = None
        
    def gen_coordinate(self,
                       density,
                       use_knearest = False,
                       ref_coor = None,
                       rank_criteria = "square"):
        """Random assign the coordinate
        Args:
            density: The sample density, number of samples in unit circle.
            
        """
        self.density = density
        if ref_coor is None:
            self.xrange = np.sqrt(np.pi*self.sample_n/self.density)
            self.coor = uniform(high = self.xrange,size = (self.sample_n,2))
        else:
            assert len(ref_coor) > self.sample_n
            central_point = np.median(ref_coor,axis = 0)
            ref_coor -= central_point
            if rank_criteria == "square":
                coor_criteria = np.max(np.abs(ref_coor),axis = 1,keepdims = False)**2*4
            elif rank_criteria == "euclidean":
                coor_criteria = np.sqrt(ref_coor[:,0]**2+ref_coor[:,1]**2)**2*np.pi
            coor_rank = np.argsort(coor_criteria)
            coor = ref_coor[coor_rank[:self.sample_n],:]
            effect_area = self.sample_n/density*np.pi
            scale = np.sqrt(effect_area/coor_criteria[coor_rank[self.sample_n-1]])
            self.coor = coor * scale
            self.xrange = max(self.coor[-1,0],self.coor[-1,1])
            self.ref_coor = ref_coor * scale
            
        self.distance_matrix = cdist(self.coor,self.coor,'euclidean')
        self.adjacency = np.zeros((self.sample_n,self.sample_n),dtype = bool)
        if use_knearest:
            for i,dist in enumerate(self.distance_matrix):
                sort_idx = np.argsort(dist)
                self.adjacency[i,sort_idx[:density+1]]
            self.exclude_adjacency = self.adjacency ^ np.eye(self.sample_n).astype(bool)
        else:
            self.adjacency = self.distance_matrix<1
            self.exclude_adjacency = self.adjacency ^ np.eye(self.sample_n).astype(bool)
                
        
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
                         tol = 1e-2,
                         max_iter = 4e2,
                         use_exist_assignment = False,
                         method = None,
                         soft_factor = None,
                         annealing = False,
                         initial_temperature = 20,
                         half_decay = 100,
                         local_criteria = True):
        """Generate cell type assignment iteratively from a given neighbourhood
        frequency, require the coordinates being generated first by calling
        self.gen_coordinate.
        Args:
            target_neighbourhood_frequency: A n-by-n target neighbourhood freuency matrix,
                n is the number of cell type, and ith row is the neighbourhood 
                frequency of ith cell, target_neighbourhood_frequency[i].
            tol: The error tolerance.
            max_iter: Max iteraton.
            method: Default is 'swap', can be 'assign-neighbour'
            annealing: Default is False. If annealing is used when the method is Metropolis-swap.
            initial_temperature: The initial temeprature to used in the annealing algorithm.
            half_decay: The steps of the half decay temperature.
            local_criteria: If calculate the swap probabiltiy based on local change.
        """
        self.target_freq = target_neighbourhood_frequency
        self.log_target_freq = np.log(self.target_freq)
        if not use_exist_assignment:
            self.cell_type_assignment = choice(np.arange(self.cell_n),
                                    size = self.sample_n,
                                    p = self.cell_prior).astype(int)
        self._get_neighbourhood_frequency()
        error = np.linalg.norm(self._neighbourhood_frequency-target_neighbourhood_frequency)
        print("    0 iteration, error %.2f"%(error))
        iter_n = 0
        perm = np.arange(self.sample_n)
        cell_types = np.arange(self.cell_n)
        error_record = []
        mps = []
        for i in np.arange(self.cell_n):
            mps.append(mp(target_neighbourhood_frequency[i]))
        while error>tol and iter_n<max_iter:
            if method=='assign-neighbour':
                np.random.shuffle(perm)
                for i in perm:
                    i_type = self.cell_type_assignment[i]
                    mask = np.copy(self.adjacency[i])
                    mask[i] = False #Exclude the self count.
                    neighbour_n = np.sum(mask)
                    self.cell_type_assignment[i] = np.random.choice(cell_types,
                                                                    size = 1,
                                                                    p = self.cell_prior)[0]
                    reasign_type = np.random.choice(cell_types,
                                                    size = neighbour_n,
                                                    p = target_neighbourhood_frequency[i_type])
                    self.cell_type_assignment[mask] = reasign_type
                    iter_n+=1
            elif method is None or method=="Metropolis-swap":
                for i in np.arange(100):
                    swap_cluster = np.random.choice(np.arange(self.cell_n),2,replace = False)
                    cell_idxs = np.arange(self.sample_n)
                    label = self.cell_type_assignment
                    i = np.random.choice(cell_idxs[label==swap_cluster[0]],1)[0]
                    j = np.random.choice(cell_idxs[label==swap_cluster[1]],1)[0]
                    nb_indexs = np.where(np.logical_or(self.exclude_adjacency[i] ,self.exclude_adjacency[j]))[0]
                    dist_before = 0
                    for k, freq in enumerate(self._neighbourhood_frequency):
                        dist_before += KL_divergence(target_neighbourhood_frequency[k],freq)
                    label[j],label[i] = label[i],label[j]
                    self._get_neighbourhood_count(nb_indexs)
                    # Update the neighbourhood count for the neighbourhood
                    # of swap pairs only.
                    self._get_neighbourhood_frequency(recount_neighbourhood = False)
                    dist_after = 0
                    for k, freq in enumerate(self._neighbourhood_frequency):
                        dist_after += KL_divergence(target_neighbourhood_frequency[k],freq)
                    swap_p = np.random.uniform()
                    temperature = initial_temperature*0.5**(iter_n/half_decay)
                    if annealing:
                        swap = temperature*np.log(swap_p)
                    else:
                        swap = 0
                    if swap>(dist_before - dist_after):
                        #Reject the swap, so swap back.
                        label[j],label[i] = label[i],label[j]
                        self._get_neighbourhood_count(nb_indexs)
                        self._get_neighbourhood_frequency(recount_neighbourhood = False)
                    iter_n+=1
            elif method=="Gibbs-sampling":
                np.random.shuffle(perm)
                ll = self._get_neighbourhood_likelihood(mps)
                ll = np.choose(self.cell_type_assignment,ll)
                perm = np.argsort(ll)
                for i in perm:
                    mask = np.copy(self.exclude_adjacency[i])
                    assign_prob = self._assign_probability(i,
                                                           self._neighbourhood_count[i],
                                                           mask,
                                                           mps,
                                                           self.cell_prior,
                                                           s =soft_factor)
                    assign_cell_type = np.random.choice(cell_types,
                                                        size = 1,
                                                        p = assign_prob)[0]
                    self.cell_type_assignment[i] = assign_cell_type
                    self._get_neighbourhood_count(i=mask)
                
                    iter_n+=1
            if iter_n %100 == 0:
                #Update error every 100 iterations.
                error = np.linalg.norm(self._neighbourhood_frequency-target_neighbourhood_frequency)
            if iter_n%1000 == 0:
                self._get_neighbourhood_frequency(recount_neighbourhood = False)
                error = np.linalg.norm(self._neighbourhood_frequency-target_neighbourhood_frequency)
                print("%5d iteration, error %.2f"%(iter_n,error))
                error_record.append(error)
                klds = np.empty(self.cell_n)
                for i, freq in enumerate(self._neighbourhood_frequency):
                    klds[i] = KL_divergence(target_neighbourhood_frequency[i],freq)
#                print("%d iteration, error %.2f"%(iter_n,error))
#                print("KL divergence %s"%(",".join([str(round(x,3)) for x in klds])))
#                print(np.unique(self.cell_type_assignment,return_counts = True))
        return error_record
    
    def _assign_probability(self,
                            cell_index,
                            neighbourhood_count,
                            neighbourhood_mask,
                            target_neighbourhood_frequency_pdf,
                            prior,
                            s=None):
        """Calculate the posterior probability given the neighbourhood cell type.
        Args:
            neighbourhood_count: A C vector indicate the count of the current cell.
            neighbourhood_mask: A length N boolean vector indicate the neighbourhood
                of current cell.
            target_neighbourhood_frequency_pdf:A length C list contain the
                multinomial distribution of target frequency of each cell type.
            prior: A length C vector indicate the prior probability of each cell type.
            s: Soft factor, if None, using the density.
        """
        neighbourhood_matrix = self._neighbourhood_count[neighbourhood_mask]
        neighbourhood_cell_type = self.cell_type_assignment[neighbourhood_mask]
        posterior = np.zeros(self.cell_n)
        current_cell_type = self.cell_type_assignment[cell_index]
        if s is None:
            s = self.density
        nb_copy = neighbourhood_matrix.copy()
        nb_copy[:,current_cell_type] -=1
        for i in np.arange(self.cell_n):
            posterior[i] = np.sum([self.log_target_freq[cl,i]-np.log(nb_copy[l,i]+1) 
                                  for l,cl in enumerate(neighbourhood_cell_type)])+\
                           np.sum([self.log_target_freq[i,j]*neighbourhood_count[j] 
                                  for j in np.arange(self.cell_n)])
        assign_prob = posterior/s
        assign_prob = softmax(assign_prob)
        return assign_prob
    
    def _get_neighbourhood_likelihood(self,mn_pdf):
        ll = np.empty((self.cell_n,self.sample_n))
        for i in np.arange(self.cell_n):
            ll[i,:] = mn_pdf[i].logpmf(self._neighbourhood_count)
        return ll
            
    def _get_neighbourhood_frequency(self,recount_neighbourhood = True):
        """Get the neighbourhood frequency from the cell type assignment or from
        a given neighbourhood count.
        """
        if self._neighbourhood_frequency is None:
            self._neighbourhood_frequency = np.zeros((self.cell_n,self.cell_n))
        if recount_neighbourhood:
            self._get_neighbourhood_count()
        count = self._neighbourhood_count
        for i in range(self.cell_n):
                self._neighbourhood_frequency[i]=np.sum(count[self.cell_type_assignment==i],axis = 0)
        self._neighbourhood_frequency = self._neighbourhood_frequency/np.sum(self._neighbourhood_frequency,axis = 1,keepdims = True)
    
    def _get_neighbourhood_count(self,i = None):
        one_hot = np.zeros((self.sample_n,self.cell_n))
        one_hot[np.arange(self.sample_n),self.cell_type_assignment] = 1
        if i is None:
            self._neighbourhood_count = np.matmul(self.exclude_adjacency,one_hot)
        else:
            self._neighbourhood_count[i] = np.matmul(self.exclude_adjacency[i],one_hot)

    
    def gen_expression(self,
                       zeroing = True, 
                       seed = None,
                       drop_rate = None):
        """Generate gene expression, need to call assign_cell_type first.
        """
        gene_expression = np.empty((self.sample_n,self.gene_n),dtype = float)
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        if drop_rate is not None:
            assert len(drop_rate) ==  self.gene_n
            mask = np.empty((self.sample_n,self.gene_n),dtype = bool)
            for i in np.arange(self.gene_n):
                mask[:,i] = np.random.choice([True,False],size = (self.sample_n),p = [drop_rate[i],1-drop_rate[i]])
        for i in range(self.cell_n):
            cell_type_mask = self.cell_type_assignment == i
            count = np.sum(cell_type_mask)
            current_expression = np.random.multivariate_normal(mean = self.g_mean[i],
                                                               cov = self.g_cov[i],
                                                               size = count)
            if zeroing:
                current_expression[current_expression<0] = 0
            if drop_rate is not None:
                current_expression[mask[cell_type_mask,:]] = 0
            gene_expression[cell_type_mask,:] = current_expression
        return np.asarray(gene_expression),self.cell_type_assignment,self._neighbourhood_count
    def gen_expression_splatter(self,seed=None):
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        celltype = self.cell_type_assignment
        n_g = self.gene_n
        sample_n = self.sample_n
        celltype_n = self.cell_n
        deval = 1
        K = celltype_n
        doubletfrac = 0
        ncells = sample_n
        ngenes=n_g
        nproggenes = 400
        ndoublets=int(doubletfrac*ncells)
        groupid = celltype
        deloc=deval
        progdeloc=deval
        descale=1.0
        progcellfrac = .35
        deprob = .025
        nproggroups = K #Enable program genes in all the groups
        proggroups = list(range(1, nproggroups+1))
        print("Simulation using splatter.")
        simulator = Scsim(ngenes=ngenes, ncells=ncells, ngroups=K,groupid = groupid, libloc=7.64, libscale=0.78,
                     mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                     expoutloc=6.15, expoutscale=0.49,
                     diffexpprob=deprob, diffexpdownprob=0., diffexploc=deloc, diffexpscale=descale,
                     bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=ndoublets,
                     nproggenes=nproggenes, progdownprob=0., progdeloc=progdeloc,
                     progdescale=descale, progcellfrac=progcellfrac, proggoups=proggroups,
                     minprogusage=.1, maxprogusage=.7, seed=seed)
        start_time = time.time()
        simulator.simulate()
        end_time = time.time()
        print("Elapsing time is %.2f"%(end_time - start_time))
        self.gene_expression = np.asarray(simulator.counts)
        return self.gene_expression,self.cell_type_assignment,self._neighbourhood_count
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
    drop_rate = np.sum(gene_expression==0,axis = 0)/(gene_expression.shape[0])
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
    gene_expression,cell_type,cell_neighbour = sim.gen_expression(drop_rate = drop_rate[:n_g])
    save_simulation(sim,model_f)
    sim2 = load_simulation(model_f)