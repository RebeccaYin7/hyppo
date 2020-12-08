# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:40:53 2020

@author: siptest
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:11:06 2020

@author: siptest
"""


import hyppo
import numpy as np
from numba import njit
from hyppo.independence.base import IndependenceTest
from hyppo._utils import perm_test
from .._utils import euclidean, check_xy_distmat
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import copy 
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree._classes import DecisionTreeClassifier
from joblib import Parallel, delayed
from scipy.stats import entropy, multivariate_normal
from scipy.integrate import nquad

import sys
sys.executable
sys.path
sys.path.append('C:\\Users\\siptest\\AppData\\Roaming\\Python\\Python36\\Scripts')

class UF(IndependenceTest): 
    
    def _init(self, compute_distance=euclidean, bias = False):
            self.is_distance = False
            if not compute_distance: 
                self.is_distance = True
            self.bias = bias 
            
            IndependenceTest._init_(self, compute_distance=compute_distance) 
            
    def _statistic(self, X, y):
           self._perm_stat(X, y)
            
    def test(self, X, y): 
            perm_test(X, y)

    def uf(X, y, n_estimators = 300, max_samples = .4, base = np.exp(1), kappa = 3):
        
        # Build forest with default parameters.
        model = BaggingClassifier(DecisionTreeClassifier(), 
                                  n_estimators=n_estimators, 
                                  max_samples=max_samples, 
                                  bootstrap=False)
        model.fit(X, y)
        n = X.shape[0]
        K = model.n_classes_
        _, y = np.unique(y, return_inverse=True)
        
        cond_entropy = 0
        for tree_idx, tree in enumerate(model):
            # Find the indices of the training set used for partition.
            sampled_indices = model.estimators_samples_[tree_idx]
            unsampled_indices = np.delete(np.arange(0,n), sampled_indices)
            
            # Randomly split the rest into voting and evaluation.
            total_unsampled = len(unsampled_indices)
            np.random.shuffle(unsampled_indices)
            vote_indices = unsampled_indices[:total_unsampled//2]
            eval_indices = unsampled_indices[total_unsampled//2:]
            
            # Store the posterior in a num_nodes-by-num_classes matrix.
            # Posteriors in non-leaf cells will be zero everywhere
            # and later changed to uniform.
            node_counts = tree.tree_.n_node_samples
            class_counts = np.zeros((len(node_counts), K))
            est_nodes = tree.apply(X[vote_indices])
            est_classes = y[vote_indices]
            for i in range(len(est_nodes)):
                class_counts[est_nodes[i], est_classes[i]] += 1
            
            row_sums = class_counts.sum(axis=1) # Total number of estimation points in each leaf.
            row_sums[row_sums == 0] = 1 # Avoid divide by zero.
            class_probs = class_counts / row_sums[:, None]
            
            # Make the nodes that have no estimation indices uniform.
            # This includes non-leaf nodes, but that will not affect the estimate.
            class_probs[np.argwhere(class_probs.sum(axis = 1) == 0)] = [1 / K]*K
            
            # Apply finite sample correction and renormalize.
            where_0 = np.argwhere(class_probs == 0)
            for elem in where_0:
                class_probs[elem[0], elem[1]] = 1 / (kappa*class_counts.sum(axis = 1)[elem[0]])
            row_sums = class_probs.sum(axis=1)
            class_probs = class_probs / row_sums[:, None]
            
            # Place evaluation points in their corresponding leaf node.
            # Store evaluation posterior in a num_eval-by-num_class matrix.
            eval_class_probs = class_probs[tree.apply(X[eval_indices])]
            # eval_class_probs = [class_probs[x] for x in tree.apply(X[eval_indices])]
            eval_entropies = [entropy(posterior) for posterior in eval_class_probs]
            cond_entropy += np.mean(eval_entropies)
    
          
        return cond_entropy / n_estimators
            
    def generate_data2(n, d, mu = 1):
        n_1 = np.random.binomial(n, .5) # number of class 1
        mean = np.zeros(d)
        mean[0] = mu
        X_1 = np.random.multivariate_normal(mean, np.eye(d), n_1)
        
        X = np.concatenate((X_1, np.random.multivariate_normal(-mean, np.eye(d), n - n_1)))
        y = np.concatenate((np.repeat(1, n_1), np.repeat(0, n - n_1)))
      
        return X, y
    
    def generate_data(self, n, d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
        
        means, Sigmas, probs = self._make_params(d, mu = mu, var1 = var1, pi = pi, three_class = three_class)
        counts = np.random.multinomial(n, probs, size = 1)[0]
        
        X_data = []
        y_data = []
        for k in range(len(probs)):
            X_data.append(np.random.multivariate_normal(means[k], Sigmas[k], counts[k]))
            y_data.append(np.repeat(k, counts[k]))
        X = np.concatenate(tuple(X_data))
        y = np.concatenate(tuple(y_data))
        
        return X, y
    
    def _make_params(self, d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
        
        if three_class:
            return self._make_three_class_params(d, mu, pi)
        
        mean = np.zeros(d)
        mean[0] = mu
        means = [mean, -mean]
    
        Sigma1 = np.eye(d)
        Sigma1[0, 0] = var1
        Sigmas = [np.eye(d), Sigma1]
        
        probs = [pi, 1 - pi]
        
        return means, Sigmas, probs
    
    def _make_three_class_params(d, mu, pi):
        
        means = []
        mean = np.zeros(d)
        
        mean[0] = mu
        means.append(copy.deepcopy(mean))
        
        mean[0] = -mu
        means.append(copy.deepcopy(mean))
        
        mean[0] = 0
        mean[d-1] = mu
        means.append(copy.deepcopy(mean))
        
        Sigmas = [np.eye(d)]*3
        probs = [pi, (1 - pi) / 2, (1 - pi) / 2]
        
        return means, Sigmas, probs
    
    def compute_mutual_info(self, d, base = np.exp(1), mu = 1, var1 = 1, pi = 0.5, three_class = False):
        
        if d > 1:
            dim = 2
        else:
            dim = 1
     
        means, Sigmas, probs = self._make_params(dim, mu = mu, var1 = var1, pi = pi, three_class = three_class)
        
        # Compute entropy and X and Y.
        def func(*args):
            x = np.array(args)
            p = 0
            for k in range(len(means)):
                p += probs[k] * multivariate_normal.pdf(x, means[k], Sigmas[k])
            return -p * np.log(p) / np.log(base)
    
        scale = 10
        lims = [[-scale, scale]]*dim
        H_X, int_err = nquad(func, lims)
        H_Y = entropy(probs, base = base)
        
        # Compute MI.
        H_XY = 0
        for k in range(len(means)):
            H_XY += probs[k] * (dim * np.log(2*np.pi) + np.log(np.linalg.det(Sigmas[k])) + dim) / (2 * np.log(base))
        I_XY = H_X - H_XY
        
        return I_XY, H_X, H_Y
    
    n = 20 
    #n = 6000
    mus = range(5)
    ds = range(1, 16)
    mu = 1
    num_trials = 10
    #reps = 1
    d = 2
    pis = [0.05 * i for i in range(1, 20)]
    
    def estimate_mi(self, X, y, est_H_Y, norm_factor): 
        return (est_H_Y - self.uf(np.array(X), y)) / norm_factor
    
    def mi(self, X, y, n, d, pis, num_trials):
        #def worker(t): 
            #X, y = generate_data(n, d, pi = elem)
            
            #I_XY, H_X, H_Y = compute_mutual_info(d, pi = elem)
            I_XY, H_X, H_Y = self.compute_mutual_info(d)
            norm_factor = min(H_X, H_Y)
            
            _, counts = np.unique(y, return_counts=True)
            est_H_Y = entropy(counts, base=np.exp(1))
            ret = []
            ret.append(self.estimate_mi(X, y, est_H_Y, norm_factor))
            #return tuple(ret)
            return ret[0]
        
        #output = np.zeros((len(pis), num_trials))
        #for i, elem in enumerate(pis): 
            #results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
            #output[i, :] = results[:, 0]
        #return output
        
    def _perm_stat(self, X, y, is_distsim = True, permuter = None): 
        #if permuter is None: 
            #print(y.shape[0])
            #order = np.random.permutation(y.shape[0])
        #else: 
            #order = permuter()
        
        #print(y.shape[0])
        
        n = 20 
        #n = 6000
        mus = range(5)
        ds = range(1, 16)
        mu = 1
        num_trials = 10
        #reps = 1
        d = 2
        pis = [0.05 * i for i in range(1, 20)]
        
        order = np.random.permutation(y.shape[0])
        
        if is_distsim: 
            permy = y[order][:, order]
        else: 
            permy = y[order]
        
        perm_stat = self.mi(X, permy, n, d, pis, num_trials)
        
        #print(perm_stat)
        return perm_stat
    
    def perm_test(self, X, y, workers = 1, is_distsim=True, perm_block = None, reps = 50): 
    
        n = 20 
        #n = 6000
        mus = range(5)
        ds = range(1, 16)
        mu = 1
        num_trials = 10
        #reps = 1
        d = 2
        pis = [0.05 * i for i in range(1, 20)]
        # calculate observed test statistic
        stat = self.mi(X, y, n, d, pis, num_trials)
        print(stat) 
        print(y)
    
        print(reps)
        # calculate null distribution
        null_dist = np.array(
                #n_jobs = workers 
            Parallel(n_jobs=-2)(
                [
                    #delayed(_perm_stat)(X, y, is_distsim)
                    delayed(self._perm_stat)(X, y, False)
                    for rep in range(reps)
                ]
            )
        )
                
        pvalue = (null_dist >= stat).sum() / reps
    
        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
    
        #return stat, pvalue, null_dist
        return stat, pvalue
 

    