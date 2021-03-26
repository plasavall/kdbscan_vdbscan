#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:37:25 2017

V-DBSCAN: Variable-Density-Based Spatial Clustering of Applications with Noise

@author: Eduardo Pla-Sacristan
"""

#============================       IMPORTS       ============================#
import numpy as np
from copy import deepcopy as copy
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import termplot as tplt

class VDBSCAN():
    
    def __init__(self,
                 kappa = 0.0017,
                 max_level=60,
                 max_non_changes=8,
                 metric = 'default', # euclidean
                 isol = True):
        self.kappa = kappa
        self.max_level       = max_level
        self.max_non_changes = max_non_changes
        self.isol            = isol
        if metric == 'default':
            self.metric      = metric
            self.dist_p2p    = euclidean
        else:
            self.metric      = metric
            self.dist_p2p    = metric
            
            
    def fit(self,X,
                 eps_0='auto', 
                 eta=0.1,
                 verbose=1):
        if verbose:
            print('-------------------------------------------------------')
            print('VDBSCAN Algorithm')
            print('-------------------------------------------------------')
            print(' Feature Matrix -> ' + str(X.shape[0]) + 'x' + str(X.shape[1]))
            print('-------------------------------------------------------')
            print(' - Kappa   = ' + str(self.kappa))
            print(' · Eps_0   = ' + str(eps_0))
            print(' · Eta_Eps = ' + str(eta))
            print('-------------------------------------------------------')
            print('Hierarchical iterations in progress...')
        self.eta        = eta
        # Compute eps_0 as 20% of the mean distances between points in dataset
        if eps_0 == 'auto':
            c = np.mean(X,axis=0)
            self.eps = (0.2 * np.mean(self.dists_p2set(c,X)))
        else:
            self.eps = eps_0
        self.n_clusters = 1 
        self.size_dataset = X.shape[0]
        self.y = np.zeros(X.shape[0]).astype(int)
  
        finished = False
        current_level = 0
        non_changes = 0
        ncluster_ev = [1]
        while not(finished):
            y_new = copy(self.y)
            self.eps = self.eps * (1 - self.eta)
            current_level += 1
            if verbose > 1:
                print('Current level: ' + str(current_level) + '/' + str(self.max_level) + \
                      ' - eps = ' + str(self.eps) + ' ')
            elif verbose > 2:
                print()
                print('\n############################################################')
                print('Current level: ' + str(current_level) + '/' + str(self.max_level) + \
                      ' - eps = ' + str(self.eps) + ' ')
                if non_changes == 0:
                    if current_level > 1:
                        print('Structure was altered in the last level!')
                elif non_changes == 1:
                    print('Structured unchanged in the last level.')
                else:
                    print('Structure unchanged in the last ' + str(non_changes) + ' levels.')
                print()
            
            for i in range(self.n_clusters):
                if verbose > 2:
                    print('Clusters analysed: ' + str(i) + '/' + \
                              str(self.n_clusters) + ' - ' + \
                              str(100 * i / self.n_clusters) + '%')
                    
                Xcluster, this_idx = get_cluster(X = X, labels = self.y, clusterID = i)
                
                if self.metric == 'default':   
                    db = DBSCAN(eps=self.eps)
                else:                        
                    db = DBSCAN(eps=self.eps, metric = self.dist_p2p)
                db.fit(Xcluster)
                
                this_n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                
                if this_n_clusters > 1:
                    if self.isol:
                        this_labels = self.separation_criterion(X = Xcluster,
                                                               labels = sort_by_size(db.labels_),
                                                               isolation = self.isolation(i,X,self.y),
                                                               kappa = self.kappa,
                                                               verbose = verbose)
                    else:
                        this_labels = self.separation_criterion(X = Xcluster,
                                                               labels = sort_by_size(db.labels_),
                                                               kappa = self.kappa,
                                                               verbose = verbose)
                    this_labels = get_new_labels(this_labels,y_new)
                    y_new[this_idx]      = this_labels
                    y_new = sort_by_size(y_new)
            if verbose > 2:
                print('Clusters analysed: ' + str(i+1) + '/' + str(self.n_clusters) + ' - 100%')
            n_clusters_before = len(set(y_new)) - (1 if -1 in y_new else 0)
            y_new = sort_by_size(y_new)
            y_new = prune_clusters(thisX = X, labels = y_new)
            y_new = sort_by_size(y_new)
            ## EVALUATE STOP CONDITION
            if np.array_equal(self.y, y_new):
                non_changes += 1
                if verbose > 1:
                    print('Unchanged levels: ' + str(non_changes))
                if non_changes >= self.max_non_changes:
                    finished = True
            else:
                self.y = copy(y_new)
                self.n_clusters = len(set(self.y)) - (1 if -1 in self.y else 0)
                non_changes = 0
            # Stopping criterion (MAX LEVEL)
            if current_level >= self.max_level:
                finished = True
            if verbose:
                n_noise = np.sum(self.y==-1)    
                ncluster_ev.append(self.n_clusters)
                if verbose > 2:
                    print('\nNumber of clusters after level: ' +\
                          str(self.n_clusters) + ' // ' +\
                          str(n_clusters_before - self.n_clusters) +\
                          ' small clusters pruned.')
                    print('Noise samples after level: ' + str(n_noise) +\
                          '(' + str(100*n_noise/self.y.shape[0]) + '%)')
                    print('\n############################################################')
                    
                elif verbose > 1:
                    print('Nº of clusters after level: ' +\
                          str(self.n_clusters) + ' // ' +\
                          str(n_clusters_before - self.n_clusters) +\
                          ' pruned // ' +\
                          'Noise at ' + str(100*n_noise/self.y.shape[0]) + '%')
        # Set final labels (order by size and remove gaps)
        self.labels_ = sort_by_size(self.y)
        if verbose > 1:
            print('\n############################################################')
            print('          Evolution of number of clusters:')
            print('############################################################\n')
            tplt.plot(ncluster_ev, plot_height=15, plot_char='.')
            print()
        if verbose:
            print('-------------------------------------------------------')
            print('Algorithm complete!')
            print('-------------------------------------------------------')
            print('-------------------------------------------------------')
        return self

    def separation_criterion(self, X, labels, isolation = 0.5, kappa = 0.50, verbose = False):
        success = False
        # Starts with all samples at same cluster (original)
        labels_new = np.zeros(labels.shape).astype(int)
        # Mark noise as noise
        labels_new[labels == -1] = -1 
        
        list_labels = np.array([pair[0] for pair in Counter(labels).most_common()])
        list_labels = list_labels[::-1]
        list_discarded = [labels!=-1]
        
        for label in list_labels:
            if label >= 0:
                # Get current cluster
                Ci = X[labels == label]
                # Get rest of the cluster
                this_list = copy(list_discarded)
                this_list.append(labels!=label)
                Cu = X[np.all(this_list, axis = 0)]
                # Check if last cluster standing
                if Cu.shape[0] == 0:
                    # Eliminate sub-cluster from the comparison
                    list_discarded = this_list
                    # Add sub-cluster to collection
                    labels_new[labels == label] = label + 1
                else:     
                    # Separation criterion
                    if self.dist_cl2cl(Ci,Cu) > self.MInt(Ci,Cu,
                                                             isolation = isolation,
                                                             kappa = kappa):
                        # Mark separation as success (at least one split cluster)
                        success = True
                        # Eliminate sub-cluster from the comparison
                        list_discarded = this_list
                        # Add sub-cluster to collection
                        labels_new[labels == label] = label + 1                        
        if success:
            return labels_new
        else:
            return np.zeros(labels.shape).astype(int)
    
    def isolation(self,clusterID,X,labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        if n_clusters == 1:
            return 1.0
        else:
            this_center  = np.mean(X[labels==clusterID], axis=0)
            other_center = np.mean(X[labels!=clusterID], axis=0)
            dist = self.dist_p2p(this_center,other_center)
            
            return dist

    def intracluster_distance(self,X,k = 4):
        X = X[:,:2]
        if X.shape[0] < (k+1):
            k = X.shape[0]-1
        if self.metric == 'default':
            n_nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm='ball_tree').fit(X)
        else:
            n_nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm='ball_tree', metric=self.dist_p2p).fit(X)
        distances, indices = n_nbrs.kneighbors(X)
        indices = indices[:,1:]
        edges = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            p = X[i]
            nbrs = X[indices[i]]
            edges[i] = np.mean(self.dists_p2set(p,nbrs))
        # Return the worst case (maximum distance) normalized and clipped to 1
        return min((1.0 * np.max(edges) / 1000), 1)
    
    def tau(self,cardinality, isolation = 1.0, kappa = 0.50): 
        return 1.0 * (isolation * kappa) / (1.0 * cardinality / self.size_dataset)
    
    def MInt(self,X1,X2, isolation = 1.0, kappa = 0.50):        
        int1 = self.intracluster_distance(X1) + self.tau(X1.shape[0], isolation = isolation, kappa = kappa)
        int2 = self.intracluster_distance(X2) + self.tau(X2.shape[0], isolation = isolation, kappa = kappa)
        return min(int1, int2)
    
    def dists_p2set(self,p,coords):
        dists = np.zeros(coords.shape[0])
        for i,pc in enumerate(coords):
            dists[i] = self.dist_p2p(p,pc)
        return dists
    
    def dist_p2cl(self,p, coords, return_closest = False):
        dist = 10000
        p_closest = None
        for i, pc in enumerate(coords): 
            this_dist = self.dist_p2p(p,pc)
            if this_dist < dist:
                dist = this_dist
                p_closest = pc
        
        if return_closest:
            return dist, p_closest
        else:
            return dist
    
    def dist_cl2cl(self,c1, c2):
        centroid = np.mean(c1, axis=0)
        d_aux, verge2 =  self.dist_p2cl(centroid, c2, return_closest = True)
        if verge2 is None:
            return d_aux
        else:
            return self.dist_p2cl(verge2,c1)

def prune_clusters(thisX, labels, min_size = 2):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(n_clusters):
        size_tot = np.sum(labels==i)
        if size_tot < min_size:
            labels[labels==i] = -1
    return labels

def get_cluster(X,labels,clusterID):
    coords = X[labels == clusterID]
    idx = (labels == clusterID).nonzero()[0]
    return coords,idx

# Updates the freshly obtained labels in order for them to not yiled merging
# problems with old ones.:
#   -The label of the divided cluster is kept for one of the subclusters
#   -The rest of the subclusters get labels with higher cardinality than the
#    ones currently in the set
def get_new_labels(sub_labels,global_labels):
    n_subclusters = get_n_clusters(sub_labels)
    highest_label = np.max(global_labels)
    sub_labels_list = np.unique(sub_labels)
    sub_labels_list = sub_labels_list[sub_labels_list != -1]
    new_labels = copy(sub_labels)
    new_labels_list = 1 + highest_label + np.arange(n_subclusters)
    for i,this_label in enumerate(sub_labels_list):
        new_labels[sub_labels==this_label] = new_labels_list[i]
    return new_labels

def get_n_clusters(y):
    return len(set(y)) - (1 if -1 in y else 0)
    
def sort_by_size(disordered_labels):
    labels = copy(disordered_labels)
    new_labels = -np.ones(labels.shape).astype(int)
    most_common_labels = Counter(disordered_labels).most_common()
    new_label = 0
    for this_label_tuple in most_common_labels:
        this_label = this_label_tuple[0]
        if this_label != -1:
            new_labels[labels==this_label] = new_label
            new_label += 1

    return new_labels