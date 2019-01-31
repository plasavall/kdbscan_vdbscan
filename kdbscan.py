#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:54:00 2018

K-DBSCAN: Kernel-Density-Based Spatial Clustering of Applications with Noise

@author: Eduardo Pla-Sacristan
"""

#============================       IMPORTS       ============================#
import numpy as np
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from copy import deepcopy as copy
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
#=============================================================================#
#
#                           USABLE CLASSES
#
#=============================================================================#
# Complete KDBSCAN (KDE + Assignment Algorithm)
#=============================================================================#

class KDBSCAN():
    
    def __init__(self, h = 0.35, t = 0.25, metric = euclidean):
        self.h = h
        self.t = t
        self.dist_p2p = metric
        
    def fit(self, X, 
                  eps_0 = 'auto',
                  eta = 0.1,
                  relevant_features = 'all',
                  return_kde = False,
                  verbose = 0):
        self.X = X
        if verbose:
            print '-------------------------------------------------------'
            print 'KDBSCAN Algorithm'
            print '-------------------------------------------------------'
            print ' Feature Matrix -> ' + str(X.shape[0]) + 'x' + str(X.shape[1])
            print '-------------------------------------------------------'
            print ' - h     = ' + str(self.h)
            print ' - t     = ' + str(self.t)
            print ' · eps_0 = ' + str(eps_0)
            print ' · eta   = ' + str(eta) 
            print '-------------------------------------------------------'
            print 'Kernel Density Estimation in progress...'    
        # Declare features used for clustering assignment
        if relevant_features == 'all':
            relevant_features = range(X.shape[1])        
        # Remove repeated samples with ALL features
        Xu = np.unique(X,axis=0)
        # Remove irrelevant features prior to clustering
        Xu = Xu[:,relevant_features]
        # OBtain the Kernel Density Estimation
        Z, xmin, xmax, ymin, ymax = get_kde_space(Xu,BW = self.h)
        # Filter out weak peaks
        centroid_idx = peak_local_max(Z.T,min_distance=5, threshold_rel=0.25)
        # Obtain the valid peaks based on prominence (independence of peak)
        kept, prominences = get_valid_peaks(centroid_idx,Z,t=self.t)
        # Save all initial centroids (pre t)
        centroids_all = get_coords_from_idx(centroid_idx, xmin, xmax, ymin, ymax)
        # Keep only the centroids that satisfy t
        centroid_idx = centroid_idx[kept,:]
        if np.sum(kept) == 1:
            self.y = np.zeros(X.shape[0])
        else:
            # Get the coordinates of the valid peaks
            centroids = get_coords_from_idx(centroid_idx, xmin, xmax, ymin, ymax)
            if verbose:
                print 'Kernel Density Estimation in complete!'
                print 'Valid Centroids: '+str(np.sum(kept))+'/'+str(len(kept))
                print '-------------------------------------------------------'
                print 'Assignment Algorithm in progress...'
            # Assign a subregion label to each sample (considering ALL peaks)
            assignment_phase = AssA(centroids.shape[0], metric = self.dist_p2p)
            assignment_phase.fit(Xu,centroids, eps_0 = eps_0, eta=eta, verbose = verbose)
            y = assignment_phase.labels_
            # Sort by size (and avoid numbering gaps)
            y = sort_by_size(y)
            # Assign labels to all samples (input dataset)
            self.y = assign_labels(X[:,relevant_features],Xu,y)
        if verbose:
            print 'Assignment Algorithm complete!'
            print '-------------------------------------------------------'
        if return_kde:
            return {'X':self.X,
                    'y':self.y,
                    'Z':Z,
                    'centroids':centroids_all,
                    'kept':kept,
                    'extent':[xmin, xmax, ymin, ymax]}
        else:
            return self

#=============================================================================#
# Define a class for the assignment algorithm (so it can be used independently)
#=============================================================================#
        
class AssA():
    
    def __init__(self,n_clusters, metric = euclidean):
        self.n_clusters = n_clusters
        self.dist_p2p = metric
    
    def fit(self, X, centroids, eps_0 = 'auto', eta = 0.1, MinPts=5, verbose = 0):
        # Initialize parameters
        self.X = X
        self.centroids = self.get_init_points_idx(centroids) # Centroid indexes
        self.unprocessed = np.arange(X.shape[0])
        self.current_cluster = 0
        self.eta = eta
        self.MinPts = MinPts
        
        # Declare initial set of unclassified samples (label = -2)
        y = -2 + np.zeros(X.shape[0])
        # Set initial set of labels (centroids):
        for i in range(self.n_clusters):
            y[self.centroids[i]] = i
        self.update_processed(y)

        # Compute eps_0 as 10% of the mean distances between points in dataset
        if eps_0 == 'auto':
            c = np.mean(X[:,:2],axis=0)
            self.eps = (0.1 * np.mean(self.dists_p2set(c,X)))
        else:
            self.eps = eps_0

        # Perform Assignment Algorithm
        while (np.sum(y==-2) > self.MinPts-1) and (self.eps < 10e6):
            #######################  Cluster expansion  #######################
            # Get the current collection of points for active cluster (queue)
            Q = list(np.where(y==self.current_cluster)[0]) 
            if verbose > 1:
                if self.current_cluster == 0:
                    print '--------------------------------'
                print
                print 'Current EPS:     ' + str(self.eps)
                print 'Nº unclassified: ' + str(np.sum(y==-2))
                print 'Current cluster: ' + str(self.current_cluster)
                print 'Size:            ' + str(len(Q))
            
            # Traverse the candidates
            while (len(Q) > 0):
                self.update_processed(y)
                # Obtain next point from the queue
                p = Q[0]
                Q = Q[1:]
                # Asign current label
                y[p] = self.current_cluster
                # Get the Neighborhood of point p
                N = self.get_neighborhood(p)
                # Check if neighborhood has enough different points
                if len(N) >= self.MinPts:
                    Q = unique_queue(Q,N)
            ###################################################################
            # Update the current cluster
            self.set_current() 
            self.update_processed(y)
        
        # Assign the potential remaining points
        # (KNN to reduce computational complexity)
        self.labels_ = assign_labels(self.X,self.X[y!=-2],y[y!=-2])
        
    def update_processed(self,y):
        self.unprocessed = np.where(y==-2)[0]
    
    def get_neighborhood(self,p):
        coord = self.X[p,:]
        N = []
        for candidate in self.unprocessed:
            if self.dist_p2p(coord,self.X[candidate,:]) < self.eps:
                N.append(candidate)
        return N
    
    def get_init_points_idx(self,centroids):
        init_points = []
        for c in centroids:
            init_points.append(self.closest_pt(c,self.X))
        return np.array(init_points)
    
    def dists_p2set(self,p,coords):
        dists = np.zeros(coords.shape[0])
        for i,pc in enumerate(coords):
            dists[i] = self.dist_p2p(p,pc)
        return dists
    
    def closest_pt(self,pt, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - pt)**2, axis=1)
        return np.argmin(dist_2)
    
    def set_current(self):
        self.current_cluster += 1
        if self.current_cluster >= self.n_clusters:
            self.current_cluster = 0
            self.eps = self.eps * (1 + self.eta)
            
    def get_label(self,p):
        idx = self.closest_pt(p,self.X)
        return self.labels_[idx]

#=============================================================================#
#                           LOCAL FUNCTIONS
#=============================================================================#

def unique_queue(Q,N):
    Q.extend(N)
    return list(np.unique(np.array(Q)))

def assign_valid_subregions(core_labels,centroids, centroid_labels, kept):
    y = copy(core_labels)
    for i in range(centroids.shape[0]):
        if not kept[i]:
            others = np.array([x for j,x in enumerate(centroids) if j!=i])
            c_label_cand = np.array([x for j,x in enumerate(centroid_labels) if j!=i])
            others = np.array([x for j,x in enumerate(centroids) if kept[j]])
            c_label_cand = np.array([x for j,x in enumerate(centroid_labels) if kept[j]])
            closest = closest_pt(centroids[i,:],others)
            y[y==i] = c_label_cand[closest]
    return y

def get_valid_peaks(centroid_idx,Z, t = 0.39):
    prominences = []
    for idx in range(centroid_idx.shape[0]):
        peak1 = centroid_idx[idx]
        intens_peak = Z[peak1[1], peak1[0]]
        if intens_peak > 0.999999999999999:
            prominences.append(1.0)
        elif intens_peak < 1:      
            prom, prom_n = prominence(peak1,Z.T)
            prominences.append(prom_n)
    prominences = np.array(prominences)
    return (prominences > t), prominences

def get_kde_space(X, BW = 0.35, step = 100j):
    # Obtain the gaussian KDE
    XT = X.T
    XT = XT[::-1]
    kernel = gaussian_kde(XT, bw_method = BW)
    # Get limits of space in coordinates
    m1 = X[:,1]
    m2 = X[:,0]
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    # Get grid
    Xo, Yo = np.mgrid[xmin:xmax:step, ymin:ymax:step]
    positions = np.vstack([Xo.ravel(), Yo.ravel()])
    # Assign intensity values to the grid
    Z = np.reshape(kernel(positions).T, Xo.shape)
    # Smooth
    Z = np.power(Z,0.5)
    Z = Z / np.max(Z)
    return Z, xmin, xmax, ymin, ymax

#=============================================================================#

def prominence(peak,Z):
    # Get intesity of peak
    intens_peak = Z[peak[0], peak[1]]
    # Get points in the kde space with the same intensity
    cut_nodes = get_cut(Z, intens_peak)
    # Retrieve the closest same level point
    same_level = cut_nodes[closest_pt(peak,cut_nodes)]
    # Get intensity of the intermediate points between the peak and said point
    intensity_line = get_intermediate_intensity(peak,same_level,Z)
    # Get the intensity of the Key Col (lowest point in the valley)
    intens_key_col = np.min(intensity_line)
    # Obtain the prominence of the peak
    prom   = intens_peak - intens_key_col
    # Obtain the relative prominence (w.r.t. the height of the peak)
    prom_n = prom / intens_peak
    # Return both the absolute and the relative prominence
    return prom, prom_n

#=============================================================================#

def get_cut(Z,intensity, tolerance = 0.015):
    mask = np.logical_and((Z > intensity), (Z < (intensity+tolerance)))
    x, y = np.where(mask)
    return np.concatenate((x[:,np.newaxis],y[:,np.newaxis]), axis=1)
    
def get_intermediate_intensity(p1,p2,Z):
    line = np.round(get_intermediates(p1,p2)).astype(int)
    return get_maxima_intensity(line,Z)

#=============================================================================#
    
def get_intermediates(p1,p2, n_points = 24):
    x1,y1 = p1
    x2,y2 = p2
    m = 1.0 * (y2-y1) / (10e-14 + x2 - x1)
    b = y1 - (m * x1)
    step = 1.0 * (x2 - x1) / n_points
    if step == 0:
        step = 1.0 * (y2 - y1) / (n_points)
        y = np.arange(y1 + step, y2, step)
        x = np.zeros(y.shape) + x1
    else:
        x = np.arange(x1 + step, x2, step)
        y = m * x + b
    return np.concatenate((x[:,np.newaxis],y[:,np.newaxis]),axis=1)

def get_maxima_intensity(maxima_idx,Z):
    intensities = []
    for i in range(maxima_idx.shape[0]):
        intensities.append(Z[maxima_idx[i,0],maxima_idx[i,1]])
    return np.array(intensities)

#=============================================================================#

def get_coords_from_idx(centroids, xmin, xmax, ymin, ymax):
    resulting = np.zeros(centroids.shape)
    if resulting.ndim > 1:
        for i in range(centroids.shape[0]):
            resulting[i,0] = ymin + (1.0 * centroids[i,0]/100) * (ymax - ymin)
            resulting[i,1] = xmin + (1.0 * centroids[i,1]/100) * (xmax - xmin)
    else:
        resulting[0] = ymin + (1.0 * centroids[0]/100) * (ymax - ymin)
        resulting[1] = xmin + (1.0 * centroids[1]/100) * (xmax - xmin)
        resulting = resulting[np.newaxis,:]
    return resulting

def closest_pt(pt, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - pt)**2, axis=1)
    return np.argmin(dist_2)

def assign_labels(X_total,X_pred,y_pred):
    knn = KNeighborsClassifier(n_neighbors=1)    
    knn.fit(X_pred, y_pred)
    return knn.predict(X_total)

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

#=============================================================================#
            
def plot_kdbscan_results(kde, fig_num=1, plot_samples=True):
    import matplotlib.pyplot as plt

    if kde is not(None):
        kept = kde['kept']
        centroids = kde['centroids']
        c1 = centroids[:,1]
        c2 = centroids[:,0]
        
        # Create the figure
        fig = plt.figure(fig_num, figsize=(25,25))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        # Plot Gaussian KDE
        ax.imshow(np.rot90(kde['Z']), cmap=plt.cm.gist_earth_r, extent=kde['extent'])

        # Plot Peaks (Valid in Magenta)
        for idx in range(centroids.shape[0]):
            if kept[idx]:
                ax.plot(c1[idx], c2[idx], 'm^', markersize=20)
                
            else:
                ax.plot(c1[idx], c2[idx], 'k^', markersize=18)
    # Plot all sampels with subregion labels
    if plot_samples:
        X = kde['X']
        y = kde['y']
        c = np.unique(y)
        for i in c:
            a = X[y==i]
            color = np.random.rand(1,3)
            plt.scatter(a[:,1],a[:,0],c=color)
