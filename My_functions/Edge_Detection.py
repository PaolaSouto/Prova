#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:17:44 2020

Algorithms for edge detection

@author: paolasouto
"""

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


def Cany_Edge_Detection(img, lowThresholdRatio=0.02, highThresholdRatio=0.07):
    
    ## Solbel vertical filter
    
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)    
    
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
            
    highThreshold = Z.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = Z.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(Z >= highThreshold)
    zeros_i, zeros_j = np.where(Z < lowThreshold)
    
    weak_i, weak_j = np.where((Z <= highThreshold) & (Z >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    
    
    
    M, N = Z.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (Z[i,j] == weak):
                try:
                    if ((Z[i+1, j-1] == strong) or (Z[i+1, j] == strong) or (Z[i+1, j+1] == strong)
                        or (Z[i, j-1] == strong) or (Z[i, j+1] == strong)
                        or (Z[i-1, j-1] == strong) or (Z[i-1, j] == strong) or (Z[i-1, j+1] == strong)):
                        Z[i, j] = strong
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
    return Z





def cMST(a, dist_arr, edges):
    """Return the full spanning tree, with points, connections and distance
    : a - point array
    : dist - distance array, from _e_dist
    : edge - edges, from mst
    """
    p_f = edges[:, 0]
    p_t = edges[:, 1]
    d = dist_arr[p_f, p_t]
    n = p_f.shape[0]
    dt = [('Orig', '<i4'), ('Dest', 'i4'), ('Dist', '<f8')]
    out = np.zeros((n,), dtype=dt)
    out['Orig'] = p_f
    out['Dest'] = p_t
    out['Dist'] = d
    return out


def _e_dist(a):
    """Return a 2D square-form euclidean distance matrix.  For other 
    :  dimensions, use e_dist in ein_geom.py"""
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = a - b
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
    #d = np.triu(d)
    return d 


def mst(W): #, copy_W=True
    """Determine the minimum spanning tree for a set of points represented
    :  by their inter-point distances... ie their 'W'eights
    :Requires:
    :--------
    :  W - edge weights (distance, time) for a set of points. W needs to be
    :      a square array or a np.triu perhaps
    :Returns:
    :-------
    :  pairs - the pair of nodes that form the edges
    """
    #if copy_W:
       # W = W.copy() 
    if W.shape[0] != W.shape[1]:
        raise ValueError("W needs to be square matrix of edge weights")
    Np = W.shape[0]
    pairs = []
    pnts_seen = [0]  # Add the first point                    
    n_seen = 1
    # exclude self connections by assigning inf to the diagonal
    diag = np.arange(Np)
    W[diag, diag] = np.inf
    # 
    while n_seen != Np:                                     
        new_edge = np.argmin(W[pnts_seen], axis=None)
        new_edge = divmod(new_edge, Np)
        new_edge = [pnts_seen[new_edge[0]], new_edge[1]]
        pairs.append(new_edge)
        pnts_seen.append(new_edge[1])
        W[pnts_seen, new_edge[1]] = np.inf
        W[new_edge[1], pnts_seen] = np.inf
        n_seen += 1
    return np.vstack(pairs)
 
 
def plot_mst(a, pairs):
    """plot minimum spanning tree test """
    plt.figure(figsize=(20,20))
    plt.scatter(a[:, 0], a[:, 1])
    ax = plt.axes()
    ax.set_aspect('equal')
    for pair in pairs:
        i, j = pair
        plt.plot([a[i, 0], a[j, 0]], [a[i, 1], a[j, 1]], c='r')
    lbl = np.arange(len(a))
    for label, xpt, ypt in zip(lbl, a[:,0], a[:,1]):
        plt.annotate(label, xy=(xpt, ypt), xytext=(2,2), size=8,
                     textcoords='offset points',
                     ha='left', va='bottom')
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
