# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 06:17:04 2018

@author: Imen
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)
#X is the generated samples
#y is the labels of each sample
X, y=make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], 
                                         [2, -3],[1, 1]], cluster_std=0.9 )
plt.scatter(X[:,0],X[:,1],marker='.')
kmeans=KMeans(init="k-means++",n_clusters=4, n_init=12)
kmeans.fit(X)
k_means_labels=kmeans.labels_
k_means_cluster_centers=kmeans.cluster_centers_
#visual plot
#initialize the plot with size
fig=plt.figure(figsize=(6,4))

#colors uses a color map wich will produce an array of colors 
#based on the number of labels there are.we use set(k_means_labels) to get the unique labels

colors=plt.cm.Spectral(np.linspace(0,1,4))

plott=fig.add_subplot(1,1,1)

for k, col in zip(range(4) , colors):
    my_members= (k_means_labels==k)
    #print (my_members)
    
    cluster_center=k_means_cluster_centers[k]
    
    plott.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col,marker='.')
    plott.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)







