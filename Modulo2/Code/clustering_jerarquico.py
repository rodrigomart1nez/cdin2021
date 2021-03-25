#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:00:00 2021

@author: gaddiel
"""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import scipy.spatial.distance as sc
from matplotlib import pyplot as plt
#%% Generar los datos a clasificar

np.random.seed(1000)

a = np.random.multivariate_normal([10,10],[[3,0],[0,3]],size=[100])
b = np.random.multivariate_normal([0,20],[[3,0],[0,3]],size=[100])
c = np.random.multivariate_normal([20,20],[[3,0],[0,3]],size=[100])

X = np.concatenate((a, b, c),)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%% Mostrar un conjunto de puntos formados de un cluster

idx= [161,132,153]
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.scatter(X[idx,0],X[idx, 1],c='r')
plt.show()

#%% Aplicar Clustering jerarquico

Z = hierarchy.linkage(X,metric='euclidean', method= 'ward')

plt.title('Dendograma completo')
plt.xlabel('Indice de la muestra')
plt.ylabel('Distancia o Similaridad')

dn = hierarchy.dendrogram(Z)
plt.show()

#%% Aplicando el algoritmo con 3 grupos
gruposmax=3
grupos = hierarchy.fcluster(Z, gruposmax,criterion='maxclust')

plt.figure()
plt.scatter(X[:,0],X[:,1], c=grupos, cmap=plt.cm.prism)
plt.show()

#%% Criterio del codo

last = Z[-15:,2]
last_rev =last[::-1]
indxs= np.arange(1,len(last_rev)+1)
plt.plot(indxs, last_rev)
plt.show()





