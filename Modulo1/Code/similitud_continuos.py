#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:59:18 2021

@author: gaddiel
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.spatial.distance as sc


#%% Generar datos 

x=np.array([[2,3],[20,30],[-2,-3],[2,-3]])

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.grid()
plt.show()

#%% Matriz de similitud

# Distancia euclideana
D1 = sc.pdist(x,'euclidean')
D1 = sc.squareform(D1)

# Distancia coseno
D2 = sc.pdist(x,'cosine')
D2 = sc.squareform(D2)

# Distancia correlacion
D3 = sc.pdist(x,'correlation')
D3 = sc.squareform(D3)



