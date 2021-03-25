#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:38:14 2021

@author: gaddiel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as skm #metricas de similitud
import scipy.spatial.distance as sc # metricas de distancia


#%% 1.- Importar dataset de test de peliculas

data = pd.read_excel('../Data/test_peliculas_cdin_2021.xlsx')

#%% 2.- Limpieza del test

c_inicial = np.array([0,1,2,3,4])
csel = np.arange(7,data.shape[1],3)
c_pelis = np.append(c_inicial, csel)

data_pelis = data.iloc[:,c_pelis]

data_pelis.dropna(subset=['Nombre'], inplace=True)
data_pelis.fillna(0, inplace=True)

data_pelis.reset_index(inplace=True)
data_pelis.drop('index', axis=1, inplace=True)

#%% 3.- Cambiar las calificaciones de estrellas por un me gustó o no me gustó

cnames = list(data_pelis.columns.values[5:])

for col in cnames:
    indx = data_pelis[col] <=3
    data_pelis[col][indx] = 0
    data_pelis[col][data_pelis[col]>3] = 1
    
# 4.- Calcular los indices de similitud en usuarios

arturo = data_pelis[data_pelis['ID']==113]
ricardo = data_pelis[data_pelis['ID']==115]

cf_m = skm.confusion_matrix(arturo.iloc[:,5:].values.flatten(),
                            ricardo.iloc[:,5:].values.flatten())    

sim_simple = (cf_m[0,0] + cf_m[1,1])/np.sum(cf_m)

sim_jackard = (cf_m[0,0])/(np.sum(cf_m)-cf_m[1,1])


#%% utilizando los métodos de la libreria scipy (hay que buscar la documentación)
#   como se calculan los indices de similitud

d1 = sc.matching(arturo.iloc[:,5:].values.flatten(), 
                 ricardo.iloc[:,5:].values.flatten())
d2 = sc.jaccard(arturo.iloc[:,5:].values.flatten(), 
                ricardo.iloc[:,5:].values.flatten())

# scipy.spatial.distance.pdist
datan = data_pelis.iloc[:,5:]

D1 = sc.pdist(datan,'matching')
D1 = sc.squareform(D1)

user = 90

D_user = D1[user]
D_user_sort = np.sort(D_user)
indx_user = np.argsort(D_user)

users_sim_ricardo = data_pelis.loc[indx_user[1:6], 'Nombre']


def search_user_sim(user, data_sim, peliculas, n_sim):
    d_user = data_sim[user]
    d_user_sort = np.sort(d_user)
    indx_user = np.argsort(d_user)
    return peliculas.loc[indx_user[1:n_sim+1],'Nombre']

# Recomendación de peliculas al usuario más parecido

user = 90
User_rodrigo = datan.loc[user]
D_user = D1[user]
D_user_sort = np.sort(D_user)
indx_user = np.argsort(D_user)

user_similar = datan.loc[indx_user[1]]





