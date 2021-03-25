#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:23:04 2020

@author: deloga
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score)

#%% Importar nuestro dataset

data = pd.read_csv('../Data/ex2data2.txt', header=None)

X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Buscar el grado del polinomio opt
modelo = linear_model.LogisticRegression(C=1e20)
grados = np.arange(1,20)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Reca = np.zeros(grados.shape)
Nvar = np.zeros(grados.shape)

for ngrado in grados:
    # Crear el polinomio
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(X)
    modelo.fit(Xa,Y)
    Yhat = modelo.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y,Yhat)
    Prec[ngrado-1] = precision_score(Y,Yhat)
    Reca[ngrado-1] = recall_score(Y,Yhat)
    Nvar[ngrado-1] = len(modelo.coef_[0])
    
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Reca)
plt.xlabel('Grado del Polinomio')
plt.legend(('Accuracy', 'precision', 'Recall'),loc='best')
plt.grid()
plt.show()


    



