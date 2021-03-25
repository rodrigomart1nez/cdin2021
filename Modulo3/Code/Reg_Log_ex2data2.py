#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 07:04:07 2020

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

#%% Preparar los datos (crear un polinomio)

ngrado = 7
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)

#%% Crear y entrenar el modelo de reg Logítica
modelo = linear_model.LogisticRegression(C=1e10)
modelo.fit(Xa,Y)
Yhat = modelo.predict(Xa)

X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.show()

plt.scatter(X[0],X[1],c=Yhat)
plt.show()

#%% Evaluar el modelo
print(precision_score(Y,Yhat))
print(recall_score(Y,Yhat))
print(accuracy_score(Y,Yhat))

#%% Visualizar 
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.01), np.arange(ymin,ymax,0.01))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Xa_new = poly.fit_transform(Xnew)
Z=modelo.predict(Xa_new)
Z = Z.reshape(xx.shape)

plt.contour(xx,yy,Z)
plt.scatter(X[0],X[1],c=Y)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()


plt.contour(xx,yy,Z)
plt.scatter(X[0],X[1],c=Yhat)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()



