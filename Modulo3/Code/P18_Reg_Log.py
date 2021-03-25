
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score)

#%% Importar datos
data = pd.read_csv('../Data/ex2data2.txt',header=None)
#data = pd.read_csv('../Data/ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Preparación de los datos (crear el polinomio)
ngrado = 7
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)

#%% Crear y entrenar la regresión logística
modelo = linear_model.LogisticRegression(C=1e20)
modelo.fit(Xa,Y)

Yhat = modelo.predict(Xa)

#%% Evaluar el modelo
accuracy_score(Y,Yhat)
#%%
precision_score(Y,Yhat)
#%%
recall_score(Y,Yhat)

#%% Visualizar la frontera de separación
h = 0.01
xmin,xmax,ymin,ymax = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Xa_new = poly.fit_transform(Xnew)
Z = modelo.predict(Xa_new)
Z = Z.reshape(xx.shape)

plt.contour(xx,yy,Z)
plt.scatter(X[0],X[1],c=Y)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()

#%% Visualizar el overfitting
W = modelo.coef_
plt.bar(np.arange(len(W[0])),W[0])
plt.show()















