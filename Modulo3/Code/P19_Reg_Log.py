

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score)

#%% Importar datos
#data = pd.read_csv('../Data/ex2data1.txt',header=None)
data = pd.read_csv('../Data/ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Buscar el grado del polinomio optimo
modelo = linear_model.LogisticRegression(C=1e20)
grados = np.arange(1,18)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Reca = np.zeros(grados.shape)
Nvar = np.zeros(grados.shape)

for ngrado in grados:
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
plt.xlabel('Grado Polinomio')
plt.ylabel('% aciertos')
plt.legend(('Accuracy','Precision','Recall'),loc='best')
plt.grid()
plt.show()

plt.plot(grados,Nvar)
plt.grid()
plt.show()

#%% seleccionar el modelo deseado
ngrado = 8
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)
modelo = linear_model.LogisticRegression(C=1e20)
modelo.fit(Xa,Y)
Yhat = modelo.predict(Xa)

W = modelo.coef_[0]
plt.bar(np.arange(len(W)),np.abs(W))
plt.grid()
plt.show()

#%% Optimización versión 1. Seleccionar coeficientes > umbral
umbral = 50
ind = np.abs(W)>umbral
Xa_simplificada = Xa[:,ind]
modelo1 = linear_model.LogisticRegression(C=1e20)
modelo1.fit(Xa_simplificada,Y)
Yhat1 = modelo1.predict(Xa_simplificada)

#%% Medidas modelo inicial
accuracy_score(Y,Yhat)

#%% Medidas modelo optimizado 1
accuracy_score(Y,Yhat1)











