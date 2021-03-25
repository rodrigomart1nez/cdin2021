# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
#%% Generar los datos de prueba
np.random.seed(103)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
Y = [0]*20 + [1]*20

#%% Visualizar los puntos
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

#%% Crear el modelo de clasificación
modelo = svm.SVC(kernel='linear')

modelo.fit(X,Y)

Yhat = modelo.predict(X)

#%% Dibujar el plano de separación
w = modelo.coef_[0]
m = -w[0]/w[1]
xx = np.linspace(-5,5)
yy = m*xx-(modelo.intercept_[0]/w[1])

vs = modelo.support_vectors_

b = vs[0]
yy_down = m*xx + (b[1]-m*b[0])

b = vs[-1]
yy_up = m*xx + (b[1]-m*b[0])



plt.plot(xx,yy,'k-')
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(vs[:,0],vs[:,1],s=80,facecolors='k')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
plt.show()

print('Accuracy : linear', accuracy_score(Y,Yhat))













