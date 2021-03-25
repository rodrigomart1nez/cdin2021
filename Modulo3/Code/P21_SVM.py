import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score)
import pandas as pd

#%% Importar los datos
data = pd.read_csv('../Data/ex2data2.txt',header=None)
X = data.iloc[:,0:2]
Y = data.iloc[:,2]

#%% Crear y entrenar el modelo SVM
modelo_linear = svm.SVC(kernel='linear')
modelo_poly = svm.SVC(kernel='poly',degree=2)
modelo_rbf = svm.SVC(kernel='rbf')
modelo_linear.fit(X,Y)
modelo_poly.fit(X,Y)
modelo_rbf.fit(X,Y)

Yhat_svm_linear = modelo_linear.predict(X)
Yhat_svm_poly = modelo_poly.predict(X)
Yhat_svm_rbf = modelo_rbf.predict(X)


accuracy_score(Y,Yhat_svm_linear)

print('Accuracy : linear', accuracy_score(Y,Yhat_svm_linear))
print('Accuracy: poly', accuracy_score(Y,Yhat_svm_poly))
print('Accuracy: rbf', accuracy_score(Y,Yhat_svm_rbf))













