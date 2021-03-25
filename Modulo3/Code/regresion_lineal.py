#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 06:53:19 2020

@author: deloga
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Salarios segun el nivel del empleado dentro de una empresa
datos = pd.read_csv('../Data/Position_Salaries.csv')

x = datos['Level'].values.reshape(-1, 1) # necesitamos un array de 2D para SkLearn
y = datos['Salary'].values.reshape(-1, 1)
plt.scatter(x,y)

# MODELO LINEAL a un modelo lineal, para lo cuál aplicamos una regresión lineal valiéndonos de la librería SkLearn.

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()

#%% Una forma de medir la bondad del ajuste es calcular la media de la raíz del error cuadrático (root mean square error) 
# que nos da una medida del error cometido por el modelo. 

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))

#%% Esta transformación la podemos realizar fácilmente son SkLearn gracias PolynomialFeatures().
# A la que le indicamos el grado de la función que queremos obtener. Haciendo uso del método fit_transform() 
# obtendremos el término cuadrático que andamos buscando para nuestro modelo.
#
from sklearn.preprocessing import PolynomialFeatures
 
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
print(x)
print(x_poly)

model.fit(x_poly, y)
y_pred = model.predict(x_poly)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))

#%% Aumentando el grado del polinomio

poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x)
 
model.fit(x_poly, y)
y_pred = model.predict(x_poly)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))

#%% Mucho mejor. ¿Y si seguimos aumentando el grado?

poly = PolynomialFeatures(degree=4, include_bias=False)
x_poly = poly.fit_transform(x)
 
model.fit(x_poly, y)
y_pred = model.predict(x_poly)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))


