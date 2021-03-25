#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 07:11:12 2020

@author: deloga
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
datos = pd.read_csv('../Data/Position_Salaries.csv')

x = datos['Level'].values.reshape(-1,1)
y = datos['Salary'].values.reshape(-1,1)

plt.scatter(x,y)
plt.show()
#%% Aplicar modelo lineal con sklearn
# \hat{y} = \beta_1 x + \beta_0

model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x, y_pred, color ='r')
plt.show()

# Aplicar metricas
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)

print('RMSE: ' + str(rmse))
print('R2: ' + str(r2))

#%% Aplicar otro modelo de datos polinomio grado 2
#  \hat{y} = \beta_2 x^2 + \beta_1 x + \beta_0

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4, include_bias = False)
x_poly = poly.fit_transform(x)
print(x)
print(x_poly)


model.fit(x_poly, y)
y_pred = model.predict(x_poly)

plt.scatter(x,y)
plt.plot(x, y_pred, color = 'r')

# Aplicar metricas
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print('RMSE: ' + str(rmse))
print('R2: ' + str(r2))

#%% predecir un vector de niveles
#% x_new=[2.5, 8.5, 11, 13, 15]

#% Graficar los salarios de los valores de niveles en x








