#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:14:08 2020

@author: deloga
"""

## Abrir un archivo en spyder y encontrar un modelo
# que más se ajuste a los datos

# 1.- reporte de calidad de datos, EDA
# 2.- Encontrar un modelo para prediccion de alguna
# variable que se les haga interesante

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('petrol_consumption.csv')
df.head()
df.describe()
df.info()

X = df[['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence(%)']]
y = df[['Petrol_Consumption']]

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.3 , random_state = 329)
from sklearn.preprocessing import PolynomialFeatures


X_train_poly = PolynomialFeatures(degree=2).fit_transform(X_train)
X_test_poly = PolynomialFeatures(degree=2).fit_transform(X_test)

from sklearn.preprocessing import StandardScaler
scaled = StandardScaler()

X_scaled_train = pd.DataFrame(scaled.fit_transform(X_train_poly))
X_scaled_test = pd.DataFrame(scaled.fit_transform(X_test_poly))

from sklearn.linear_model import LinearRegression

model_lr = LinearRegression().fit(X_scaled_train,y_train)

predict = model_lr.predict(X_scaled_test)
y_test['result'] = predict


y_test.head()

from sklearn import metrics

print('MAE', metrics.mean_absolute_error(y_test['Petrol_Consumption'],y_test['result']))
print('MSE', metrics.mean_squared_error(y_test['Petrol_Consumption'],y_test['result']))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test['Petrol_Consumption'],y_test['result'])))


