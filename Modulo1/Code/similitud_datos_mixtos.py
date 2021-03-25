#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:24:06 2021

@author: gaddiel
"""
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import scipy.spatial.distance as sc
from CDIN import CDIN as eda

#%% accidents
accidents = pd.read_csv('../Data/Accidents_2015.csv')

#%% dqr
accidents_report = eda.dqr(accidents)

#%% Filtrar la base de datos por columnas específicas:
    
cols = ['Longitude', 'Latitude', 'Number_of_Vehicles', 'Number_of_Casualties',
        'Accident_Severity', 'Day_of_Week']

accidents_mixtos = accidents[cols]

#%% 1.- Variables categoricas

cols_cat = ['Day_of_Week', 'Accident_Severity']

#%% 2.- Variables continuas 

cols_cont = ['Longitude', 'Latitude', 'Number_of_Vehicles',
             'Number_of_Casualties']

#%% 3.- Hacer la transformación a las variables dummys

accidents_dummy = pd.get_dummies(accidents_mixtos[cols_cat[0]],
                                 prefix=cols_cat[0])

for col in cols_cat[1:]:
    temp = pd.get_dummies(accidents_mixtos[col], prefix = col)
    accidents_dummy = accidents_dummy.join(temp)
    
del temp

col_list_cat_dummy = accidents_dummy.columns.to_list()


accidents_mixtos_dummy = accidents_mixtos[cols_cont].join(accidents_dummy)

#%% Estandarizar los datos (standard scaler)

accidents_std = (accidents_mixtos_dummy - 
                 accidents_mixtos_dummy.mean(axis=0))/accidents_mixtos_dummy.std(axis=0)

accidents_std[col_list_cat_dummy] = accidents_std[col_list_cat_dummy]*(1/np.sqrt(2))

#%% Aplicar los indices de similitud a el dataset accidents_std

# 1.- Aplicar indice de similitud a los primeros 100 muestras
#     (euclidean, coseno, correlacion)
# 2.- Buscar los accidentes más parecidos al accidente que ccorresponde al Acc´idente 0
# 3.- comparar los accidentesa mas parecidos al accidente 0 de este archivo con lo que
#     resultó en el archivo pasado




