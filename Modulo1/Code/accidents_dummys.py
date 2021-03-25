#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:02:06 2021

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

#%% Elegir columnas con datos de tipo int64
indx = np.array(accidents.dtypes == 'int64')
col_list = list(accidents.columns.values[indx])
accidents_int = accidents[col_list]
del indx

#%% aplicar nuevamente dqr

accidents_int_report = eda.dqr(accidents_int)

#%% eliminar columnas con valores unicos >20

indx = np.array(accidents_int_report.Unique_Values <= 20)
col_list_unique = np.array(col_list)[indx]
accidents_int_unique = accidents_int[col_list_unique]

#%% variables dummies 'Accident_Severity'

dummy1 = pd.get_dummies(accidents_int_unique['Accident_Severity'], prefix='Accident_Severity')

accidents_dummy = pd.get_dummies(accidents_int_unique[col_list_unique[0]], prefix=col_list_unique[0])

for col in col_list_unique[1:]:
    temp = pd.get_dummies(accidents_int_unique[col], prefix = col)
    accidents_dummy = accidents_dummy.join(temp)
    
del temp

#%% aplicar los indices de similitud para 100 primeros datos 

dist_accidents = sc.pdist(accidents_dummy.iloc[0:100,:], 'jaccard')

DIST1 = sc.squareform(dist_accidents)
temp = pd.DataFrame(DIST1)

#%%buscar los accidentes más parecidos al accidente que corresponda al indice 0

D1=temp.iloc[:,0]
D1_sort = np.sort(D1)

D1_index= np.argsort(D1)








