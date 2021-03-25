#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:34:22 2021

@author: gaddiel
"""

import pandas as pd
import string

class CDIN:
    
    #Función data quality report
    
    def dqr(data):
        #%% Lista de variables de la base de datos
        columns = pd.DataFrame(list(data.columns.values),
                               columns= ['Nombres'],
                               index= list(data.columns.values)) 
        #%% Lista de tipos de datos
        data_types = pd.DataFrame(data.dtypes, columns=['Data_Types'])
        
        #%% lista de datos perdidos
        missing_values = pd.DataFrame(data.isnull().sum(),
                                      columns=['Missing_values'])
        #%% lista de valores presentes
        present_values = pd.DataFrame(data.count(), 
                                      columns=['Present Values'])
        
        #%% Lista de valores únicos para cada variable
        unique_values = pd.DataFrame(columns=['Unique_Values'])
        for col in list(data.columns.values):
            unique_values.loc[col] = [data[col].nunique()]
    
        #%% Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min'])
        for col in list(data.columns.values):
            try:
                min_values.loc[col] = [data[col].min()]
            except:
                pass
        
        #%% Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max'])
        for col in list(data.columns.values):
            try:
                max_values.loc[col] = [data[col].max()]
            except:
                pass
        
        #%% Columna 'Categorica' que sea booleana que cuando sea True represente
        #   que la variable es categorica, y False represente que la variable es
        #  numérica
        return columns.join(data_types).join(missing_values).join(present_values).join(unique_values).join(min_values).join(max_values)
    
    #%% Funciones para limpieza de datos
 #%% Remover signos de puntuación

    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            pass
        return x
  #%% Remover digitos
    def remove_digits(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.digits)
        except:
            pass
        return x
    
    #%% 
    # Remover los espacios en blanco
    def remove_whitespace(x):
        try:
            x=''.join(x.split())
        except:
            pass
        return x
    
    #%% Reemplazar texto
    def replace_text(x, to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x
    
    #%% Convertir a mayusculas
    def uppercase_text(x):
        try:
            x=x.upper()
        except:
            pass
        return x
    #%% 
    def lowercase_text(x):
        try:
            x=x.lower()
        except:
            pass
        return x
    #%%
    def fill_ceros_ssn(x):
        try:
            x=str(x)
            while (len(x)<9):
                x='0'+x
        except:
            pass
        return x
    #%%
    def remove_character(x):
        try:
            x = ''.join(ch for ch in x if ch in string.digits)
        except:
            pass
        return x
            