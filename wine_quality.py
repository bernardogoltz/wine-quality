# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:25:07 2022

@author: bernardogoltz
"""


# importing the libraries / dependencies
import pandas as pd 
import numpy as np
import matplotlib as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# processing the data

path = 'C:/Users/angel/OneDrive/Documentos/Projetos/wine-quality/winequality-red.csv'

wine_dataset = pd.read_csv(path)

# checking the number of rows and columns in the dataset
print(wine_dataset.shape)

# checking if the wine dataset has missing values: 
print(wine_dataset.isna().sum())
print("\nTotal summation of the missing values: {}".format(wine_dataset.isna().sum().sum()))


# Label Binarization 
X = wine_dataset.drop('quality', axis = 1)

Y = wine_dataset['quality'].apply(lambda y_value : 1 if y_value >=7 else 0)

# Train Test Split 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2 , random_state = 3)


# Training The model
model = RandomForestClassifier()
model.fit(X_train , Y_train)

# Model Evaluation 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("\nAccuracy: ",test_data_accuracy)

