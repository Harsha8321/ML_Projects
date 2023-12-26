import pandas as pd
import numpy as np

import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('/Users/harsha/Documents/INTERN/iNeuron/Project')
data_df = pd.read_csv('ENB2012_data.csv')
print(data_df.head())

df_cols = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area','Overall Height', 'Orientation',
'Glazing Area','Glazing Area Distribution','Heating Load', 'Cooling Load','Unnamed: 10', 'Unnamed: 11']
data_df.columns = df_cols

# Dropping irrelevant columns
data = data_df.drop(columns=['Unnamed: 10', 'Unnamed: 11'])

# Splitting the dataset into Dependent and Independent Variables
X = data.drop(columns=['Heating Load', 'Cooling Load'])  # Features/Independent Variable
Y = data[['Heating Load', 'Cooling Load']]  # Targets/Dependent Variable

# Splitting the dataset into trainig and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, Y_train)

import pickle
os.chdir('/Users/harsha/Documents/INTERN/iNeuron/Project/Website')
pickle.dump(rf_model,open('Model.pkl','wb'))
