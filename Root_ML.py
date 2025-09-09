#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:35:07 2025

@author: keiragupta
"""
# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Import dataset and subset it
root_df = pd.read_csv('Desktop/THESIS/Root_Python.csv')

root_df_subset = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Bankfull', 'Scour_Pool_DS_', 'Culvert_Material']]

# Replace all "NA' values with nan
root_df_subset = root_df_subset.replace('NA', np.nan)

root_df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove all rows with nan values
root_df_subset = root_df_subset.dropna()

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
root_df_subset['Culvert_Material'] = root_df_subset['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].apply(lambda val: 1 if val == 'Yes' else 0)

# Establish seed for random state (ensures reproducibility)
SEED = 42

# Model 1: Classification model for scour presence/absence

# Define X and y for splitting dataset
X = root_df_subset.drop('Scour_Pool_DS_', axis = 1)
y = root_df_subset['Scour_Pool_DS_']

# Split dataset into train (75%) and test (25%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED, stratify = y)

"""
# Instantiate simple decision tree classifier
dt = DecisionTreeClassifier(max_depth = 3, criterion = 'gini', random_state = SEED)

# Instantiate cross validation score to understand under vs over fitting
acc_CV = cross_val_score(dt, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs = -1)

# Fit the model to the data and make predictions on both the train and test datasets
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

CV_acc = acc_CV.mean() # Compute K-fold cross validation accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Compare all 3 accuracies
print(f'CV accuracy: {CV_acc}')
print(f'Train accuracy: {train_acc}')
print(f'Test accuracy: {test_acc}')

# The training accuracy is much higher than the CV and the test accuracy, likely suggesting high variance
"""
rf = RandomForestClassifier(max_depth = 4, max_features = 'log2', min_samples_leaf = 0.1, n_estimators = 300, random_state = SEED)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_pred, y_test)

#print(rf.get_params())

print(f'RF Accuracy: {rf_acc}')

#params_rf = {'n_estimators': [300, 400, 500], 'max_depth': [4, 5, 6], 'min_samples_leaf': [0.1, 0.15, 0.2], 'max_features': ['log2', 'sqrt']}

#grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs = -1)

#grid_rf.fit(X_train, y_train)

#best_hyp = grid_rf.best_params_

#best_model = grid_rf.best_estimator_
#y_pred = best_model.predict(X_test)

#acc = accuracy_score(y_test, y_pred)

#print(f'Best hyperparameters: {best_hyp}')
#print(f'Test set accuracy of rf: {acc}')



"""
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='pink')
plt.title('Features Importances')
plt.show()
"""

"""
NOTES
- Will need to load in lateral scour severity
- Mess with parameters
- For initial CART test - max depth was too high?
- Test other types of ensembles? (boosting?)
"""