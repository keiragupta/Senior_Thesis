#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:35:07 2025

@author: keiragupta
"""
# Note: parts of this code are commented out to reduce computational expense per run

### IMPORT NECESSARY PACKAGES

# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Accuracy tests and other misc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

## Import dataset and subset it
root_df = pd.read_csv('Downloads/Root_Python.csv')

root_df_subset = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Scour_Pool_DS_', 'Culvert_Shape', 'Culvert_Material']]


### DATA PRE-PROCESSING

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
root_df_subset['Culvert_Material'] = root_df_subset['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].apply(lambda val: 1 if val == 'Yes' else 0)

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Color' column
shape_encoder = encoder.fit_transform(root_df_subset[['Culvert_Shape']])

# Convert to DataFrame and concatenate
shape_df = pd.DataFrame(shape_encoder, columns=encoder.get_feature_names_out(['Culvert_Shape']))
root_df_subset = pd.concat([root_df_subset.drop('Culvert_Shape', axis=1), shape_df], axis=1)

# Replace all "NA' values with nan
root_df_subset = root_df_subset.replace('NA', np.nan)

root_df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove all rows with nan values
root_df_subset = root_df_subset.dropna()

# Establish seed for random state (ensures reproducibility)
SEED = 42
"""
### MODEL 1: CLASSIFICATION MODEL FOR SCOUR PRESENCE/ABSENCE

## Split dataset into training and test data
# Define X and y for splitting dataset
X = root_df_subset.drop('Scour_Pool_DS_', axis = 1)
y = root_df_subset['Scour_Pool_DS_']

# Split dataset into train (75%) and test (25%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED, stratify = y)

## Decision Tree Classification model
# Instantiate simple decision tree classifier
dt = DecisionTreeClassifier(max_depth = 4, max_features = 'log2', min_samples_leaf = 0.1, random_state = SEED)

# Instantiate cross validation score to evaluate under vs over fitting
acc_CV = cross_val_score(dt, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs = -1)

# Fit the model to the data and make predictions on both the train and test datasets
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# Calculate accuries of CV test, training predictions, and testing predictions
CV_acc = acc_CV.mean() # Compute K-fold cross validation accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Compare all 3 accuracies
print(f'CV accuracy: {CV_acc}')
print(f'Train accuracy: {train_acc}')
print(f'Test accuracy: {test_acc}')

# Conclusion: The training accuracy is much higher than the CV and the test accuracy, likely suggesting high variance

## Random Forest Classification model
# Instantiate random forest classification model
rf = RandomForestClassifier(max_depth = 4, max_features = 'log2', min_samples_leaf = 0.1, n_estimators = 300, random_state = SEED)
# The hyperparameters used were found using GridSearch cross-validation

# Fit model to training dataset and predict on test data
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate accuracy
rf_acc = accuracy_score(y_pred, y_test)

print(f'RF Accuracy: {rf_acc}')

## GridSearch cross-validation for hyperparameter tuning

# Print current random forest parameters
#print(rf.get_params())

# Dictionary of hyperparameters to test
#params_rf = {'n_estimators': [300, 400, 500], 'max_depth': [4, 5, 6], 'min_samples_leaf': [0.1, 0.15, 0.2], 'max_features': ['log2', 'sqrt']}

# Instantiate GridSearchCV on random forest model and using the params dictionary 
#grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs = -1)

# Fit to training data
#grid_rf.fit(X_train, y_train)

# Define the best (most accurate) hyperparameters
#best_hyp = grid_rf.best_params_

# Use the best (most accurate hyperparameters) to make predictions on test data
#best_model = grid_rf.best_estimator_
#y_pred = best_model.predict(X_test)

# Calculate accuracy
#acc = accuracy_score(y_test, y_pred)

#print(f'Best hyperparameters: {best_hyp}')
#print(f'Test set accuracy of rf: {acc}')
# These hyperparameters were then applies to the Random Forest model above

## Visualizing feature importances
# Random Forest models give each feature a different "weight" (importance) to make predictions

# Create a pd.Series of feature importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)
# Sort feature importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of sorted feature importances
importances_sorted.plot(kind='barh', color='pink')
plt.title('Features Importances')
plt.show()

"""
### MODEL 2: REGRESSION MODEL FOR LATERAL SCOUR SEVERITY

## Data Pre-Processing
# Create another df subset, this time with lateral scour width instead of scour pool presence/absence (indexing a different column)
root_df_subset1 = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Lateral_Scour_DS', 'Culvert_Material', 'Culvert_Shape']]

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Color' column
shape_encoder = encoder.fit_transform(root_df_subset1[['Culvert_Shape']])

# Convert to DataFrame and concatenate
shape_df = pd.DataFrame(shape_encoder, columns=encoder.get_feature_names_out(['Culvert_Shape']))
root_df_subset1 = pd.concat([root_df_subset1.drop('Culvert_Shape', axis=1), shape_df], axis=1)

# Establish seed for random state (ensures reproducibility)
SEED = 42

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
root_df_subset1['Culvert_Material'] = root_df_subset1['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Replace all "NA' values with nan
root_df_subset1 = root_df_subset1.replace('NA', np.nan)

# Remove all rows with nan values
root_df_subset1 = root_df_subset1.dropna()

## Split data into train and test groups
# Define X and y for splitting dataset
X_reg = root_df_subset1.drop('Lateral_Scour_DS', axis = 1)
y_reg = root_df_subset1['Lateral_Scour_DS']

# Split dataset into train (75%) and test (25%) data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size = 0.25, random_state = SEED)
# Took out the stratify hyperparameter, might be an issue

## Decision Tree Regression model and K-fold Cross Validation
# Instantiate simple decision tree regressor
dr = DecisionTreeRegressor(max_depth = 3, random_state = SEED)

# Instantiate cross validation score to understand under vs over fitting
mse_CV = cross_val_score(dr, X_train_reg, y_train_reg, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = -1)

# Fit the model to the data and make predictions on both the train and test datasets
dr.fit(X_train_reg, y_train_reg)
y_pred_train_reg = dr.predict(X_train_reg)
y_pred_test_reg = dr.predict(X_test_reg)

# Calculate mean standard error and root mean standard error to evaluate model accuracy
CV_mse = mse_CV.mean() * -1 # Compute K-fold cross validation MSE
CV_rmse = CV_mse ** (1/2)
train_rmse = MSE(y_train_reg, y_pred_train_reg) ** (1/2)
test_rmse = MSE(y_test_reg, y_pred_test_reg) ** (1/2)

# Compare all 3 RMSE
print(f'CV RMSE: {CV_rmse}')
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Conclusion: The RMSE of the training data is lower than that of the CW and test RMSE - likely indicates high variance

## Random Forest Regression model to combat high variance
# Instantiate Random Forest Regression model
rr = RandomForestRegressor(max_depth = 4, max_features = 'log2', min_samples_leaf = 0.2, n_estimators = 500, random_state = SEED)
# Uses the hyperparameters found with the GridSearch cross validation method

# Fit the model to the training data and make predictions on test dataset
rr.fit(X_train_reg, y_train_reg)
y_pred_reg = rr.predict(X_test_reg)

# Calculate MSE and RMSE
rr_mse = MSE(y_pred_reg, y_test_reg)

rr_rmse = rr_mse ** (1/2)

print(f'RF RMSE: {rr_rmse}')

## Visualize feature importances used by Random Forest
# Create a pd.Series of feature importances
importances_reg = pd.Series(data=rr.feature_importances_,
                        index= X_train_reg.columns)

# Sort feature importances
importances_sorted_reg = importances_reg.sort_values()

# Draw a horizontal barplot of the sorted feature importances
importances_sorted_reg.plot(kind='barh', color='pink')
plt.title('Features Importances')
plt.show()

## GridSearch cross-validation for hyperparameter tuning

# Print current hyperparameters
#print(rr.get_params())

# Dictionary of hyperparameters to test
#params_rr = {'n_estimators': [200, 300, 400, 500], 'max_depth': [4, 5, 6], 'min_samples_leaf': [0.1, 0.15, 0.2], 'max_features': ['log2', 'sqrt']}

# Instantiate GridSearchCV with the params dictionary to test different hyperparameter combinations
#grid_rr = GridSearchCV(estimator = rr, param_grid = params_rr, cv = 3, scoring = 'neg_mean_squared_error', verbose = 1, n_jobs = -1)

# Fit GridSearch CV to training data
#grid_rr.fit(X_train_reg, y_train_reg)

# Extract the best (lowest RMSE) hyperparameters
#best_hyp_reg = grid_rr.best_params_

# Use the best model (lowest RMSE hyperparameters) to predict on test dataset
#best_model_reg = grid_rr.best_estimator_
#y_pred_reg = best_model_reg.predict(X_test_reg)

# Calculate MSE and RMSE
#mse = MSE(y_test_reg, y_pred_reg)
#rmse = mse ** (1/2)

#print(f'Best hyperparameters: {best_hyp_reg}')
#print(f'Test set RMSE of rf: {rmse}')
# These hyperparameters were applied to the Random Forest model above

"""
NOTES
- For initial CART test - match rf parameters
- Test other types of ensembles? (boosting?)
- Accuracy is better without shape taken into account
- SHOULD THE DATA BE LOG TRANSFORMED
- Try voting classifier?
"""