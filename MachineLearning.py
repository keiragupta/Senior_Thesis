#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:37:22 2025

@author: keiragupta
"""
# Note: certain parts of this code have been commented out to reduce computational expense

# IMPORT NECESSARY PACKAGES

# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Models
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

pd.options.mode.chained_assignment = None  # disables warning

### DATA PRE-PROCESSING

# Import Crow Wing dataset and subset it to necessary columns
cw_df = pd.read_csv('Desktop/THESIS/CW_Python.csv')

cw_df_subset = cw_df[['Pipe_length_Thalweg', 'DRNAREA_SQ_Miles', 'Constriction_Ratio', 'Culvert_Slope', 'Width_Span_P1', 'Scour Pool?', 'Corrugated Metal?', 'Circular?', 'Outlet_Water_Depth']]

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
cw_df_subset['Scour Pool?'] = cw_df_subset['Scour Pool?'].apply(lambda val: 1 if val == 'Yes' else 0)
cw_df_subset['Corrugated Metal?'] = cw_df_subset['Corrugated Metal?'].apply(lambda val: 1 if val == 'Y' else 0)
cw_df_subset['Circular?'] = cw_df_subset['Circular?'].apply(lambda val: 1 if val == 'Y' else 0)

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
#encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Color' column
#design_encoder = encoder.fit_transform(cw_df_subset[['Outlet Design']])

# Convert to DataFrame and concatenate
#design_df = pd.DataFrame(design_encoder, columns=encoder.get_feature_names_out(['Outlet Design']))
#cw_df_subset = pd.concat([cw_df_subset.drop('Outlet Design', axis=1), design_df], axis=1)

# Replace all "NA' values with nan
cw_df_subset = cw_df_subset.replace('NA', np.nan)

# Remove all rows with nan values
cw_df_subset = cw_df_subset.dropna()

## Import Root dataset and subset it
root_df = pd.read_csv('Desktop/THESIS/Root_Python.csv')

root_df_subset = root_df[['Tailwater_elevation','Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Scour Pool?', 'Square/Rectangle?']]

### DATA PRE-PROCESSING

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
#root_df_subset['Culvert_Material'] = root_df_subset['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset['Scour Pool?'] = root_df_subset['Scour Pool?'].apply(lambda val: 1 if val == 'Yes' else 0)
root_df_subset['Square/Rectangle?'] = root_df_subset['Square/Rectangle?'].apply(lambda val: 1 if val == 'Yes' else 0)

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
#encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Color' column
#shape_encoder = encoder.fit_transform(root_df_subset[['Culvert_Shape']])

# Convert to DataFrame and concatenate
#shape_df = pd.DataFrame(shape_encoder, columns=encoder.get_feature_names_out(['Culvert_Shape']))
#root_df_subset = pd.concat([root_df_subset.drop('Culvert_Shape', axis=1), shape_df], axis=1)

# Replace all "NA' values with nan
root_df_subset = root_df_subset.replace('NA', np.nan)

# Remove all rows with nan values
root_df_subset = root_df_subset.dropna()

# Establish seed for random state (ensures reproducibility)
SEED = 42

DATA = cw_df_subset

### MODEL 1: CLASSIFICATION MODEL FOR SCOUR PRESENCE/ABSENCE

## Split dataset
# Define X and y for splitting dataset
X = DATA.drop('Scour Pool?', axis = 1) # all columns except dependent variable (scour pool yes/no)
y = DATA['Scour Pool?']

# Split dataset into train (75%) and test (25%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED, stratify = y)
"""
## Decision-Tree Classification
# Instantiate simple decision tree classifier
dt = DecisionTreeClassifier(max_depth = 4, max_features = 'log2', min_samples_leaf = 0.1, random_state = SEED)

# Instantiate cross validation score to understand under vs over fitting
acc_CV = cross_val_score(dt, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs = -1)

# Fit the model to the data and make predictions on both the train and test datasets
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# Calculate accuracy scores
CV_acc = acc_CV.mean() # Compute K-fold cross validation accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Compare all 3 accuracies
print(f'CV accuracy: {CV_acc}')
print(f'Train accuracy: {train_acc}')
print(f'Test accuracy: {test_acc}')

# Conclusion: The training accuracy is much higher than the CV and the test accuracy, likely suggesting high variance
"""
## Random Forest Classification model (used to help combat high variance)

# Instantiate random forest classification model
rc = RandomForestClassifier(max_features = 'log2', random_state = SEED)

rc.fit(X_train, y_train)

## Hyperparameter tuning for Random Forest Model

# Print current hyperparameters
#print(rf.get_params())

# Define dictionary of hyperparameters to check
params_rc = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [0.1, 0.15, 0.2]}

# Instantiate GridSearch cross-validation, specifying the params_rf dictionary to test
grid_rc = GridSearchCV(estimator = rc, param_grid = params_rc, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs = -1)

# Fit GridSearchCV to the training data
grid_rc.fit(X_train, y_train)

# Determine the best (most accurate) hyperparameters
best_hyp_class = grid_rc.best_params_

# Use the model with the most accurate hyperparameters to predict on test dataset
best_model_class = grid_rc.best_estimator_
y_pred = best_model_class.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

print(f'Best hyperparameters: {best_hyp_class}')
print(f'RF Accuracy: {acc}')
# These hyperparameters were put into the Random Forest model above

## Create a graph of feature importances
# Random Forest Models give each feature a different "weight" (or importance) - this graph helps visualize which features are prioritized

# Create a pd.Series of feature importances
class_importances = pd.Series(data=rc.feature_importances_,
                        index= X_train.columns)

# Sort the importances
class_importances_sorted = class_importances.sort_values()

# Create a horizontal barplot of the sorted importances
plt.figure()
class_importances_sorted.plot(kind='barh', color='pink')
plt.title('Classification Feature Importances')

plt.show()


### MODEL 2: REGRESSION MODEL FOR LATERAL SCOUR SEVERITY

cw_df_subset1 = cw_df[['Pipe_length_Thalweg', 'DRNAREA_SQ_Miles', 'Constriction_Ratio', 'Lateral Scour', 'Culvert_Slope', 'Width_Span_P1', 'Scour Pool?', 'Corrugated Metal?', 'Circular?', 'Outlet_Water_Depth', 'Bankfull_Width']]

root_df_subset1 = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Lateral Scour', 'Scour Pool?', 'Square/Rectangle?']]

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
cw_df_subset1['Scour Pool?'] = cw_df_subset1['Scour Pool?'].apply(lambda val: 1 if val == 'Yes' else 0)
cw_df_subset1['Corrugated Metal?'] = cw_df_subset1['Corrugated Metal?'].apply(lambda val: 1 if val == 'Y' else 0)
cw_df_subset1['Circular?'] = cw_df_subset1['Circular?'].apply(lambda val: 1 if val == 'Y' else 0)

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
#encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Color' column
#design_encoder = encoder.fit_transform(cw_df_subset1[['Outlet Design']])

# Convert to DataFrame and concatenate
#design_df = pd.DataFrame(design_encoder, columns=encoder.get_feature_names_out(['Outlet Design']))
#cw_df_subset1 = pd.concat([cw_df_subset1.drop('Outlet Design', axis=1), design_df], axis=1)

# Replace all "NA' values with nan
cw_df_subset1 = cw_df_subset1.replace('NA', np.nan)

# Remove all rows with nan values
cw_df_subset1 = cw_df_subset1.dropna()

### DATA PRE-PROCESSING

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
#root_df_subset1['Culvert_Material'] = root_df_subset1['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset1['Scour Pool?'] = root_df_subset1['Scour Pool?'].apply(lambda val: 1 if val == 'Yes' else 0)
root_df_subset1['Square/Rectangle?'] = root_df_subset1['Square/Rectangle?'].apply(lambda val: 1 if val == 'Y' else 0)

## Use One Hot Encoding for Culvert Material (categorical -> numerical data)
# Create an instance of OneHotEncoder
#encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the 'Shape' column
#shape_encoder = encoder.fit_transform(root_df_subset1[['Culvert_Shape']])

# Convert to DataFrame and concatenate
#shape_df = pd.DataFrame(shape_encoder, columns=encoder.get_feature_names_out(['Culvert_Shape']))
#root_df_subset1 = pd.concat([root_df_subset1.drop('Culvert_Shape', axis=1), shape_df], axis=1)

# Replace all "NA' values with nan
root_df_subset1 = root_df_subset1.replace('NA', np.nan)

# Remove all rows with nan values
root_df_subset1 = root_df_subset1.dropna()

DATA1 = cw_df_subset1

## Split data into train and test groups
# Define X and y for splitting dataset
X_reg = DATA1.drop('Lateral Scour', axis = 1)
y_reg = DATA1['Lateral Scour']

# Split dataset into train (75%) and test (25%) data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size = 0.25, random_state = SEED)
# Took out the stratify hyperparameter, might be an issue
"""
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

# Compare all 3 aRMSE
print(f'CV RMSE: {CV_rmse}')
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Conclusion: The RMSE of the training data is lower than that of the CW and test RMSE - likely indicates high variance
"""
## Random Forest Regression model to combat high variance
# Instantiate Random Forest Regression model
rr = RandomForestRegressor(max_features = 'log2', random_state = SEED)

rr.fit(X_train_reg, y_train_reg)

## Visualize feature importances used by Random Forest
# Create a pd.Series of feature importances
importances_reg = pd.Series(data=rr.feature_importances_,
                        index= X_train_reg.columns)

# Sort feature importances
importances_sorted_reg = importances_reg.sort_values()

# Draw a horizontal barplot of the sorted feature importances
plt.figure()
importances_sorted_reg.plot(kind='barh', color='pink')
plt.title('Regression Feature Importances')
plt.show()

## GridSearch cross-validation for hyperparameter tuning

# Print current hyperparameters
#print(rr.get_params())

# Dictionary of hyperparameters to test
params_rr = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [0.1, 0.15, 0.2]}

# Instantiate GridSearchCV with the params dictionary to test different hyperparameter combinations
grid_rr = GridSearchCV(estimator = rr, param_grid = params_rr, cv = 3, scoring = 'neg_mean_squared_error', verbose = 1, n_jobs = -1)

# Fit GridSearch CV to training data
grid_rr.fit(X_train_reg, y_train_reg)

# Extract the best (lowest RMSE) hyperparameters
best_hyp_reg = grid_rr.best_params_

# Use the best model (lowest RMSE hyperparameters) to predict on test dataset
best_model_reg = grid_rr.best_estimator_
y_pred_reg = best_model_reg.predict(X_test_reg)

# Calculate MSE and RMSE
mse = MSE(y_test_reg, y_pred_reg)
rmse = mse ** (1/2)

print(f'Best hyperparameters: {best_hyp_reg}')
print(f'RF RMSE: {rmse}')
# These hyperparameters were applied to the Random Forest model above


"""
NOTES
- Do a little more hyperparameter tuning - n_estimators = 100?
    - Likely depends on dataset...
    - Set up rf so it hyperparameter tunes each time?
- For the Root: added in outlet water depth - increased accuracy but significantly decreased dataset size
"""