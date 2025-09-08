#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:06:26 2025

@author: keiragupta
"""

# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

### Import Crow Wing River Dataset
cw_df = pd.read_csv('Documents/THESIS/CW_Python.csv')

### Subset dataset to only necessary columns
cw_df_subset = cw_df[['bankfull_width', 'DRNAREA_SQ_Miles', 'scour_pool', 'scour_pool_width', 'Width_Span_P1', 'Constriction_Ratio', 'Pipe_length_Thalweg', 'Culvert_Slope']]

###

# Replace all "NA' values with nan
cw_df_subset = cw_df_subset.replace('N/A', np.nan)

cw_df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove all rows with nan values
cw_df_subset = cw_df_subset.dropna()

# Check that all nan values have been removed
#print(cw_df_subset.isna().sum())

###

# Change Scour Pool DS? column to boolean datatype
#root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].map({'Yes': True, 'No': False}) # map Yes/No to corresponding T/F

#root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].astype('bool') # change datatype

#print(root_df_subset['Scour_Pool_DS_'].dtype) # check work

# Encode scour_pool column as 1 = Yes and 0 = No
cw_df_subset['scour_pool'] = cw_df_subset['scour_pool'].apply(lambda val: 1 if val == 'Yes' else 0)

# Encode Culvert Material column as True/False series for each material type
#material_dummies = pd.get_dummies(cw_df['Culvert_material_Thalweg'], dtype = float)
#print(material_dummies)

# Change Culvert Length variable to float datatype
cw_df_subset['Pipe_length_Thalweg'] = cw_df_subset['Pipe_length_Thalweg'].astype(float)


###

# Apply log transformations on variables with high variance
#print(cw_df_subset.var())

cw_df_subset['log_drainage_area'] = np.log(cw_df_subset['DRNAREA_SQ_Miles'])
cw_df_subset['log_scour_pool_width'] = np.log(cw_df_subset['scour_pool_width'].mask(cw_df_subset['scour_pool_width'] <= 0)) # mask zeros when using np.log
cw_df_subset['log_culvert_length'] = np.log(cw_df_subset['Pipe_length_Thalweg']) 

# Does it make sense to log transform bankfull too?

print(cw_df_subset.var()) # Check variance

###

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

# Flatten the axes array for easy iteration 
axes = axes.flatten()

# Plot variables on subplots

# CONSTRICTION RATIO
sns.regplot(x = 'Constriction_Ratio', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[0]) # adds a treadline from linear regression

# CULVERT LENGTH
sns.regplot(x = 'log_culvert_length', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[1])

# CULVERT WIDTH
sns.regplot(x = 'Width_Span_P1', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[2])

# CULVERT SLOPE
sns.regplot(x = 'Culvert_Slope', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[3])

# DRAINAGE AREA - missing culvert height
sns.regplot(x = 'log_drainage_area', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[4])

###

# Correlation and p-value for relationship between culvert components and lateral scour severity
# SHOULD I BE USING LOG TRANSFORMATIONS?

def corr(group_a, group_b):
    correlation, p_value = pearsonr(group_a, group_b)
    return correlation, p_value

r = []
p_value_r = []
decision_r = []

alpha = 0.05
for col_name, col_data in cw_df_subset[['Constriction_Ratio', 'Pipe_length_Thalweg', 'Width_Span_P1', 'Culvert_Slope', 'DRNAREA_SQ_Miles']].items():
    group_a = col_data
    group_b = cw_df_subset['scour_pool_width']
    correlation, p_value = corr(group_a, group_b)
    print(f'{col_name} Pearson r: {correlation}')
    print(f'{col_name} p-value: {p_value}')
    if p_value < alpha:
        decision = 'Yes' # Correlation is statistically significant
    else:
        decision = 'No' # Correlation is not statistically significant
    r.append(correlation)
    p_value_r.append(p_value)
    decision_r.append(decision)

name = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Drainage_Area'] # name list for building DataFrame
pearson_zipped = list(zip(name, r, p_value_r, decision_r)) # zip all 4 lists together
pearson_r_df = pd.DataFrame(pearson_zipped, columns = ['Component', 'Pearson r', 'P-value', 'Statistically Significant?']) # Combine zipped lists into DataFrame

pearson_r_df.to_csv('Documents/THESIS/CW_Severity_Correlation.csv', index = False) # Export as .csv


###
# T-tests for scour presence/absence

def two_sample_t_test(group1, group2, equal_var = False, alpha = 0.05):
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var = equal_var, alternative = 'two-sided')
    if p_value < alpha:
        decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
    else:
        decision = 'Fail to reject the null hypothesis: There is not a significant difference between the group means.'
    return t_statistic, p_value, decision

t_stat = []
p_value_t = []
decision_t = []

for col_name1, col_data1 in cw_df_subset[['Constriction_Ratio', 'Pipe_length_Thalweg', 'Width_Span_P1', 'Culvert_Slope', 'DRNAREA_SQ_Miles']].items():
    yes_scour = cw_df_subset[cw_df_subset['scour_pool'] == 1][col_name1]
    no_scour = cw_df_subset[cw_df_subset['scour_pool'] == 0][col_name1]
    t_statistic, p_value, decision = two_sample_t_test(yes_scour, no_scour, equal_var = False)
    print(f'{col_name1} t-statistic: {t_statistic}')
    print(f'{col_name1} p-value: {p_value}')
    print(f'{col_name1} decision: {decision}')
    t_stat.append(t_statistic)
    p_value_t.append(p_value)
    decision_t.append(decision)

n1 = len(yes_scour) 
n2 = len(no_scour)
df = n1 + n2 -2
print(f'Degrees of Freedom: {df}')

name_t = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Drainage_Area']
t_test_zipped = list(zip(name, t_stat, p_value_t, decision_t))
t_test_df = pd.DataFrame(t_test_zipped, columns = ['Component', 't-statistic', 'p-value', 'Statistically Significant?'])

t_test_df.to_csv('Documents/THESIS/CV_T-TestPA_Correlation.csv', index = False)

###
# t - test for Culvert Material and Lateral Scour Severity
#plt.bar(cw_df['Culvert_material_Thalweg'], cw_df['scour_pool'])

#concrete = root_df_subset[root_df_subset['Culvert_Material'] == 1]['Lateral_Scour_DS']
#metal = root_df_subset[root_df_subset['Culvert_Material'] == 0]['Lateral_Scour_DS']

#t_stat_CM, p_value_CM, decision_CM = two_sample_t_test(concrete, metal, equal_var = False)
#print(f'CM t-statistic: {t_stat_CM}')
#print(f'CM p_value: {p_value_CM}')
#print(f'CM Decision: {decision_CM}')

"""
NOTES
- Figure out culvert material with multiple categorical variables
- log transformations with correlation/t-tests?
- Export datasets like Root code
"""


