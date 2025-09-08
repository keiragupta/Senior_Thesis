#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:23:07 2025

@author: keiragupta
"""

# Import necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

### Import Root River Dataset
root_df = pd.read_csv('Documents/THESIS/Root_Python.csv')

### Subset dataset to only necessary columns
root_df_subset = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Bankfull', 'Lateral_Scour_DS', 'Scour_Pool_DS_', 'Culvert_Material']]

# OBJECTID', 'Inventory__' (SPARE COLUMNS)

###

# Replace all "NA' values with nan
root_df_subset = root_df_subset.replace('NA', np.nan)

root_df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove all rows with nan values
root_df_subset = root_df_subset.dropna()

# Check that all nan values have been removed
#root_df_subset.isna().sum()

###

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].apply(lambda val: 1 if val == 'Yes' else 0)

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
root_df_subset['Culvert_Material'] = root_df_subset['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

###

# Apply log transformations on variables with high variance
root_df_subset.var()

root_df_subset['log_Culvert_Length'] = np.log(root_df_subset['Culvert_Length'])
#root_df_subset['log_Total_Culvert_width'] = np.log(root_df_subset['Total_Culvert_width'])
root_df_subset['log_Lateral_Scour_DS'] = np.log(root_df_subset['Lateral_Scour_DS'].mask(root_df_subset['Lateral_Scour_DS'] <= 0)) # mask zeros when using np.log

# Does it make sense to log transform constriction ratio and bankfull too?

#print(root_df_subset.var())

###

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

# Flatten the axes array for easy iteration 
axes = axes.flatten()

# Plot variables on subplots

# CONSTRICTION RATIO
sns.regplot(x = 'Constriction_Ratio', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[0]) # adds a treadline from linear regression

# CULVERT LENGTH
sns.regplot(x = 'log_Culvert_Length', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[1])

# CULVERT WIDTH
sns.regplot(x = 'Culvert_Width', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[2])

# CULVERT SLOPE
sns.regplot(x = 'RR_Culvert_Slope', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[3])

# CULVERT HEIGHT
sns.regplot(x = 'Culvert_Height', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[4])

###

# Correlation and p-value for relationship between culvert components and lateral scour severity
# SHOULD I BE USING LOG TRANSFORMATIONS? - probably but the nan values throw back an error :(

# Define function for calculating correlation (Pearson's r) and p-value
def corr(group_a, group_b): 
    correlation, p_value = pearsonr(group_a, group_b)
    return correlation, p_value

# Create empty lists for pearson's r, p-value, and statistical significance
r = []
p_value_r = []
decision_r = []

alpha = 0.05 # alpha value to test p-value against

# Iterate through each column in the subset to calculate pearson's r and p-value, as well as determine statistical significance
for col_name, col_data in root_df_subset[['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'RR_Culvert_Slope', 'Culvert_Height']].items():
    group_a = col_data
    group_b = root_df_subset['Lateral_Scour_DS']
    correlation, p_value = corr(group_a, group_b) # calculate values between column and lateral scour
    print(f'{col_name} Pearson r: {correlation}')
    print(f'{col_name} p-value: {p_value}')
    if p_value < alpha:
        decision = 'Yes' # Correlation is statistically significant
    else:
        decision = 'No' # Correlation is not statistically significant
    print(f'{col_name} Significance: {decision}')
    r.append(correlation) # append values to their respective lists
    p_value_r.append(p_value)
    decision_r.append(decision)

name = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Culvert_Height'] # name list for building DataFrame
pearson_zipped = list(zip(name, r, p_value_r, decision_r)) # zip all 4 lists together
pearson_r_df = pd.DataFrame(pearson_zipped, columns = ['Component', 'Pearson r', 'P-value', 'Statistically Significant?']) # Combine zipped lists into DataFrame

pearson_r_df.to_csv('Documents/THESIS/Root_Severity_Correlation.csv', index = False) # Export as .csv

###
# T-tests for scour presence/absence

# Define two-sample unpaired t-test function that returns t-statistic, p-value, and statistical significance
def two_sample_t_test(group1, group2, equal_var = False, alpha = 0.05):
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var = equal_var, alternative = 'two-sided', random_state = 42)
    if p_value < alpha:
        decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
    else:
        decision = 'Fail to reject the null hypothesis: There is not a significant difference between the group means.'
    return t_statistic, p_value, decision

t_stat = [] # Empty lists for t-statistic, p-value, and statistical significance
p_value_t = []
decision_t = []

# Iterate through columns and perform t-test between scour presence/absence
for col_name1, col_data1 in root_df_subset[['Constriction_Ratio', 'log_Culvert_Length', 'Culvert_Width', 'RR_Culvert_Slope', 'Culvert_Height']].items():
    yes_scour = root_df_subset[root_df_subset['Scour_Pool_DS_'] == 1][col_name1]
    no_scour = root_df_subset[root_df_subset['Scour_Pool_DS_'] == 0][col_name1]
    t_statistic, p_value, decision = two_sample_t_test(yes_scour, no_scour, equal_var = False)
    print(f'{col_name1} t-statistic: {t_statistic}')
    print(f'{col_name1} p-value: {p_value}')
    print(f'{col_name1} decision: {decision}')
    t_stat.append(t_statistic) # append values to empty lists
    p_value_t.append(p_value)
    decision_t.append(decision)
    data = pd.DataFrame({'Value': np.concatenate([yes_scour, no_scour]), 'Group': ['yes_scour'] * len(yes_scour) + ['no_scour'] * len(no_scour)}) # define a dataframe for boxplots
    plt.figure()
    sns.boxplot(data, x = 'Group', y = 'Value') # create boxplot
    plt.title(f'{col_name1}')

n1 = len(yes_scour) 
n2 = len(no_scour)
df = n1 + n2 -2
print(f'Degrees of Freedom: {df}')

# Create exportable .csv like the correlation data
name_t = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Culvert_Height']
t_test_zipped = list(zip(name, t_stat, p_value_t, decision_t))
t_test_df = pd.DataFrame(t_test_zipped, columns = ['Component', 't-statistic', 'p-value', 'Statistically Significant?'])

t_test_df.to_csv('Documents/THESIS/Root_T-TestPA_Correlation.csv', index = False)

###
# t - test for Culvert Material and Lateral Scour Severity
# Should use log transformation for lateral scour but having same nan values issue

concrete = root_df_subset[root_df_subset['Culvert_Material'] == 1]['Lateral_Scour_DS']
metal = root_df_subset[root_df_subset['Culvert_Material'] == 0]['Lateral_Scour_DS']

t_stat_CM, p_value_CM, decision_CM = two_sample_t_test(concrete, metal, equal_var = False)
print(f'Culvert_Material t-statistic: {t_stat_CM}')
print(f'Culvert_Material p_value: {p_value_CM}')
print(f'Culvert_Material Decision: {decision_CM}')

data_CM = pd.DataFrame({'Value': np.concatenate([concrete, metal]), 'Group': ['concrete'] * len(concrete) + ['metal'] * len(metal)})

plt.figure()
sns.boxplot(data_CM, x = 'Group', y = 'Value')
plt.title('Culvert Material')


"""
TO DO:
- Log transformations for correlations - nan values in lateral scour throw back errors
    - Same for culvert materials t-test
- Could change p-value to 0.1 but 0.05 seems like it is okay
- Split data for ML
- Figure out how to test relationship between material and presence/absence
- Clean up figures, especially axis bounds for each variable
- What does it mean for something to be statistically significant?
- Should the t-test plot be a violin plot?
- Experiment with log transformations
- ADD IN SHAPE?
"""












