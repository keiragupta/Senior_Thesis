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
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

### Import Crow Wing River Dataset
cw_df = pd.read_csv('Desktop/THESIS/CW_Python.csv')

### Subset dataset to only necessary columns
cw_df_subset = cw_df[['bankfull_width', 'DRNAREA_SQ_Miles', 'scour_pool', 'scour_pool_width', 'Width_Span_P1', 'Constriction_Ratio', 'Pipe_length_Thalweg', 'Culvert_Slope', 'Culvert_material_Thalweg']]

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

# Replace 0 with 1 in the scour pool width column to prevent nan values in the transformation (log(1) = 0)
cw_df_subset['scour_pool_width'] = cw_df_subset['scour_pool_width'].replace(0, 1)

cw_df_subset['log_drainage_area'] = np.log(cw_df_subset['DRNAREA_SQ_Miles'])
cw_df_subset['log_scour_pool_width'] = np.log(cw_df_subset['scour_pool_width'].mask(cw_df_subset['scour_pool_width'] <= 0)) # mask zeros when using np.log
cw_df_subset['log_culvert_length'] = np.log(cw_df_subset['Pipe_length_Thalweg']) 

# Does it make sense to log transform bankfull too?

#rint(cw_df_subset.var()) # Check variance

###

"""
# A little experiment - what if we get rid of all the instances where lateral scour severity = 0?

cw_df_subset1 = cw_df_subset[cw_df_subset['log_scour_pool_width'] != 0]

# CONSTRICTION RATIO
sns.regplot(x = 'Constriction_Ratio', y = 'log_scour_pool_width', data = cw_df_subset1, ci = None, ax = axes[0]) # adds a treadline from linear regression

# CULVERT LENGTH
sns.regplot(x = 'log_culvert_length', y = 'log_scour_pool_width', data = cw_df_subset1, ci = None, ax = axes[1])

# CULVERT WIDTH
sns.regplot(x = 'Width_Span_P1', y = 'log_scour_pool_width', data = cw_df_subset1, ci = None, ax = axes[2])

# CULVERT SLOPE
sns.regplot(x = 'Culvert_Slope', y = 'log_scour_pool_width', data = cw_df_subset1, ci = None, ax = axes[3])

# DRAINAGE AREA - missing culvert height
sns.regplot(x = 'log_drainage_area', y = 'log_scour_pool_width', data = cw_df_subset1, ci = None, ax = axes[4])

plt.show()

def corr(group_a, group_b):
    correlation, p_value = pearsonr(group_a, group_b)
    return correlation, p_value

alpha = 0.05
for col_name, col_data in cw_df_subset1[['Constriction_Ratio', 'Pipe_length_Thalweg', 'Width_Span_P1', 'Culvert_Slope', 'DRNAREA_SQ_Miles']].items():
    group_a = col_data
    group_b = cw_df_subset1['log_scour_pool_width']
    correlation, p_value = corr(group_a, group_b)
    print(f'{col_name} Pearson r: {correlation}')
    print(f'{col_name} p-value: {p_value}')
    if p_value < alpha:
        decision = 'Yes' # Correlation is statistically significant
    else:
        decision = 'No' # Correlation is not statistically significant
    print(f'{col_name} Decision: {decision}')

"""
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
    group_b = cw_df_subset['log_scour_pool_width']
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

pearson_r_df.to_csv('Desktop/THESIS/CW_Severity_Correlation.csv', index = False) # Export as .csv
###

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))

# Flatten the axes array for easy iteration 
axes = axes.flatten()

# Plot variables on subplots

# CONSTRICTION RATIO
sns.regplot(x = 'Constriction_Ratio', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[0]) # adds a treadline from linear regression
axes[0].set_xlabel('Constriction Ratio')
axes[0].set_ylabel('Lateral Scour Severity (log)')
axes[0].set_title("Constriction Ratio")
axes[0].annotate('r = -0.172', xy = (2.189, 4.369))
axes[0].annotate('p-value = 0.007', xy = (2.012, 4.00))

# CULVERT LENGTH
sns.regplot(x = 'log_culvert_length', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[1])
axes[1].set_xlabel('Culvert Length (log)')
axes[1].set_ylabel('Lateral Scour Severity (log)')
axes[1].set_title("Culvert Length")
axes[1].annotate('r = 0.141', xy = (4.572, 1.412))
axes[1].annotate('p-value = 0.028', xy = (4.393, 0.866))

# CULVERT WIDTH
sns.regplot(x = 'Width_Span_P1', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[2])
axes[2].set_xlabel('Culvert Width (ft)')
axes[2].set_ylabel('Lateral Scour Severity (log)')
axes[2].set_title('Culvert Width')
axes[2].annotate('r = 0.172', xy = (11.50, 1.444))
axes[2].annotate('p-value = 0.248', xy = (10.29, 1.031))

# CULVERT SLOPE
sns.regplot(x = 'Culvert_Slope', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[3])
axes[3].set_xlabel('Culvert Slope (ft)')
axes[3].set_ylabel('Lateral Scour Severity (log)')
axes[3].set_title('Culvert Slope')
axes[3].annotate('r = 0.016', xy = (6.39, 4.362))
axes[3].annotate('p-value = 0.803', xy = (5.28, 3.948))

# DRAINAGE AREA - missing culvert height
sns.regplot(x = 'log_drainage_area', y = 'log_scour_pool_width', data = cw_df_subset, ci = None, ax = axes[4])
axes[4].set_xlabel('Drainage Area (log)')
axes[4].set_ylabel('Lateral Scour Severity (log)')
axes[4].set_title('Drainage Area')
axes[4].annotate('r = 0.086', xy = (-3.34, 4.323))
axes[4].annotate('p-value = 0.177', xy = (-3.42, 3.850))

plt.show()

plt.subplots_adjust(hspace=0.3) 
axes[5].set_visible(False)
fig.suptitle('Crow Wing: Correlation between Culvert Attributes and Lateral Scour Width')

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

titles = ['Constriction Ratio', 'Culvert Length', 'Culvert Width', 'Culvert Slope', 'Drainage Area']
ylabels = ['Constriction Ratio', 'Culvert Length (log)', 'Culvert Width (ft)', 'Culvert Slope (ft)', 'Drainage Area (log)']

fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
ax1 = ax1.flatten()

plt.subplots_adjust(hspace=0.3) 
ax1[5].set_visible(False)
fig1.suptitle('Crow Wing: Boxplots between Culvert Attributes and Scour Presence/Absence')

for i, (col_name1, col_data1) in enumerate(cw_df_subset[['Constriction_Ratio', 'Pipe_length_Thalweg', 'Width_Span_P1', 'Culvert_Slope', 'DRNAREA_SQ_Miles']].items()):
    yes_scour = cw_df_subset[cw_df_subset['scour_pool'] == 1][col_name1]
    no_scour = cw_df_subset[cw_df_subset['scour_pool'] == 0][col_name1]
    t_statistic, p_value, decision = two_sample_t_test(yes_scour, no_scour, equal_var = False)
    print(f'{col_name1} t-statistic: {t_statistic}')
    print(f'{col_name1} p-value: {p_value}')
    print(f'{col_name1} decision: {decision}')
    t_stat.append(t_statistic)
    p_value_t.append(p_value)
    decision_t.append(decision)
    data = pd.DataFrame({'Value': np.concatenate([yes_scour, no_scour]), 'Group': ['Presence'] * len(yes_scour) + ['Absence'] * len(no_scour)}) # define a dataframe for boxplots
    sns.boxplot(data, x = 'Group', y = 'Value', ax = ax1[i]) # create boxplot
    ax1[i].set_title(f'{titles[i]}')
    ax1[i].set_ylabel(f'{ylabels[i]}')
    ax1[i].set_xlabel('Scour Presence/Absence')

n1 = len(yes_scour) 
n2 = len(no_scour)
df = n1 + n2 -2
print(f'Degrees of Freedom: {df}')

name_t = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Drainage_Area']
t_test_zipped = list(zip(name, t_stat, p_value_t, decision_t))
t_test_df = pd.DataFrame(t_test_zipped, columns = ['Component', 't-statistic', 'p-value', 'Statistically Significant?'])

t_test_df.to_csv('Desktop/THESIS/CV_T-TestPA_Correlation.csv', index = False)

###
# ANOVA to test culvert material and lateral scour severity

# Separate into different culvert material groups
group_a = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Corrugated Metal']['log_scour_pool_width']
group_b = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Concrete']['log_scour_pool_width']
group_c = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Smooth_Metal_Pipe']['log_scour_pool_width']
group_d = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Corrugated Plastic']['log_scour_pool_width']

# Perform ANOVA test
f_statistic, p_value_f = f_oneway(group_a, group_b, group_c, group_d)

alpha = 0.05
if p_value_f < alpha:
    print('Reject the null hypothesis: There is a significant difference between the group means.')
else:
    print('Fail to reject the null hypothesis: There is not a significant difference between the group means.')

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='Culvert_material_Thalweg', y='log_scour_pool_width', data= cw_df_subset)
sns.stripplot(x='Culvert_material_Thalweg', y='log_scour_pool_width', data=cw_df_subset, color='black', size=7, jitter=True)
plt.title('Crow Wing: Boxplot for Culvert Material and Lateral Scour Severity')
plt.xlabel('Culvert Material')
plt.ylabel('Lateral Scour (log)')

###
# Chi-squared test of independence to test relationship between culvert material and scour presence/absence

contingency_table = pd.crosstab(cw_df_subset['Culvert_material_Thalweg'], cw_df_subset['scour_pool'])

chi2, p_value_chi, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)

alpha = 0.05
if p_value_chi < alpha:
    decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
else:
    decision = 'Fail to reject the null hypothesis: There is no significant difference between the group means'

print(f'Chi2: {chi2}')
print(f'p-value: {p_value_chi}')
print(f'DOF: {degrees_of_freedom}')
print(f'Expected Frequencies: {expected_frequencies}')
print(f'Decision: {decision}')

# Expected frequencies should be greater than 5 for results to be valid

"""
NOTES
- log transformations didn't make a huge difference i think :(
- Could change p-value to 0.1? - RESEARCH
- Chi2 visualization
- What does it mean for something to be statistically significant? - RESEARCH
- Should the t-test plot be a violin plot?
- ADD IN SHAPE?
- Low expected frequencies for chi2?
"""


