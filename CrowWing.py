#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:06:26 2025

@author: keiragupta
"""

### IMPORT NECESSARY PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

### DATA PRE-PROCESSING
# Import Crow Wing River Dataset
cw_df = pd.read_csv('Desktop/THESIS/CW_Python.csv')

# Subset dataset to only necessary columns
cw_df_subset = cw_df[['DRNAREA_SQ_Miles', 'scour_pool', 'scour_pool_width', 'Width_Span_P1', 'Constriction_Ratio', 'Pipe_length_Thalweg', 'Culvert_Slope', 'Culvert_material_Thalweg']]

# Replace all "NA' values with nan
cw_df_subset = cw_df_subset.replace('N/A', np.nan)

# Remove all rows with nan values
cw_df_subset = cw_df_subset.dropna()

# Encode scour_pool column as 1 = Yes and 0 = No
cw_df_subset['scour_pool'] = cw_df_subset['scour_pool'].apply(lambda val: 1 if val == 'Yes' else 0)

# Change Culvert Length variable to float datatype
cw_df_subset['Pipe_length_Thalweg'] = cw_df_subset['Pipe_length_Thalweg'].astype(float)

## Apply log transformations on variables with high variance
#print(cw_df_subset.var()) # check variance of each variable

# Replace 0 with 1 in the scour pool width column to prevent nan values in the transformation (log(1) = 0)
cw_df_subset['scour_pool_width'] = cw_df_subset['scour_pool_width'].replace(0, 1)

# Apply log transformations
cw_df_subset['log_drainage_area'] = np.log(cw_df_subset['DRNAREA_SQ_Miles'])
cw_df_subset['log_scour_pool_width'] = np.log(cw_df_subset['scour_pool_width'].mask(cw_df_subset['scour_pool_width'] <= 0)) # mask zeros when using np.log
cw_df_subset['log_culvert_length'] = np.log(cw_df_subset['Pipe_length_Thalweg']) 

# Does it make sense to log transform bankfull too?

#print(cw_df_subset.var()) # check work

### EVALUATE CORRELATION AND REGRESSION BETWEEN INDEPENDENT VARIABLES AND LATERAL SCOUR SEVERITY

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
## Calculate correlation
# Define function that calculates Pearson's r correlation coefficient and p-value
def corr(group_a, group_b):
    correlation, p_value = pearsonr(group_a, group_b)
    return correlation, p_value

# Empty lists for results
r = []
p_value_r = []
decision_r = []

# Evaluate correlation, p-value, and statistical significance for each independent variable
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

# Export results as .csv file
name = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Drainage_Area'] # name list for building DataFrame
pearson_zipped = list(zip(name, r, p_value_r, decision_r)) # zip all 4 lists together
pearson_r_df = pd.DataFrame(pearson_zipped, columns = ['Component', 'Pearson r', 'P-value', 'Statistically Significant?']) # Combine zipped lists into DataFrame

pearson_r_df.to_csv('Desktop/THESIS/CW_Severity_Correlation.csv', index = False) # Export as .csv

## Create linear regression plots
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
axes[0].annotate('r = -0.172', xy = (2.189, 4.369)) # annotates the subplot with corresponding correlation coefficient
axes[0].annotate('p-value = 0.007', xy = (2.012, 4.00)) # annotates the subplot with the corresponding p-value

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

plt.subplots_adjust(hspace=0.3) # adjust subplot spacing 
axes[5].set_visible(False) # hide empty subplot
fig.suptitle('Crow Wing: Correlation between Culvert Attributes and Lateral Scour Width')

### T-TESTS TO EVALUATE RELATIONSHIP BETWEEN INDEPENDENT VARIABLES AND SCOUR PRESENCE/ABSENCE

# Define function that calculates t-statistic, p-value, and statistical significance
def two_sample_t_test(group1, group2, equal_var = False, alpha = 0.05):
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var = equal_var, alternative = 'two-sided')
    if p_value < alpha:
        decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
    else:
        decision = 'Fail to reject the null hypothesis: There is not a significant difference between the group means.'
    return t_statistic, p_value, decision

# Empty lists for results
t_stat = []
p_value_t = []
decision_t = []

# Lists for subplot titles and y labels
titles = ['Constriction Ratio', 'Culvert Length', 'Culvert Width', 'Culvert Slope', 'Drainage Area']
ylabels = ['Constriction Ratio', 'Culvert Length (log)', 'Culvert Width (ft)', 'Culvert Slope (ft)', 'Drainage Area (log)']

# Create subplot figure
fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
ax1 = ax1.flatten()

plt.subplots_adjust(hspace=0.3) # adjust subplot spacing
ax1[5].set_visible(False) # hide empty subplot
fig1.suptitle('Crow Wing: Boxplots between Culvert Attributes and Scour Presence/Absence')

# Calculate t-statistic, p-value, and statistical significance for each independent variable, as well as create a subplot figure of boxplots
for i, (col_name1, col_data1) in enumerate(cw_df_subset[['Constriction_Ratio', 'Pipe_length_Thalweg', 'Width_Span_P1', 'Culvert_Slope', 'DRNAREA_SQ_Miles']].items()):
    yes_scour = cw_df_subset[cw_df_subset['scour_pool'] == 1][col_name1] # define groups for t-test
    no_scour = cw_df_subset[cw_df_subset['scour_pool'] == 0][col_name1]
    t_statistic, p_value, decision = two_sample_t_test(yes_scour, no_scour, equal_var = False) # run t-test
    print(f'{col_name1} t-statistic: {t_statistic}')
    print(f'{col_name1} p-value: {p_value}')
    print(f'{col_name1} decision: {decision}')
    t_stat.append(t_statistic) # append lists with their respective results
    p_value_t.append(p_value)
    decision_t.append(decision)
    data = pd.DataFrame({'Value': np.concatenate([yes_scour, no_scour]), 'Group': ['Presence'] * len(yes_scour) + ['Absence'] * len(no_scour)}) # define a dataframe for boxplots
    sns.boxplot(data, x = 'Group', y = 'Value', ax = ax1[i]) # create boxplot
    ax1[i].set_title(f'{titles[i]}')
    ax1[i].set_ylabel(f'{ylabels[i]}')
    ax1[i].set_xlabel('Scour Presence/Absence')

# Calculate degrees of freedom as a sanity check
n1 = len(yes_scour) 
n2 = len(no_scour)
df = n1 + n2 -2
print(f'Degrees of Freedom: {df}')

# Export results as .csv file
name_t = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Drainage_Area']
t_test_zipped = list(zip(name, t_stat, p_value_t, decision_t))
t_test_df = pd.DataFrame(t_test_zipped, columns = ['Component', 't-statistic', 'p-value', 'Statistically Significant?'])

t_test_df.to_csv('Desktop/THESIS/CV_T-TestPA_Correlation.csv', index = False)

### ANOVA TO TEST RELATIONSHIP BETWEEN CULVERT MATERIAL AND LATERAL SCOUR SEVERITY

# Separate into different culvert material groups
group_a = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Corrugated Metal']['log_scour_pool_width']
group_b = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Concrete']['log_scour_pool_width']
group_c = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Smooth_Metal_Pipe']['log_scour_pool_width']
group_d = cw_df_subset[cw_df_subset['Culvert_material_Thalweg'] == 'Corrugated Plastic']['log_scour_pool_width']

# Perform ANOVA test
f_statistic, p_value_f = f_oneway(group_a, group_b, group_c, group_d)

# Evaluate statistical significance
alpha = 0.05
if p_value_f < alpha:
    print('Reject the null hypothesis: There is a significant difference between the group means.')
else:
    print('Fail to reject the null hypothesis: There is not a significant difference between the group means.')

# Print results
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Create boxplot
plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='Culvert_material_Thalweg', y='log_scour_pool_width', data= cw_df_subset) # boxplot
sns.stripplot(x='Culvert_material_Thalweg', y='log_scour_pool_width', data=cw_df_subset, color='black', size=7, jitter=True) # show distribution of points on top of boxplot
plt.title('Crow Wing: Boxplot for Culvert Material and Lateral Scour Severity')
plt.xlabel('Culvert Material')
plt.ylabel('Lateral Scour (log)')

### CHI-SQUARE TEST OF INDEPENDENCE TO EVALUATE RELATIONSHIP BETWEEN CULVERT MATERIAL AND SCOUR PRESENCE/ABSENCE

# Subset dataset to only necessary columns
cw_df_subset1 = cw_df[['DRNAREA_SQ_Miles', 'scour_pool', 'scour_pool_width', 'Width_Span_P1', 'Constriction_Ratio', 'Pipe_length_Thalweg', 'Culvert_Slope', 'Culvert_material_Thalweg']]

cw_df_subset1 = cw_df_subset1.rename(columns={'scour_pool': 'Scour Pool?'})

# Create contingency table of observed frequencies for each variable combination
contingency_table = pd.crosstab(cw_df_subset1['Culvert_material_Thalweg'], cw_df_subset1['Scour Pool?'])

# Replace all "NA' values with nan
cw_df_subset1 = cw_df_subset1.replace('N/A', np.nan)

# Remove all rows with nan values
cw_df_subset1 = cw_df_subset1.dropna()

# Run chi-square test
chi2, p_value_chi, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)

# Evaluate statistical significance
alpha = 0.05
if p_value_chi < alpha:
    decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
else:
    decision = 'Fail to reject the null hypothesis: There is no significant difference between the group means'

# Print results
print(f'Chi2: {chi2}')
print(f'p-value: {p_value_chi}')
print(f'DOF: {degrees_of_freedom}')
print(f'Expected Frequencies: {expected_frequencies}')
print(f'Decision: {decision}')

row_proportions = contingency_table.apply(lambda x: x / x.sum(), axis=1)

row_proportions.plot(kind='bar', figsize=(10, 6))
plt.title('Frequency of Scour Presence and Absence for each Culvert Material')
plt.xlabel('Culvert Material')
plt.ylabel('Frequency')
plt.xticks(rotation = 360)

plt.show()

# Note: Expected frequencies should be greater than 5 for results to be valid
"""
NOTES
- Should I be using log transformations? Compare results and check normalization
- Chi2 visualization
- Should the t-test plot be a violin plot?
- ADD IN SHAPE?
- Low expected frequencies for chi2?
"""


