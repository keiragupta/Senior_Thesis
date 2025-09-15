#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:23:07 2025

@author: keiragupta
"""

### IMPORT NECESSARY PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

### DATA PRE-PROCESSING
# Import Root River Dataset
root_df = pd.read_csv('Downloads/Root_Python.csv')

# Subset dataset to only necessary columns
root_df_subset = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Lateral_Scour_DS', 'Scour_Pool_DS_', 'Culvert_Material']]

## Encode categorical variables

pd.options.mode.chained_assignment = None  # disables warning

# Encode Scour Pool DS? column as 1 = Yes and 0 = No
root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].apply(lambda val: 1 if val == 'Yes' else 0)

# Encode Culvert Material column as 1 = Concrete and 0 = Metal
root_df_subset['Culvert_Material'] = root_df_subset['Culvert_Material'].apply(lambda val: 1 if val == 'CONCRETE' else 0)

# Replace all "NA' values with nan
root_df_subset = root_df_subset.replace('NA', np.nan)

# Remove all rows with nan values
root_df_subset = root_df_subset.dropna()

# Check that all nan values have been removed
#root_df_subset.isna().sum()

## Apply log transformations on variables with high variance
root_df_subset.var() # check variance

# Replace 0 with 1 in Lateral_Scour_DS column to prevent nan values in the transformation (log(1) = 0)
root_df_subset['Lateral_Scour_DS'] = root_df_subset['Lateral_Scour_DS'].replace(0, 1)

# Apply log transformations
root_df_subset['log_Culvert_Length'] = np.log(root_df_subset['Culvert_Length'])
root_df_subset['log_Lateral_Scour_DS'] = np.log(root_df_subset['Lateral_Scour_DS'].mask(root_df_subset['Lateral_Scour_DS'] <= 0)) # mask zeros when using np.log

# Does it make sense to log transform constriction ratio and bankfull too?

#print(root_df_subset.var()) # check work

### CALCULATE AND VISUALIZE CORRELATION BETWEEN CULVERT COMPONENTS AND LATERAL SCOUR SEVERITY

## Calculate correlation
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

## Export correlation, p-value, and statistical significance as .csv file
name = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Culvert_Height'] # name list for building DataFrame
pearson_zipped = list(zip(name, r, p_value_r, decision_r)) # zip all 4 lists together
pearson_r_df = pd.DataFrame(pearson_zipped, columns = ['Component', 'Pearson r', 'P-value', 'Statistically Significant?']) # Combine zipped lists into DataFrame

pearson_r_df.to_csv('Desktop/THESIS/Root_Severity_Correlation.csv', index = False) # Export as .csv

## Regression plots
# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Flatten the axes array for easy iteration 
axes = axes.flatten()

# Plot variables on subplots

# CONSTRICTION RATIO
sns.regplot(x = 'Constriction_Ratio', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[0]) # adds a treadline from linear regression
axes[0].set_xlabel('Constriction Ratio')
axes[0].set_ylabel('Lateral Scour Severity (log)')
axes[0].set_title("Constriction Ratio")
axes[0].annotate('r = -0.274', xy = (17.55, 4.07)) # annotates graph with Pearson's r correlation coefficient 
axes[0].annotate('p-value = 0.004', xy = (14.27, 3.78)) # annotates graph with p-value

# CULVERT LENGTH
sns.regplot(x = 'log_Culvert_Length', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[1])
axes[1].set_xlabel('Culvert Length (log)')
axes[1].set_ylabel('Lateral Scour Severity (log)')
axes[1].set_title("Culvert Length")
axes[1].annotate('r = -0.140', xy = (4.748, 2.399))
axes[1].annotate('p-value = 0.146', xy = (4.468, 2.081))

# CULVERT WIDTH
sns.regplot(x = 'Culvert_Width', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[2])
axes[2].set_xlabel('Culvert Width (ft)')
axes[2].set_ylabel('Lateral Scour Severity (log)')
axes[2].set_title('Culvert Width')
axes[2].annotate('r = -0.112', xy = (28.36, 4.14))
axes[2].annotate('p-value = 0.248', xy = (23.80, 3.88))

# CULVERT SLOPE
sns.regplot(x = 'RR_Culvert_Slope', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[3])
axes[3].set_xlabel('Culvert Slope (ft)')
axes[3].set_ylabel('Lateral Scour Severity (log)')
axes[3].set_title('Culvert Slope')
axes[3].annotate('r = -0.165', xy = (-4.01, 2.886))
axes[3].annotate('p-value = 0.087', xy = (-4.04, 2.455))

# CULVERT HEIGHT
sns.regplot(x = 'Culvert_Height', y = 'log_Lateral_Scour_DS', data = root_df_subset, ci = None, ax = axes[4])
axes[4].set_xlabel('Culvert Height (ft)')
axes[4].set_ylabel('Lateral Scour Severity (log)')
axes[4].set_title('Culvert Height')
axes[4].annotate('r = -0.144', xy = (10.34, 2.006))
axes[4].annotate('p-value = 0.136', xy = (9.34, 1.482))

plt.subplots_adjust(hspace=0.3) # adjust subplot spacing
axes[5].set_visible(False) # hide empty subplot
fig.suptitle('Root: Correlation between Culvert Attributes and Lateral Scour Width')

### T-TESTS TO EVALUATE RELATIONSHIP BETWEEN INDEPENDENT VARIABLES AND SCOUR PRESENCE/ABSENCE

# Define two-sample unpaired t-test function that returns t-statistic, p-value, and statistical significance
def two_sample_t_test(group1, group2, equal_var = False, alpha = 0.05):
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var = equal_var, alternative = 'two-sided', random_state = 42)
    if p_value < alpha:
        decision = 'Reject the null hypothesis: There is a significant difference between the group means.'
    else:
        decision = 'Fail to reject the null hypothesis: There is not a significant difference between the group means.'
    return t_statistic, p_value, decision

# Empty lists for t-statistic, p-value, and statistical significance
t_stat = []
p_value_t = []
decision_t = []

# Lists of titles and y labels for creating subplots within the for loop
titles = ['Constriction Ratio', 'Culvert Length', 'Culvert Width', 'Culvert Slope', 'Culvert_Height']
ylabels = ['Constriction Ratio', 'Culvert Length (log)', 'Culvert Width (ft)', 'Culvert Slope (ft)', 'Culvert Height (ft)']

# Create figure
fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
ax1 = ax1.flatten()

plt.subplots_adjust(hspace=0.3) # adjust subplot spacing
ax1[5].set_visible(False) # hide empty subplot
fig1.suptitle('Root: Boxplots between Culvert Attributes and Scour Presence/Absence')

# Iterate through columns and perform t-test between scour presence/absence
for i, (col_name1, col_data1) in enumerate(root_df_subset[['Constriction_Ratio', 'log_Culvert_Length', 'Culvert_Width', 'RR_Culvert_Slope', 'Culvert_Height']].items()):
    Presence = root_df_subset[root_df_subset['Scour_Pool_DS_'] == 1][col_name1] # Define groups
    Absence = root_df_subset[root_df_subset['Scour_Pool_DS_'] == 0][col_name1]
    t_statistic, p_value, decision = two_sample_t_test(Presence, Absence, equal_var = False) # Apply t-test
    print(f'{col_name1} t-statistic: {t_statistic}')
    print(f'{col_name1} p-value: {p_value}')
    print(f'{col_name1} decision: {decision}')
    t_stat.append(t_statistic) # append values to empty lists
    p_value_t.append(p_value)
    decision_t.append(decision)
    data = pd.DataFrame({'Value': np.concatenate([Presence, Absence]), 'Group': ['Presence'] * len(Presence) + ['Absence'] * len(Absence)}) # define a dataframe for boxplots
    sns.boxplot(data, x = 'Group', y = 'Value', ax = ax1[i]) # create subplot of boxplots for each variable
    ax1[i].set_title(f'{titles[i]}')
    ax1[i].set_ylabel(f'{ylabels[i]}')
    ax1[i].set_xlabel('Scour Presence/Absence')
    
# Calculate degrees of freedom as a sanity check
n1 = len(Presence) 
n2 = len(Absence)
df = n1 + n2 -2
print(f'Degrees of Freedom: {df}')

# Create exportable .csv like the correlation data
name_t = ['Constriction_Ratio', 'Culvert_Length', 'Culvert_Width', 'Culvert_Slope', 'Culvert Height']
t_test_zipped = list(zip(name, t_stat, p_value_t, decision_t))
t_test_df = pd.DataFrame(t_test_zipped, columns = ['Component', 't-statistic', 'p-value', 'Statistically Significant?'])

t_test_df.to_csv('Desktop/THESIS/Root_T-TestPA_Correlation.csv', index = False)

### T-TEST TO EVALUATE RELATIONSHIP BETWEEN CULVERT MATERIAL AND LATERAL SCOUR SEVERITY

# Establish groups for t-test
concrete = root_df_subset[root_df_subset['Culvert_Material'] == 1]['Lateral_Scour_DS']
metal = root_df_subset[root_df_subset['Culvert_Material'] == 0]['Lateral_Scour_DS']

# Run t-test and print results
t_stat_CM, p_value_CM, decision_CM = two_sample_t_test(concrete, metal, equal_var = False)
print(f'Culvert_Material t-statistic: {t_stat_CM}')
print(f'Culvert_Material p_value: {p_value_CM}')
print(f'Culvert_Material Decision: {decision_CM}')

# Compile data for boxplot
data_CM = pd.DataFrame({'Value': np.concatenate([concrete, metal]), 'Group': ['concrete'] * len(concrete) + ['metal'] * len(metal)})

# Create boxplot
plt.figure()
sns.boxplot(data_CM, x = 'Group', y = 'Value')
plt.title('Boxplot of Culvert Material and Lateral Scour')
plt.ylabel('Lateral Scour (log)')
plt.xlabel('Culvert Material')

### CHI-SQUARE TEST OF INDEPENDENCE TO EVALUATE RELATIONSHIP BETWEEN CULVERT MATERIAL & SCOUR PRESENCE/ABSENCE

root_df_subset1 = root_df[['Culvert_Length', 'Culvert_Height', 'Constriction_Ratio', 'RR_Culvert_Slope', 'Culvert_Width', 'Lateral_Scour_DS', 'Scour_Pool_DS_', 'Culvert_Material']]

root_df_subset1 = root_df_subset1.rename(columns={'Scour_Pool_DS_': 'Scour Pool?'})

# Replace all "NA' values with nan
root_df_subset1 = root_df_subset1.replace('NA', np.nan)

# Remove all rows with nan values
root_df_subset1 = root_df_subset1.dropna()

# Create contingency table that compiles observed frequencies of each variable combination
contingency_table = pd.crosstab(root_df_subset1['Culvert_Material'], root_df_subset1['Scour Pool?'])

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
plt.title('Frequency of Scour Presence and Absence for Concrete and Metal Culverts')
plt.xlabel('Culvert Material')
plt.ylabel('Frequency')
plt.xticks(rotation = 360)

plt.show()

# Note: Expected frequencies should be greater than 5 for results to be valid

"""
TO DO:
- Should I be using log transformations? Compare results and check normalization
- Should the t-test plot be a violin plot?
- ADD IN SHAPE?
"""