#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:51:00 2025

@author: keiragupta
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Import Root River Dataset
root_df = pd.read_csv('Desktop/THESIS/Root_Python.csv')

# Subset dataset to only necessary columns
root_df_subset = root_df[['Lateral_Scour_DS', 'Scour_Pool_DS_']]

# Import Crow Wing River Dataset
cw_df = pd.read_csv('Desktop/THESIS/CW_Python.csv')

# Subset dataset to only necessary columns
cw_df_subset = cw_df[['scour_pool', 'scour_pool_width']]


fig, ax = plt.subplots(nrows = 1, ncols = 2)

ax = ax.flatten()

sns.boxplot(cw_df_subset['scour_pool_width'], ax = ax[0])
sns.boxplot(root_df_subset['Lateral_Scour_DS'], ax = ax[1])

ax[0].set_title('Crow Wing')
ax[1].set_title('Root')

ax[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax[1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

fig.suptitle('Lateral Scour')

plt.show()

fig1, ax1 = plt.subplots(nrows = 1, ncols = 2)

root_df_subset['Scour_Pool_DS_'] = root_df_subset['Scour_Pool_DS_'].replace('NO', 'No')

sns.countplot(x='Scour_Pool_DS_', data= root_df_subset, ax = ax1[0])
sns.countplot(x = 'scour_pool', data = cw_df_subset, ax = ax1[1])
ax1[0].set_title('Root')
ax1[1].set_title('Crow Wing')

fig1.suptitle('Scour Presence/Absence: Bar Chart')

plt.show()

fig2, ax2 = plt.subplots(nrows = 1, ncols = 2)

category_counts = root_df_subset['Scour_Pool_DS_'].value_counts()
category_counts1 = cw_df_subset['scour_pool'].value_counts()

ax2[0].pie(category_counts, labels=category_counts.index)
ax2[1].pie(category_counts1, labels = category_counts1.index)
ax2[0].set_title('Root')
ax2[1].set_title('Crow Wing')

fig2.suptitle('Scour Presence/Absence: Pie Chart')

plt.show()

"""
NOTES
- Do I have to calculate statistics on these too :(
    - T-test?
"""