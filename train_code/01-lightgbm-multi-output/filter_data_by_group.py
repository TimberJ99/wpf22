#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np


df = pd.read_csv("../datasets/wtbdata_245days.csv")

print(df.shape)
print(df.columns.values)
print(len(df.loc[(df.Patv < 0) | (df.Patv.isna())]))

dfilter = df.copy(deep=True)

def filter_no_valid(dfilter):
    """
    filter no valid data
    :param dfilter: DataFrame
    :return: DataFrame
    """
    no_valid_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
    
    dfilter.loc[dfilter.Patv < 0, no_valid_cols] = np.nan
    
    dfilter.loc[((dfilter.Wspd < 1) & (dfilter.Patv > 10)), no_valid_cols] = np.nan
    dfilter.loc[((dfilter.Wspd < 2) & (dfilter.Patv > 100)), no_valid_cols] = np.nan
    dfilter.loc[((dfilter.Wspd < 3) & (dfilter.Patv > 200)), no_valid_cols] = np.nan
    dfilter.loc[((dfilter.Wspd > 2.5) & (dfilter.Patv == 0)), no_valid_cols] = np.nan
    
    dfilter.loc[((dfilter.Wspd == 0) & (dfilter.Wdir == 0) & (dfilter.Etmp == 0)), no_valid_cols] = np.nan
    
    dfilter.loc[(dfilter.Etmp < -21), no_valid_cols] = np.nan
    dfilter.loc[(dfilter.Etmp > 60), 'Etmp'] = np.nan
    dfilter.loc[(dfilter.Itmp < -21), no_valid_cols] = np.nan
    dfilter.loc[(dfilter.Itmp > 70), 'Itmp'] = np.nan
    
    dfilter.loc[((dfilter.Wdir > 180) | (dfilter.Wdir < -180)), no_valid_cols] = np.nan
    dfilter.loc[((dfilter.Ndir > 720) | (dfilter.Ndir < -720)), no_valid_cols] = np.nan
    dfilter.loc[((dfilter.Pab1 > 89) | (dfilter.Pab2 > 89) | (dfilter.Pab3 > 89)), no_valid_cols] = np.nan
    
    return dfilter


dfilter = filter_no_valid(dfilter)


turb_groups = {
    "group1": {"key": [32, 33, 34, 56], "member": [9, 10, 11, 12, 31, 32, 33, 34, 35, 52, 53, 54, 55, 56, 57]},
    "group2": {"key": [4, 5, 28, 51], "member": [3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 50, 51]},
    "group3": {"key": [23, 46, 47, 25], "member": [1, 2, 22, 23, 24, 25, 26, 45, 46, 47, 66, 67, 68, 48, 49]},
    "group4": {"key": [19, 42, 43, 64], "member": [18, 19, 20, 21, 41, 42, 43, 44, 62, 63, 64, 65]},
    "group5": {"key": [15, 37, 38, 39], "member": [13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 58, 59, 60, 61]},
    "group6": {"key": [125, 103, 104, 84], "member": [124, 125, 126, 102, 103, 104, 105, 80, 81, 82, 83, 84, 85]},
    "group7": {"key": [129, 107, 108, 88], "member": [127, 128, 129, 130, 131, 106, 107, 108, 109, 86, 87, 88, 89]},
    "group8": {"key": [133, 111, 112, 70], "member": [132, 133, 134, 113, 110, 111, 112, 91, 92, 90, 69, 70, 71, 72]},
    "group9": {"key": [115, 94, 74, 75], "member": [114, 115, 116, 117, 93, 94, 95, 96, 73, 74, 75, 76]},
    "group10": {"key": [119, 120, 98, 99], "member": [118, 119, 120, 121, 122, 123, 97,  98,  99, 100, 101, 77, 78, 79]}
}

mean_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']

def assign_mean(item):
    item[mean_cols] = item[mean_cols].fillna(item[mean_cols].mean().round(decimals=2), axis=0)
    return item


dfilter2 = dfilter.copy(deep=True)

for group in turb_groups.keys():
    print("-------Cur group: ", group, ", members: ", turb_groups[group]["member"])
    members = turb_groups[group]["member"]
    df_members = dfilter2.loc[(dfilter2.TurbID.isin(members))]
    df_idx = df_members.index
    dfilter2.loc[df_idx] = df_members.groupby(['Day', 'Tmstamp']).apply(assign_mean)
    
nan_result = []

for turb in range(1, 135):
    turb_result = []
    cur = dfilter2.loc[(dfilter2.TurbID == turb)]
    for day in range(1, 246):
        turb_result.append(len(cur.loc[(cur.Day == day) & (cur.Patv.isna())]))
    nan_result.append(turb_result)

pd_nan_result = pd.DataFrame(nan_result)

for turb in range(134):
    cur_turb = pd_nan_result.loc[turb]
    day_index = cur_turb.loc[cur_turb <= 28].index.values + 1
    
    print("Turb-", turb+1, ": ", cur_turb.sum(), ", ", len(day_index))
    fill_index = dfilter2.loc[(dfilter2.TurbID == turb+1) & (dfilter2.Day.isin(day_index))].index
    dfilter2.loc[fill_index] = dfilter2.loc[fill_index].fillna(method='bfill', axis=0)
    dfilter2.loc[fill_index] = dfilter2.loc[fill_index].fillna(method='ffill', axis=0)
 

dfilter2.to_csv("../datasets/wtbdata_245days_filternan_by_group.csv", index=False)

print(len(dfilter2.loc[(dfilter2.Patv < 0) | (dfilter2.Patv.isna())]))





