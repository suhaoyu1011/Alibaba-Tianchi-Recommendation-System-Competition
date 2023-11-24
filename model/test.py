# -*- coding: utf-8 -*

'''

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),
            here we use sub-sample on negative samples.
            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: subsample on each clusters based on the same ratio,
                  the ratio was selected to be the best by testing in random sub_sample + WDL
            ## 3-rd: selecting the best parameter for WDL classifier
            4-th: using WDL model for training and predicting on sub_sample set.

            here is 2-nd to 4-th step
'''

########## file path ##########
##### input file
# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "../../data/mobile/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "../../data/mobile/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic       = "../../data/mobile/raw/df_part_3_uic.csv"

# data_set features
path_df_part_1_U   = "../../data/mobile/feature/df_part_1_U.csv"
path_df_part_1_I   = "../../data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "../../data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "../../data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "../../data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "../../data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "../../data/mobile/feature/df_part_2_U.csv"
path_df_part_2_I   = "../../data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "../../data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "../../data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "../../data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "../../data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "../../data/mobile/feature/df_part_3_U.csv"
path_df_part_3_I   = "../../data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "../../data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "../../data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "../../data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "../../data/mobile/feature/df_part_3_UC.csv"

# item_sub_set P
path_df_P = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"

##### output file
path_df_result = "../../data/mobile/gbdt_res_gbdt_k_means_subsample.csv"
path_df_result_tmp = "../../data/mobile/gbdt_df_result_tmp.csv"

# depending package
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

import time

# some functions
def df_read(path, mode = 'r'):
    '''the definition of dataframe loading function
    '''
    data_file = open(path, mode)
    try:     df = pd.read_csv(data_file, index_col = False)
    finally: data_file.close()
    return   df

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size

    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df) : return df
    else : return df.sample(n = sub_size)

##### loading data of part 1 & 2
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)

print(df_part_1_uic_label_cluster['user_id'].unique().shape)
print(df_part_2_uic_label_cluster['user_id'].unique().shape)


print(df_part_1_uic_label_cluster['item_id'].unique().shape)
print(df_part_2_uic_label_cluster['item_id'].unique().shape)

print(df_part_1_uic_label_cluster['item_category'].unique().shape)
print(df_part_2_uic_label_cluster['item_category'].unique().shape)

print(df_part_1_uic_label_cluster['class'].unique().shape)
print(df_part_2_uic_label_cluster['class'].unique().shape)

