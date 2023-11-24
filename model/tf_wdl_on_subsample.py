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
import itertools

from tensorflow.python.framework import dtypes

path_df_part_1_uic_label_cluster = "../data/mobile/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "../data/mobile/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic       = "../data/mobile/raw/df_part_3_uic.csv"

# data_set features
path_df_part_1_U   = "../data/mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "../data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "../data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "../data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "../data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "../data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "../data/mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "../data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "../data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "../data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "../data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "../data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "../data/mobile/feature/df_part_3_U.csv"  
path_df_part_3_I   = "../data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "../data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "../data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "../data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "../data/mobile/feature/df_part_3_UC.csv"

# item_sub_set P
path_df_P = "../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"

##### output file
path_df_result = "../data/mobile/wdl_res_wdl_k_means_subsample.csv"
path_df_result_tmp = "../data/mobile/wdl_df_result_tmp.csv"

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

df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U  = df_read(path_df_part_2_U )   
df_part_2_I  = df_read(path_df_part_2_I )
df_part_2_C  = df_read(path_df_part_2_C )
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

print("数据load完毕")

DenseFeatureNames = ['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6',
                     'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3',
                     'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1',
                     'u_b4_rate','u_b4_diff_hours',
                     'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                     'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6',
                     'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                     'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1',
                     'i_b4_rate','i_b4_diff_hours',
                     'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                     'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                     'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                     'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                     'c_b4_rate','c_b4_diff_hours',
                     'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c',
                     'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                     'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                     'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1',
                     'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                     'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                     'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6',
                     'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3',
                     'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                     'uc_b_count_rank_in_u',
                     'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours']
SparseFeatureNames = ['user_id','item_id','item_category']
# 稀疏特征是很重要的！！这里我们没有那么多。

Feature_Names = DenseFeatureNames + SparseFeatureNames

##### generation of training set & valid set
def train_set_construct(np_ratio = 1, sub_ratio = 1):
    '''
    # generation of train set
    @param np_ratio: int, the sub-sample rate of training set for N/P balanced.
    @param sub_ratio: float ~ (0~1], the further sub-sample rate of training set after N/P balanced.
    '''
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    
    frac_ratio = sub_ratio * np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")
    
    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)
    
    # # using all the features for training wdl model
    # train_X = train_df[Feature_Names].values
    # train_y = train_df['label'].values
    train_X = train_df[Feature_Names]
    train_y = train_df['label']
    print("train subset is generated.")
    return train_X, train_y
    
def valid_set_construct(sub_ratio = 0.1):
    '''
    # generation of valid set
    @param sub_ratio: float ~ (0~1], the sub-sample rate of original valid set
    '''
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)

    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
    
        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)
    
    # using all the features for valid wdl model
    valid_X = valid_df[Feature_Names].values
    valid_y = valid_df['label'].values
    print("valid subset is generated.")
 
    return valid_X, valid_y


##### generation and splitting to training set & valid set
def valid_train_set_construct(valid_ratio = 0.5, valid_sub_ratio = 0.5, train_np_ratio = 1, train_sub_ratio = 0.5):
    '''
    # generation of train set
    @param valid_ratio: float ~ [0~1], the valid set ratio in total set and the rest is train set
    @param valid_sub_ratio: float ~ (0~1), random sample ratio of valid set
    @param train_np_ratio:(1~1200), the sub-sample ratio of training set for N/P balanced.
    @param train_sub_ratio: float ~ (0~1), random sample ratio of train set after N/P subsample
    
    @return valid_X, valid_y, train_X, train_y
    '''
    msk_1 = np.random.rand(len(df_part_1_uic_label_cluster)) < valid_ratio
    msk_2 = np.random.rand(len(df_part_2_uic_label_cluster)) < valid_ratio
        
    valid_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[msk_1]
    valid_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[msk_2]
    
    valid_part_1_uic_label = valid_df_part_1_uic_label_cluster[ valid_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio)
    valid_part_2_uic_label = valid_df_part_2_uic_label_cluster[ valid_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio)
    
    ### constructing valid set
    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == i]
        if len(valid_part_1_uic_label_0_i) != 0:
            valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = valid_sub_ratio)
            valid_part_1_uic_label     = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
        
        valid_part_2_uic_label_0_i = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == i]
        if len(valid_part_2_uic_label_0_i) != 0:
            valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = valid_sub_ratio)
            valid_part_2_uic_label     = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)
    
    # using all the features for valid rf model
    valid_X = valid_df[Feature_Names].values
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    ### constructing training set
    train_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[~msk_1]
    train_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[~msk_2] 
    
    train_part_1_uic_label = train_df_part_1_uic_label_cluster[ train_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio)
    train_part_2_uic_label = train_df_part_2_uic_label_cluster[ train_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio)
    
    frac_ratio = train_sub_ratio * train_np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == i]
        if len(train_part_1_uic_label_0_i) != 0:
            train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
            train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == i]
        if len(train_part_2_uic_label_0_i) != 0:
            train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
            train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    
    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)
    
    # using all the features for training rf model
    train_X = train_df[Feature_Names].values
    train_y = train_df['label'].values
    print("train subset is generated.")
    
    return valid_X, valid_y, train_X, train_y


#######################################################################

#######################################################################
'''Step 2: training the optimal wdl model and predicting on part_3 
'''
train_X, train_y = train_set_construct(np_ratio=60, sub_ratio=1)
print(train_X.info())
print(train_X.dtypes)
print(train_y.dtypes)
print('训练数据集划分完毕')


# build WDL model and fitting
# 其实我们构造的特征多数都是实数值，并没有太多sparse特征，所以并不一定适合WDL；这里只是为了说明用法
def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    wide_columns = []  # 类别特征、类别交叉特征
    deep_columns = []   # 数值特征、类别特征one-hot、Embedding

    # 用户约 18252
    # SparseFeatureNames = ['user_id','item_id','item_category']
    # user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', hash_bucket_size=19000, dtype=dtypes.int64)
    # 商品约 1262961
    # item_id = tf.feature_column.categorical_column_with_hash_bucket('item_id', hash_bucket_size=1270000, dtype=dtypes.int64)
    # 商品类别约 7814
    item_category = tf.feature_column.categorical_column_with_hash_bucket('item_category', hash_bucket_size=8000, dtype=dtypes.int64)
    # # class约 1001；不能用，因为prediction的时候没有
    # classfea = tf.feature_column.categorical_column_with_hash_bucket('class', hash_bucket_size=1001, dtype=dtypes.int64)

    # 词表 类别特征例子
    # education = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'education', [
    #         'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    #         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    #         '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    # 分组特征例子
    # age_buckets = tf.feature_column.bucketized_column(
    #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    # 交叉特征例子
    # tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE)

    wide_columns += [
        # user_id,
                     # item_id,
                     item_category
                     ]

    # 实数值特征
    for fea in DenseFeatureNames:
        deep_columns.append(
            tf.feature_column.numeric_column(fea)
        )
    #  类别特征 indicator 例子
    # tf.feature_column.indicator_column(relationship),
    #  Embedding
    deep_columns += [
    #     tf.feature_column.embedding_column(user_id, dimension=6),
    #     # tf.feature_column.embedding_column(item_id, dimension=8),
        tf.feature_column.embedding_column(item_category, dimension=4)
    ]

    return wide_columns, deep_columns

# 模型
def build_wdl_model(wide_columns,deep_columns):
    estimator =  tf.estimator.DNNLinearCombinedClassifier(
        model_dir="./_tf_model",
        # wide settings
        linear_feature_columns=wide_columns,
        linear_optimizer='Ftrl',
        # deep settings
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[128, 10, 4],
        dnn_optimizer='Adagrad')

    # To apply L1 and L2 regularization, you can set dnn_optimizer to:
    tf.compat.v1.train.ProximalAdagradOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)
    # To apply learning rate decay, you can set dnn_optimizer to a callable:
    # lambda: tf.keras.optimizers.Adam(
    #     learning_rate=tf.compat.v1.train.exponential_decay(
    #         learning_rate=0.1,
    #         global_step=tf.compat.v1.train.get_global_step(),
    #         decay_steps=10000,
    #         decay_rate=0.96))
    return estimator

wide_columns, deep_columns = build_model_columns()
estimator = build_wdl_model(wide_columns, deep_columns)

# Input builders
def input_fn_train(train_X,train_y,num_epochs):
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: train_X[k].values for k in Feature_Names}),
        y = pd.Series(train_y.values),
        num_epochs=num_epochs,
        shuffle=False)

print('模型构造完毕，开始训练')
# 训练
estimator.train(input_fn=input_fn_train(train_X,train_y,num_epochs=2), steps=100)

print('模型训练完成，开始预测')

# 后面做预测
# predictions = estimator.predict(input_fn=input_fn_predict)

# del 内存不足的话可以删掉前面训练部分的数据
# del df_part_2_uic_label_cluster

##### predicting
# loading feature data
df_part_3_U  = df_read(path_df_part_3_U )   
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)
# pred_uic = df_read(path_df_part_3_uic).head(1000)
pred_uic = df_read(path_df_part_3_uic)

def test_set_construct():
    pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
    pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
    pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
    pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
    pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
    pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])

    # fill the missing value as -1 (missing value are time features)
    pred_df.fillna(-1, inplace=True)

    # # using all the features for training wdl model
    # train_X = train_df[Feature_Names].values
    # train_y = train_df['label'].values
    pred_X = pred_df[Feature_Names]
    print("测试样本生成完成.")
    return pred_X

def input_fn_predict(pred_X):
    # Returns tf.data.Dataset of (x, None) tuple.
    # dataset = tf.data.Dataset.from_tensor_slices(dict(pred_X))
    # dataset = dataset.batch(10240)
    # return dataset
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: pred_X[k].values for k in Feature_Names}),
        y = None,
        num_epochs=1,
        batch_size = 1024,
        shuffle=False)
pred_X = test_set_construct()
pred_y = estimator.predict(input_fn=input_fn_predict(pred_X))

print(pred_y)
# for pred_y_data in pred_y:
#     print(pred_y_data)
#     break
# pred_y_data = next(pred_y)
# print(pred_y_data)
print(pred_X.shape)
# pred_y_data = list(itertools.islice(pred_y,10))
# pred_y_data = [p["classes"] for p in pred_y]

pred_y_data = []
for p in pred_y:
    if len(pred_y_data) % 10000 == 0:
        print('预测进度', len(pred_y_data))
    # pred_y_data.append(p["classes"][0])
    pred_y_data.append(p['logistic'][0])
    # break
pred_y_data = pd.DataFrame(pred_y_data)
print(pred_y_data.head(10))

# generation of U-I pairs those predicted to buy
pred_X['pred_label'] = (pred_y_data > 0.5).astype(int)
# pred_X['pred_label'] = pred_y_data
print(pred_X[['user_id','item_id','pred_label']].head(30))
# add to result csv
pred_X[pred_X['pred_label'] == 1 ].to_csv(path_df_result_tmp,
                                         columns=['user_id','item_id'],
                                         index=False, header=False, mode='w')
# pred_X.to_csv(path_df_result_tmp,
#                                          columns=['user_id','item_id'],
#                                          index=False, header=False, mode='a')
print("预测结束")



#######################################################################
'''Step 3: generation result on items' sub set P
'''
# loading data
df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)

print(' 完成。 ')