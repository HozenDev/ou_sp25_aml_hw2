'''
Advanced Machine Learning

Andrew H. Fagg

Tools for working with DataFrames that contain a set of experimental results

Context: we are performing 2D and 3D Cartesian experiments that are indexed by 
training set size and rotation (N, R), and 
training set size, rotation, and hyper-parameter set (N, R, H)

Our DataFrame is just a linear list of all of the experiments.  Each row
in the DataFrame should describe the details of the experiment (what 
hyper-parameters that were used, how much training data was used, etc.), 
as well as the performance of the model with respect to the training,
validation, and testing data (I log the metrics loss, rmse, and fvaf 
for each of these).

The following functions will first translate this list into either the 
1D or 2D tensor of experiment cells before 1) collecting a list of the 
individual metric values for each cell, and 2) computing the mean metric
value for each cell.  The dimensions for these cells are defined by key0 
(for our purposes, 'Ntraining') and by key1 (for our purposes, the column 
containing the hyperparameter).  Any experiments that share key0 and key1 are
grouped together into the cell (for our purposes, these are our rotations).

'''

import pickle
import pandas as pd
import numpy as np

def load_results_df(fname:str)->pd.DataFrame: 
    '''
    Extract a single DataFrame from a pickle file

    :param fname: Name of the file
    :return: The corresponding DataFrame
    '''
    with open(fname, "rb") as fp:
        df = pickle.load(fp)
    return df

def compute_means_1D(df:pd.DataFrame, key0:str, metrics_keys:list[str], 
                     key1:str=None, filter_tuples:[[str,str]]=None)->dict:
    '''
    Group rows in a DataFrame into cells based on a single key (key0), and then 
    compute the mean of a set of metrics for each cell.

    :param df: DataFrame with one experiment per row
    :param key0: Key used to define the cells.  The cells correspond to the unique 
                    key0 values
    :param metrics_keys: Columns in the DataFrame that contain the metrics of 
                    interest
                    
    Optional arguments (used for a 3D experiment that we are collapsing into 2D):                
    :param key1: a second key value
    :param filter_tuples: a list of key0/key1 tuples to include in the cell definition.
            Typical use: for each key0 value, there is exactly one key1 value.  E.g., 
            when key0 is 'Ntraining' and key1 is 'dropout', and filter_tuples has
            exactly one occurence of each training size, then we use this to select
            a particular N, H combination.

    :return: Dictionary that contains
                0. 'key0_values': the values for key0
                1. For each metrics_key: vector that contains the mean of the metric for 
                    each cell (cells are ordered the same as 'values_key')
                2. For each metrics_key + "_individual": matrix that contains the metrics
                    for each each individual for each key0 value.  For our use case,
                    the shape is (N, R)

    Example 1:
    metrics_keys = ['predict_testing_eval_loss', 'predict_validation_eval_loss']
    df2d = DataFrame that describes a 2D Cartesian product of N x R
    
    compute_means_1D(df2d, 'Ntraining', metrics_keys)
    
        returned dict will contain:
            'key0_values': List of unique key values.  Shape is (N,)
            'predict_testing_eval_loss': vector that contains testing loss as a function 
                of number of training folds.  Shape is (N,)
            'predict_validation_eval_loss': vector that contains validation loss as a function 
                of number of training folds.  Shape is (N,)
            'predict_testing_eval_loss_individual': matrix that contains testing loss. 
                Shape is (N, R)
            'predict_validation_eval_loss_individual': matrix that contains validation loss.  
                Shape is (N, R)

    Example 2:
    metrics_keys = ['predict_testing_eval_loss', 'predict_validation_eval_loss']
    df3d = DataFrame that describes a 3D Cartesian product of Ntraining, rotations and dropout_probs
    tuple_list = [(1, 0.5), (2, 0.5), (3, 0.25)]    # tuple = one combination (Ntraining, dropout)
    
    compute_means_1D(df3d, 'Ntraining', metrics_keys, key1='dropout', tuple_list)
        df3d will be reduced to a df2d by selecting only rows of the DF that match the tuple_list.  
        This df2d represents the Cartesian product of Ntraining, rotations.
        Assuming that the filter_tuples is configured correctly, we will select only the best 
        dropout prob for each Ntraining.
        
        returned dict will contain the same list of entries and shapes as in Example 1
    
    '''
    
    if key1 is not None:
        assert filter_tuples is not None, "key1 and filter_tuples must both be defined"
        # Filter the rows in the DF to just those that match key0/key1 pairs
        df = df[df[[key0, key1]].apply(tuple, axis=1).isin(filter_tuples)]
        
    # Group by key 
    group = df[[key0]+metrics_keys].groupby(key0)

    # Compute the means of each group
    group_df = group.mean(numeric_only=True).reset_index()

    # Extract lists of individual group items
    group_df_individual = group.agg(list).reset_index()

    # Accumulate results in a dict
    d = {}
    
    # Loop over metrics that we are interested in
    for k in metrics_keys:
        # Extract a vector that is the same length as <key0>
        d[k] = group_df[k].values
        
        # Individual values
        m = group_df_individual[k].values
        d[k+'_individual'] = np.array([np.array(r) for r in m])

    # Extract the unique values for this key
    d['key0_values'] = np.array(group_df[key0])
    
    return d

def compute_means_2D(df:pd.DataFrame, key0:str, key1:str, metrics_keys:list[str])->dict:
    '''
    Group rows in a DataFrame into individual cells and then compute the mean of a set of 
    metrics for each cell.  The cells are arranged in 2D, corresponding to the Cartesian 
    product of the unique values of key0 and key1

    :param df: DataFrame with one experiment per row
    :param key0: Key used to define first cell dimension.  
    :param key1: Key used to define second cell dimension.  The cells correspond to the unique 
                combination of the key0/key1 values in df
    :param metrics_keys: Columns in the DataFrame that contain the metrics

    :return: Dictionary that contains
                0. 'key0_values': the unique key0 values
                1. 'key1_values': the unique key1 values
                2. For each metrics_key: matrix that contains the mean of the metric for 
                    each group (defined by the unique combination of key0 and key1).  Matrix
                    is of shape (n0, n1), corresponding to the number of key0 values, and the
                    number of key1 values, respectively.  For our case, the shape is (N, H)
    '''
    
    # Group the rows by key0 and key1
    # Within each key0/key1 combination, compute the mean of all of the values
    group_df = df[[key0, key1]+metrics_keys].groupby([key0, key1]).mean(numeric_only=True).reset_index()
   
    # Accumulate results in a dict
    d = {}

    # Set of key values in order
    key0_values = d['key0_values'] = np.array(sorted(group_df[key0].unique()))
    key1_values = d['key1_values'] = np.array(sorted(group_df[key1].unique()))
    
    # Loop over values that we are interested in
    for k in metrics_keys:
        # Extract a matrix that is key0 x key1 in shape
        mat = group_df.pivot(index=key0, columns=key1, values=k).reindex(index=key0_values, columns=key1_values).values
        # Save this mat
        d[k] = mat
    
    return d


