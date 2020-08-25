## import modules here 
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################
from helper import *
def buc_rec_optimized(df):# do not change the heading of the function
    for i in df.columns[:-1].tolist():        # change v_1  v_2 ...  v_d to str
        df[i] = df[i].apply(str)
    
    df_out = pd.DataFrame(columns=list(df))
    rec(df, [], df.columns, df_out)
    if df.shape[1] > 1:
        df_out.sort_values(by=df.columns[:-1].tolist(),inplace=True)
    return df_out

def single_row_optimized(df, pre_dims, cols_name_ls, df_out):                    # optimized for only single row
    df_sg = pd.DataFrame(columns=cols_name_ls)
    cols = df.shape[1] - 1
    
    for num in range(0, 2 ** (cols)):
        concat = [i for i in pre_dims]
        binary = bin(num).replace("0b", "")
        temp = np.array(df[0:1])[0]
        for i in range(0, len(binary)):
            if binary[i] == '1':
                temp[len(temp) - len(binary) + i - 1] = "ALL"
        concat.extend(temp)
        df_out.loc[len(df_out)] = concat
    
'''
df: given df
pre_dims: a list of nums removed from df previous
cols_name: name of coloums in given df
df_out: new record calculated
'''
def rec(df, pre_dims, cols_name, df_out):
    cols = df.shape[1]               # num of dims curent have
    rows = df.shape[0]
    
    if cols == 1:                    # the case of only M left
        new = [i for i in pre_dims]
        new.append(sum(project_data(df, 0).apply(int)))
        df_out.loc[len(df_out)] = new
    elif rows == 1:                    # the case of only one row left, use single_row_optimized function
        single_row_optimized(df, pre_dims, cols_name, df_out)
    else:
        dim0_vals = set(project_data(df, 0).values)
        for dim0_v in dim0_vals:
            sub_data = slice_data_dim0(df, dim0_v)
            new = [i for i in pre_dims]
            new.append(dim0_v)
            rec(sub_data, new, cols_name, df_out)         # the case of recursive body
        sub_data = remove_first_dim(df)
        new = [i for i in pre_dims]
        new.append('ALL')
        rec(sub_data, new, cols_name, df_out)
