import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

def fe_time(df, time_col):
    """
    extend hour and minute features
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :return: new DataFrame
    """
    result = pd.DataFrame()
    prefix = time_col + "_"

    df[time_col] = pd.to_datetime(df[time_col])

    # result[prefix + 'day'] = df[time_col].dt.day
    result[prefix + 'hour'] = df[time_col].dt.hour
    result[prefix + 'minute'] = df[time_col].dt.minute
    return result


def fe_rolling_stat(df, id_col, time_col, time_varying_cols, window_size):
    """
    extend new features by rolling window size
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param window_size: window size
    :return: DataFrame rolling
    """
    window_size = [144*7]
    time_varying_cols = ['Etmp']
    result = df[[id_col, time_col]]
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            for op in [ 'max']:
                name = f'{key}__{val}__{cur_ws}__{op}'
                add_feas.append(name)
                if op == 'mean':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).mean())
                if op == 'std':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).std())
                if op == 'median':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).median())
                if op == 'max':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).max())
                if op == 'min':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).min())
                if op == 'kurt':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).kurt())
                if op == 'skew':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).skew())
    return result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')[add_feas]

def fe_lag(df, id_col, time_col, time_varying_cols, lag):
    """
    features lag
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame lag
    """
    result = df[[id_col, time_col]].copy()
    df = df.sort_values(by=time_col)
    add_feas = []
    key = id_col
    for value in time_varying_cols:
        for cur_lag in lag:
            name = f'{key}__{value}__lag__{cur_lag}'
            add_feas.append(name)
            df[name] = df.groupby(key)[value].shift(cur_lag)
    return result.merge(df[[id_col, time_col] + add_feas], on=[id_col, time_col], how='left')[add_feas]


def fe_diff(df, id_col, time_col, time_varying_cols, lag):
    """
    features diff
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: DataFrame diff by lag list
    """
    result = df[[id_col, time_col]].copy()
    df = df.sort_values(by=time_col)
    add_feas = []
    key = id_col
    for value in time_varying_cols:
        for cur_lag in lag:
            name = f'{key}__{value}__diff__{cur_lag}'
            add_feas.append(name)
            df[name] = df[value] - df.groupby(key)[value].shift(cur_lag)
    return result.merge(df[[id_col, time_col] + add_feas], on=[id_col, time_col], how='left')[add_feas]

def feature_combination(df_list):
    """
    feature combination
    :param df_list: DataFrame list
    :return: DataFrame
    """
    result = df_list[0]
    for df in tqdm(df_list[1:], total=len(df_list[1:])):
        if df is None or df.shape[0] == 0:
            continue

        assert (result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
        print(result.shape[0], df.shape[0])
    return result


def feature_filter(train,time_col, target_col):
    """
    remove not used columns
    :param train: DataFrame
    :return: used features list
    """
    not_used = ['y', 't1']
    used_features = [x for x in train.columns if x not in not_used]
    return used_features

def engineering(df):
    """
    1. set threshold of Ndir, Wdir, Etmp, Itmp and pab
    2. extend new features of hour and minute
    3. fixed Etmp value by Itmp
    :param df: DataFrame
    :return: DataFrame
    """
    df.loc[(df['Patv'] < 0) |
           ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)) | \
           ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
            (df['Ndir'] > 720)), 'Patv'] = 0
    df.loc[((df['Patv'] == 0) & (df['Wspd'] > 2.5)), 'Wspd'] = 0
    df.loc[((df['Patv'] != 0) & (df['Wspd'] < 2.5)), 'Patv'] = 0

    def filter_ndir(ndir):
        ndir = ndir % 360
        return ndir

    def filter_wdir(wdir):
        if (wdir < 0):
            wdir = -wdir
        if wdir > 85:
            wdir = 85
        return wdir

    def filter_tmp(tmp):
        if (tmp < -25):
            tmp = -25
        if (tmp > 57):
            tmp = 57
        return tmp

    def filter_pab(pab):
        if (pab > 89):
            pab = 89
        return pab

    df['Ndir'] = df.Ndir.apply(filter_ndir)
    df['Wdir'] = df.Wdir.apply(filter_wdir)
    df['Etmp'] = df.Etmp.apply(filter_tmp)
    df['Itmp'] = df.Itmp.apply(filter_tmp)
    df['Pab1'] = df.Pab1.apply(filter_pab)
    df['Pab2'] = df.Pab2.apply(filter_pab)
    df['Pab3'] = df.Pab3.apply(filter_pab)
    df['hour'] = df['t1'].dt.hour
    df['hour_sin'] = np.sin(df['hour'] / 23 * 2 * np.pi)
    df['hour_cos'] = np.cos(df['hour'] / 23 * 2 * np.pi)
    df['cross'] = (df['Wdir'] ** 3) * df['Etmp'] * np.cos(df['Wdir'] / 360 * 2 * np.pi) ** 3
    df.loc[df['Etmp'] == 57,'Etmp'] = df.loc[df['Etmp'] == 57,'Itmp'] -9
    # df['Wdir'] = np.cos(df['Wdir'] / 360 * 2 * np.pi)**3
    # df['Wspd'] =  df['Wspd']**3
    df.drop(columns='hour', inplace=True)
    return df


def get_date(k):
    """
    1. convert day column data type, string to datetime
    2. set base date is 2020-01-01
    :param k: day of number
    :return: begin with '2020-01-01' datetime value
    """
    cur_date = "2020-01-01"
    one_day = timedelta(days=k - 1)
    return str(datetime.strptime(cur_date, '%Y-%m-%d') + one_day)[:10]


def cols_concat(df, con_list):
    """
    etmp processing
    """
    name = 't1'
    df[name] = df[con_list[0]].astype(str)
    for item in con_list[1:]:
        df[name] = df[name] + ' ' + df[item].astype(str)
    return df

def forecast(settings):
    id_col = 'TurbID'
    time_col = 't1'
    time_varying_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv','cross']
    lag_range = list(range(1, 7, 1))
    test_df = pd.read_csv(settings["path_to_test_x"])
    test_df['Day'] = test_df['Day'].apply(lambda x: get_date(x))
    test_df = cols_concat(test_df, ["Day", "Tmstamp"])

    test_df = test_df[['TurbID', 't1', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']]
    test_df['t1'] = pd.to_datetime(test_df['t1'])
    test_df.replace(to_replace=np.nan, value=0, inplace=True)
    test_df = engineering(test_df)
 

    test_df.index = range(len(test_df))
    test_df_rolling_stat = fe_rolling_stat(test_df,
                                          id_col=id_col,
                                          time_col=time_col,
                                          time_varying_cols=time_varying_cols,
                                          window_size=[6,12,18])
    test_df_lag = fe_lag(test_df,
                         id_col=id_col,
                         time_col=time_col,
                         time_varying_cols=time_varying_cols,
                         lag=lag_range)

    test_df_diff = fe_diff(test_df,
                           id_col=id_col,
                           time_col=time_col,
                           time_varying_cols=time_varying_cols,
                           lag=lag_range)
    test_df_time = fe_time(test_df, time_col=time_col)
    combination_test = feature_combination([test_df,  test_df_lag, test_df_diff, test_df_time,test_df_rolling_stat])
    test_days = 2
    test_delta = timedelta(days=test_days)
    test_time_split = combination_test[time_col].max() - test_delta
    combination_test.index = range(len(combination_test))
    combination_test = combination_test[combination_test[time_col] > test_time_split]

    test_df_all = combination_test.copy()
    step_range = list(range(288, 108, -6)) + list(range(108, 12, -2)) + [12, 6, 4, 2]
    step_range = list(reversed(step_range))
    if ModelSet.models is None: 
        filepath = os.path.join(settings["checkpoints"], "lightgbm_models")
        ModelSet.models = {}
        model_files = {}
        for fname in os.listdir(filepath):
            if not fname.endswith(".m"):
                continue
            model_path = os.path.join(filepath, fname)
            time_step = fname[:fname.find("_")]
            model_files[time_step] = model_path
        for model_step in step_range:
            ModelSet.models[str(model_step)] = lgb.Booster(model_file=model_files[str(model_step)])
    pred = np.zeros((134,288, 1))
    times = 0
    for step_length in step_range:#range(288, 1 , -1):
        test_df_all['y'] = test_df_all.groupby(id_col)['Patv'].shift(-step_length)
        test = test_df_all.loc[test_df_all['y'].isnull()]
        test_used_features = feature_filter(test, time_col, target_col='y')
        model = ModelSet.models[str(step_length)]
        test_result = model.predict(test[test_used_features])
        one_pred = test_result.reshape(134, step_length, 1)
        
        for tid in range(134):
            one_turb_result = one_pred[tid]
            
            if times == 0:
                #prediction[0:output_len,:] = one_pred
                pred[tid][0:step_length,:] = one_turb_result
            else:
                pred[tid] = merge_prediction(pred[tid], one_turb_result, pre_output_len, next_output_len = step_length)#, coefficient = coefficients[times-1])
        pre_output_len = step_length    
        times += 1

    return pred


class ModelSet:
    models = None
    step = 0
    def __init__(self):
        print("init")

def merge_prediction(prediction, next_prediction, pre_output_len, next_output_len, coefficient=0.4):
    #print(  pre_output_len, next_output_len)
    first_prediction_mean = np.mean(prediction[:pre_output_len])
    
    next_predictions_mean = np.mean(next_prediction[:pre_output_len])
    prediction[pre_output_len:next_output_len,:] = next_prediction[pre_output_len:] - (next_predictions_mean-first_prediction_mean)*coefficient
    
    return prediction