# -*-Encoding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import paddle
from model import BaselineGruModel
from tqdm import tqdm
from datetime import timedelta

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


def max_of_day(df):
    """
    Calculate the maximum temperature of the day (average of the first 6 items)
    :param df: DataFrame
    :param cols: column list
    :return: new DataFrame
    """
    df_etmp = df.groupby(by=['TurbID', 'Day'])['Etmp'].nlargest(6)
    df_etmp = df_etmp.groupby(by=['TurbID', 'Day']).mean()
    df = df.merge(df_etmp, how='left', on=['TurbID', 'Day'], suffixes=('', '_Max'))
    return df

def fe_rolling_stat(df, id_col, time_col, time_varying_cols = ['Etmp_Max'], window_size = [144*7]):
    """
    extend new features by rolling window size
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param window_size: window size
    :return: DataFrame rolling
    """
    result = df[[id_col, time_col]]
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            ops = ['max']
            if val == 'Wspd':
                ops = ['mean']
            for op in ops:#['mean', 'std', 'median', 'max', 'min', 'kurt', 'skew']:
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
                if op == 'max_top':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).agg(
                            lambda x: x.sort_values(ascending=False).iloc[0:7].mean()))

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
    return result


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

    df.fillna(method='bfill', inplace=True)
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
    df.loc[df['Etmp'] == 57, 'Etmp'] = df.loc[df['Etmp'] == 57, 'Itmp'] - 9

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


def get_test_df(test_df):
    """
    etmp processing
    """
    etmp_abnormal = []
    for i in range(1, 135):
        na_len = len(test_df.loc[(test_df.TurbID == i) & (test_df.Etmp.isna())])
        big_len = len(test_df.loc[(test_df.TurbID == i) & (test_df.Etmp > 55)])
        small_len = len(test_df.loc[(test_df.TurbID == i) & (test_df.Etmp < -25)])
        # print(i, ":", na_len, ",", big_len, ",", small_len)
        etmp_abnormal.append(na_len + big_len + small_len)

    np_etmp_abnormal = np.array(etmp_abnormal)
    fullest_turbid = np.argmin(np_etmp_abnormal) + 1
    test_df.loc[test_df.TurbID==fullest_turbid, 'Etmp'] = test_df.loc[test_df.TurbID==fullest_turbid, 'Etmp'].fillna(method='bfill', axis=0)
    return fullest_turbid, test_df


def cols_concat(df, con_list):
    """
    columns concat
    :param df: DataFrame
    :param con_list: columns of DataFrame
    :return: Symmetric Mean Absolute Percentage Error
    """
    name = 't1'
    df[name] = df[con_list[0]].astype(str)
    for item in con_list[1:]:
        df[name] = df[name] + ' ' + df[item].astype(str)
    return df

def forecast(settings):
    """
    1. linear regression predict forecast
    2. polynomial + linear regression predict forecast
    """
    id_col = 'TurbID'
    time_col = 't1'
    test_df = pd.read_csv(settings["path_to_test_x"])

    test_df['Day'] = test_df['Day'].apply(lambda x: get_date(x))
    test_df = cols_concat(test_df, ["Day", "Tmstamp"])

    test_df = test_df[['TurbID', 'Day', 't1', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']]
    test_df['t1'] = pd.to_datetime(test_df['t1'])

    # emtp correction begin
    fullest_turbid, test_df = get_test_df(test_df)
    normal_df = test_df[test_df['TurbID'] == fullest_turbid].reset_index(drop=True)
    for turbid in range(1, 135):
        if turbid == fullest_turbid:
            continue
        matched_indexes = test_df.loc[
            (test_df.TurbID == turbid) & ((test_df.Etmp.isna()) | (test_df.Etmp > 55) | (test_df.Etmp < -25))].index
        if len(matched_indexes) == 0:
            continue
        cur_test_df = test_df.loc[test_df.TurbID == turbid].reset_index(drop=True)
        replace_index = cur_test_df.loc[
            (cur_test_df.Etmp.isna()) | (cur_test_df.Etmp > 55) | (cur_test_df.Etmp < -25)].index
        test_df.loc[matched_indexes, 'Etmp'] = normal_df.iloc[replace_index]['Etmp'].to_numpy()
    # emtp correction end

    test_df = engineering(test_df)
    test_df = max_of_day(test_df)

    test_df.index = range(len(test_df))
    test_df_rolling_stat = fe_rolling_stat(test_df, id_col=id_col, time_col=time_col, time_varying_cols=['Etmp_Max'])

    test_df_time = fe_time(test_df, time_col=time_col)
    test_df_all = feature_combination([test_df, test_df_time, test_df_rolling_stat])#

    # gru model predict
    gru_pred_rsts = forecast_gru(settings, test_df_all.copy())

    # predict data correction
    gru_pred_rsts = gru_pred_rsts.reshape(134, 288)
    for i in range(134):
        first_144_mean = np.mean(gru_pred_rsts[i, :144])
        last_144_mean = np.mean(gru_pred_rsts[i, 144:])
        gru_pred_rsts[i, 144:] = gru_pred_rsts[i, 144:] - (last_144_mean - first_144_mean) * 0.2

    return gru_pred_rsts

def forecast_gru(settings, test_df):
    '''
    gru model predict
    :param settings: settings
    :praram test_df: DataFrame
    :return: predict rst
    '''
    test_df = test_df.groupby(by='TurbID').tail(144)
    x_cols = ['t1_hour', 't1_minute', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv',
              'Patv', 'Etmp_Max', 'TurbID__Etmp_Max__1008__max']
    test_df_x = test_df[x_cols]
    df_tids = pd.get_dummies(test_df[['TurbID']].astype('str')) # turbIDs to one-hot code

    df_x_mean = pd.Series(data=[11.5,  25.,   4.64961189,  12.81895343,
        21.30357538,  28.08165651, 183.48454203,  26.48563845,
        26.46179722,  26.46014277, -13.5327978, 352.31441098,
        26.50406199,  31.01784447],
              index=x_cols)

    df_x_std = pd.Series(data=[6.92218728,  17.07825308,   3.45731353,  21.86603939,
        13.21341263,  11.86931683, 113.60663821,  38.35331004,
        38.34317362,  38.34066324,  70.72137391, 429.30993922,
        12.97492221,  12.35120604],
                          index=x_cols)
    # normalization encode
    test_df_x = (test_df_x-df_x_mean)/df_x_std
    test_df_x = pd.concat([df_tids, test_df_x], axis=1)
    test_df_x = test_df_x[test_df_x['TurbID_1'] == 1]
    in_var_size = len(test_df_x.columns)
    args = {
        "output_len": 288,
        "in_var": in_var_size,
        "out_var": 1,
        "dropout": 0.05,
        "lstm_layer": 2,
    }

    model = BaselineGruModel(args)
    path_to_model = os.path.join(settings["checkpoints"], "gru-single-models/10_model_144_288.pdparams")
    model.set_state_dict(paddle.load(path_to_model))
    model.eval()
    test_df_x = test_df_x.values.reshape(1, 144, in_var_size)
    test_df_x = paddle.to_tensor(test_df_x, dtype='float32')
    pred = model(test_df_x)
    # normalization decode
    pred = pred * df_x_std['Patv'] + df_x_mean['Patv']
    pred[pred < 0] = 0
    pred[pred > 1521] = 1521
    rst = np.asarray(pred.tolist()*134)
    return rst




