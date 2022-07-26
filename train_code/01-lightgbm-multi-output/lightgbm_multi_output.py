# -*-Encoding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from datetime import datetime, timedelta
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import random
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)

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
    result[prefix + 'hour'] = df[time_col].dt.hour
    result[prefix + 'minute'] = df[time_col].dt.minute
    return result


def fe_rolling_stat(df, id_col, time_col, time_varying_cols = ['Etmp'], window_size = [144*7]):
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
            for op in ['max']:
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


def feature_filter(train):
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


def SMAPE(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated target values
    :return: Symmetric Mean Absolute Percentage Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    score = np.where(np.isnan(score), 0, score)
    return score

def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated target values
    :return: Mean Absolute Percentage Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    idx = y_true > 0
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    score = np.abs(y_pred - y_true) / np.abs(y_true)
    return np.mean(score)

def _get_score_metric(y_true, y_pred, metric='mape'):
    '''
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs). Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs). Estimated target values.
    :param metric: str, one of ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle', 'smape'], default = 'mape'.
    :return:
    '''
    y_true = np.array(y_true) if type(y_true) == list else y_true
    y_pred = np.array(y_pred) if type(y_true) == list else y_pred

    if metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    if metric == 'mape':
        return MAPE(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'rmse':
        return np.mean((y_true - y_pred) ** 2) ** 0.5
    elif metric == 'msle':
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    elif metric == 'rmsle':
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2) ** 0.5
    elif metric == 'smape':
        return np.mean(SMAPE(y_true, y_pred))
    return (y_true == y_pred).sum()


def process(df, id_col, time_col, time_varying_cols, lag):
    """
    batch process featues
    :param df: DataFrame
    :param id_col: group by id column
    :param time_col: time column
    :param time_varying_cols: time varying columns
    :param lag: lag list
    :return: last DataFrame
    """
    use_columns = ['TurbID', 't1', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
    df['Day'] = df['Day'].apply(lambda x: get_date(x))
    df = cols_concat(df, ["Day", "Tmstamp"])
    df = df[use_columns]
    df['t1'] = pd.to_datetime(df['t1'])
    df.replace(to_replace=np.nan, value=0, inplace=True)

    df = engineering(df)
    df_rolling_stat = fe_rolling_stat(df,
                                      id_col=id_col,
                                      time_col=time_col
                                      )

    df_lag = fe_lag(df,
                    id_col=id_col,
                    time_col=time_col,
                    time_varying_cols=time_varying_cols,
                    lag=lag)
    df_diff = fe_diff(df,
                      id_col=id_col,
                      time_col=time_col,
                      time_varying_cols=time_varying_cols,
                      lag=lag)
    df_time = fe_time(df, time_col=time_col)
    df_all = feature_combination([df, df_lag, df_diff, df_time, df_rolling_stat])  #
    return df_all

def train(dataset_file, model_dir):
    """
    model train
    :param dataset_file: dataset file full path
    :param model_dir: model output directory
    """

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, mode=0o777, exist_ok=True)
    id_col = 'TurbID'
    time_col = 't1'
    target_col = 'Patv'

    target_col_y = 'y'
    category_cols = [id_col]
    valid_days = 21
    time_varying_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv', 'cross']

    df = pd.read_csv(dataset_file)
    lag_range = list(range(1, 7, 1))
    df_all = process(df=df,
                     id_col=id_col,
                     time_col=time_col,
                     time_varying_cols=time_varying_cols,
                     lag=lag_range
                     )
    for time_step in range(1, 289):
        # print(df_all.info())
        df_all['y'] = df_all.groupby(id_col)[target_col].shift(-time_step)
        train = df_all.loc[df_all['y'].notnull()]
        used_features = feature_filter(train)
        train.index = range(len(train))

        delta = timedelta(days=valid_days)
        valid_time_split = train[time_col].max() - delta
        valid_idx = train.loc[train[time_col] > valid_time_split].index
        train_idx = train.loc[train[time_col] <= valid_time_split].index
        quick = False
        if quick:
            lr = 0.1
            Early_Stopping_Rounds = 150
        else:
            lr = 0.006883242363721497  # 0.006883242363721497
            Early_Stopping_Rounds = 300

        N_round = 5000
        Verbose_eval = 100
        metric = 'rmse'
        params = {'num_leaves': 61,
                  'min_child_weight': 0.03454472573214212,
                  'feature_fraction': 0.3797454081646243,
                  'bagging_fraction': 0.4181193142567742,
                  'min_data_in_leaf': 96,
                  'objective': 'regression',
                  "metric": metric,
                  'max_depth': -1,
                  'learning_rate': lr,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "verbosity": -1,
                  'reg_alpha': 0.3899927210061127,
                  'reg_lambda': 0.6485237330340494,
                  'random_state': 47,
                  'num_threads': 16
                  }

        category = category_cols
        folds_metrics = []
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = train[used_features].columns

        N_MODEL = 1.0
        # for model_i in tqdm(range(int(N_MODEL))):
        model_i = 0

        # params['seed'] = model_i + 1123

        start_time = time()
        print('Training on model {}'.format(time_step))

        ## All data
        # train[target_col_y] = np.log1p(train[target_col_y])
        trn_data = lgb.Dataset(train.iloc[train_idx][used_features], label=train.iloc[train_idx][target_col_y],
                               categorical_feature=category)
        val_data = lgb.Dataset(train.iloc[valid_idx][used_features], label=train.iloc[valid_idx][target_col_y],
                               categorical_feature=category)

        clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[val_data],
                        verbose_eval=Verbose_eval, early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror
        val = clf.predict(train.iloc[valid_idx][used_features])
        cur_metric = _get_score_metric(train.iloc[valid_idx][target_col_y], val, metric)
        print(f'{metric}: {cur_metric}')
        ## All data
        trn_data = lgb.Dataset(train[used_features], label=train[target_col_y], categorical_feature=category)
        clf = lgb.train(params, trn_data, num_boost_round=int(clf.best_iteration * 1.05),
                        valid_sets=[trn_data], verbose_eval=Verbose_eval)
        val = clf.predict(train.iloc[valid_idx][used_features])
        print(val)

        cur_metric = _get_score_metric(train.iloc[valid_idx][target_col_y], val, metric)
        print(f'{metric}: {cur_metric}')
        file_name = '{}_model_245_time_v{}_m{}_{}'.format(time_step, valid_days, metric, cur_metric)
        # feature_importances.to_csv('fillnan_by_group_models_245_alltrain/' + file_name + '.csv')

        clf.save_model(f'{model_dir}/{file_name}.m')
        df_all.drop(['y'], axis=1, inplace=True)
    print(".............................Done")


if __name__ == '__main__':
    dataset_file = '../datasets/wtbdata_245days_filternan_by_group.csv'
    model_dir = './lightgbm_models/'
    train(dataset_file, model_dir)
    print('finished')