from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

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


def max_of_day(df, cols=['Etmp', 'Itmp']):
    """
    Calculate the maximum temperature of the day (average of the first 6 items)
    :param df: DataFrame
    :param cols: column list
    :return: new DataFrame
    """
    df_tmp = df.groupby(by=['TurbID', 'Day'])
    for col in cols:
        df_etmp = df_tmp[col].nlargest(6)
        df_etmp = df_etmp.groupby(by=['TurbID', 'Day']).mean()
        df = df.merge(df_etmp, how='left', on=['TurbID', 'Day'], suffixes=('', '_Max'))
    return df


def fe_rolling_stat(df, id_col, time_col, time_varying_cols=['Etmp'], window_size=[144 * 7]):
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
    df = df.sort_values(by=time_col)
    add_feas = []
    key = id_col
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            ops = ['max']
            if val == 'Wspd':
                ops = ['mean']
            for op in ops:  # ['mean', 'std', 'median', 'max', 'min', 'kurt', 'skew']:
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

    df_rolling_stat = result.merge(df[[id_col, time_col] + add_feas], on=[id_col, time_col], how='left')[add_feas]
    df_rolling_stat.bfill(inplace=True)
    return df_rolling_stat


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
    # df.replace(to_replace=np.nan, value=0, inplace=True)
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
    # df.fillna(method='bfill', inplace=True)
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


def assign_mean(item):
    """
    fill nan value by mean
    :param item: DataFrame
    :return: DataFrame
    """
    mean_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
    item[mean_cols] = item[mean_cols].fillna(item[mean_cols].mean().round(decimals=2), axis=0)
    return item

def fixed_df(test_df):
    """
    fixed df
    :param test_df: DataFrame
    :return: new DataFrame
    """
    test_df_filter = filter_no_valid(test_df)
    test_df_filter = test_df_filter.groupby(['Day', 'Tmstamp']).apply(assign_mean)

    nan_result = []
    for turb in range(1, 135):
        turb_result = []
        cur = test_df_filter.loc[(test_df_filter.TurbID == turb)]
        for day in range(1, 15):
            turb_result.append(len(cur.loc[(cur.Day == day) & (cur.Patv.isna())]))
        nan_result.append(turb_result)
    pd_nan_result = pd.DataFrame(nan_result)

    for turb in range(134):
        cur_turb = pd_nan_result.loc[turb]
        day_index = cur_turb.loc[cur_turb <= 28].index.values + 1
        # print("Turb-", turb + 1, ": ", cur_turb.sum(), ", ", len(day_index))
        fill_index = test_df_filter.loc[(test_df_filter.TurbID == turb + 1) & \
                                        (test_df_filter.Day.isin(day_index))].index
        test_df_filter.loc[fill_index] = test_df_filter.loc[fill_index].fillna(method='bfill', axis=0)
        test_df_filter.loc[fill_index] = test_df_filter.loc[fill_index].fillna(method='ffill', axis=0)
    return test_df_filter

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


def linear_regression_val(model, df_val, x_cols_lr, y_cols):
    """
    linear regression validation
    :param model: model object
    :param df_val: validation DataFrame
    :param x_cols_lr: input columns
    :param y_cols: output columns
    :return: rmse and mae
    """
    val = model.predict(df_val[x_cols_lr])
    rmse = _get_score_metric(df_val[y_cols], val, 'rmse')
    mae = _get_score_metric(df_val[y_cols], val, 'mae')
    return rmse, mae


def linear_regression_train(df_train, df_val, y_cols, model_dir):
    """
    linear regression training, save model file to directory
    :param df_train: model object
    :param df_val: validation DataFrame
    :param y_cols: output columns
    :param model_dir: model output directory
    """
    model = LinearRegression()
    x_cols_lr = ['TurbID', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv', 'hour_sin',
                 'hour_cos', 'cross', 'Etmp_Max', 'TurbID__Wspd__lag__1', 'TurbID__Wspd__lag__2',
                 'TurbID__Wspd__lag__3',
                 'TurbID__Wspd__lag__4', 'TurbID__Wspd__lag__5', 'TurbID__Wspd__lag__6', 'TurbID__Wspd__lag__7',
                 'TurbID__Wspd__lag__8', 'TurbID__Wspd__lag__9', 'TurbID__Wspd__lag__10', 'TurbID__Wspd__lag__11',
                 'TurbID__Wspd__lag__12', 'TurbID__Etmp__lag__1', 'TurbID__Etmp__lag__2', 'TurbID__Etmp__lag__3',
                 'TurbID__Etmp__lag__4', 'TurbID__Etmp__lag__5', 'TurbID__Etmp__lag__6', 'TurbID__Etmp__lag__7',
                 'TurbID__Etmp__lag__8', 'TurbID__Etmp__lag__9', 'TurbID__Etmp__lag__10', 'TurbID__Etmp__lag__11',
                 'TurbID__Etmp__lag__12', 'TurbID__Wspd__diff__1', 'TurbID__Wspd__diff__2', 'TurbID__Wspd__diff__3',
                 'TurbID__Wspd__diff__4', 'TurbID__Wspd__diff__5', 'TurbID__Wspd__diff__6', 'TurbID__Wspd__diff__7',
                 'TurbID__Wspd__diff__8', 'TurbID__Wspd__diff__9', 'TurbID__Wspd__diff__10', 'TurbID__Wspd__diff__11',
                 'TurbID__Wspd__diff__12', 'TurbID__Etmp__diff__1', 'TurbID__Etmp__diff__2', 'TurbID__Etmp__diff__3',
                 'TurbID__Etmp__diff__4', 'TurbID__Etmp__diff__5', 'TurbID__Etmp__diff__6', 'TurbID__Etmp__diff__7',
                 'TurbID__Etmp__diff__8', 'TurbID__Etmp__diff__9', 'TurbID__Etmp__diff__10', 'TurbID__Etmp__diff__11',
                 'TurbID__Etmp__diff__12', 't1_hour', 't1_minute', 'TurbID__Etmp_Max__1008__max']
    model.fit(df_train[x_cols_lr], df_train[y_cols])
    rmse, mae = linear_regression_val(model, df_val, x_cols_lr, y_cols)
    joblib.dump(model, f'{model_dir}/288-linear-{round(rmse.mean(), 3)}-{round(mae.mean(), 3)}_245days.pkl')


def polynomial_linear_regression_val(linear2, df_val, x_cols_pipeline, y_cols):
    """
    polynomial and linear regression validation
    :param model: model object
    :param df_val: validation DataFrame
    :param x_cols_lr: input columns
    :param y_cols: output columns
    :return: rmse and mae
    """
    val_linear2 = linear2.predict(df_val[x_cols_pipeline])
    rmse2 = _get_score_metric(df_val[y_cols], val_linear2, 'rmse')
    mae2 = _get_score_metric(df_val[y_cols], val_linear2, 'mae')
    return rmse2, mae2


def polynomial_linear_regression_train(df_train, df_val, y_cols, model_dir):
    """
    polynomial and linear regression training, save model file to directory
    :param df_train: model object
    :param df_val: validation DataFrame
    :param y_cols: output columns
    :param model_dir: model output directory
    """
    x_cols_pipeline = ['Wspd','Etmp','Patv','TurbID__Patv__lag__1', 'TurbID__Patv__lag__2','TurbID__Wspd__lag__1','TurbID__Wspd__lag__2']
    linear2 = Pipeline([('poly', PolynomialFeatures(degree=4)),
                        ('linear', LinearRegression())])
    linear2.fit(df_train[x_cols_pipeline], df_train[y_cols])
    rmse2, mae2 = polynomial_linear_regression_val(linear2, df_val, x_cols_pipeline, y_cols)
    joblib.dump(linear2, f'{model_dir}/288-pipline-{round(rmse2.mean(), 3)}-{round(mae2.mean(), 3)}_245days.pkl')

def build_df_x(df0):
    """
    build input features DataFrame
    :param df0: DataFrame
    :return: DataFrame
    """
    df = df0.copy()
    # column "Day" to Date type
    df['Day'] = df['Day'].apply(lambda x: get_date(x))
    # column "Day" and "Tmstamp" concat to datetime
    df = cols_concat(df, ["Day", "Tmstamp"])
    df = df[['TurbID', 'Day', 't1', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']]
    df['t1'] = pd.to_datetime(df['t1'])
    df = engineering(df)
    df = max_of_day(df, cols=['Etmp'])

    id_col = 'TurbID'
    time_col = 't1'
    time_varying_cols = ['Wspd', 'Etmp', 'Patv']

    df_rolling_stat = fe_rolling_stat(df, id_col=id_col, time_col=time_col, time_varying_cols=['Etmp_Max'])
    lag_range = list(range(1, 13, 1))
    df_lag = fe_lag(df, id_col=id_col, time_col=time_col, time_varying_cols=time_varying_cols, lag=lag_range)
    df_diff = fe_diff(df, id_col=id_col, time_col=time_col, time_varying_cols=time_varying_cols, lag=lag_range)
    df_time = fe_time(df, time_col=time_col)

    df_x = feature_combination([df, df_lag, df_diff, df_time, df_rolling_stat])  #
    return df_x

def build_df_y(df_x):
    """
    build output features DataFrame
    :param df_x: DataFrame
    :return: DataFrame
    """
    output_nums = 6 * 24 * 2
    df_group = df_x.groupby('TurbID')[['Patv']]
    df_y_arr = []
    for i in range(1, output_nums + 1):
        # print(i)
        shift_df = df_group.shift(0 - i)
        shift_df.rename(columns={'Patv': f'y_{i}'}, inplace=True)
        df_y_arr.append(shift_df)

    df_y = pd.concat(df_y_arr, axis=1)
    return df_y

def split_df(df_all):
    """
    split dataframe to train and validation dataset
    :param df_all: DataFrame
    :return: train DataFrame, validation DataFrame
    """
    time_col = 't1'
    valid_days = 20
    delta = timedelta(days=valid_days)
    valid_time_split = df_all[time_col].max() - delta
    df_val = df_all.loc[df_all[time_col] > valid_time_split]
    df_train = df_all.loc[df_all[time_col] <= valid_time_split]
    return df_train, df_val

def run(dataset_file, model_dir):
    """
    run and train
    :param df_all: DataFrame
    :return: train DataFrame, validation DataFrame
    """
    print('begin')
    df0 = pd.read_csv(dataset_file)
    print('load dataset finished')
    df0 = fixed_df(df0)
    print('fixed df finished')
    df_x = build_df_x(df0)
    print('build_df_x finished')
    df_y = build_df_y(df_x)
    print('build_df_y finished')
    y_cols = list(df_y.columns)

    df_all = pd.concat([df_x, df_y], axis=1)
    df_all.dropna(inplace=True)
    print('df_all finished')
    df_train, df_val = split_df(df_all)
    print('split_df finished')
    linear_regression_train(df_train, df_val, y_cols, model_dir)
    print('linear_regression_train finished')
    polynomial_linear_regression_train(df_train, df_val, y_cols, model_dir)
    print('polynomial_linear_regression_train finished')

if __name__ == '__main__':
    dataset_file = '../datasets/wtbdata_245days.csv'
    model_dir = './linear_models/'
    run(dataset_file, model_dir)
    print('finished')
