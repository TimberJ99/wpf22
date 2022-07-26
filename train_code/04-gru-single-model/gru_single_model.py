from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime, timedelta

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.io import Dataset
paddle.device.set_device('gpu:1')

day_size = 144
train_day = 225
val_day = 20
in_len = 144
out_len = 288
train_size = train_day * day_size


class WindTurbineDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """
    def __init__(self,
                 df,
                 df_y,
                 indexes,
                 in_len=144,
                 out_len=288
                 ):
        super().__init__()
        self.df = df
        self.df_y = df_y
        self.indexes = indexes
        self.in_len = in_len
        self.out_len = out_len
        print('WindTurbineDataset------init')

    def __getitem__(self, index):
        s_begin = self.indexes[index]
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len
        seq_x = self.df.iloc[s_begin:s_end].astype(np.float32).values
        seq_y = self.df_y.iloc[r_begin:r_end].astype(np.float32).values
        return seq_x, seq_y

    def __len__(self):
        return len(self.indexes)


def val(model, data_loader, criterion):
    # type: (Experiment, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    model.eval()
    validation_loss = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        # batch_x = batch_x.astype('float32') # numpy to tensor
        # batch_y = batch_y.astype('float32') # numpy to tensor
        sample = model(batch_x)
        loss = criterion(sample, batch_y)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss

def adjust_learning_rate(optimizer, epoch, lr):
    # lr = args.lr * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: lr * (0.50 ** (epoch - 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)


class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = False

    def save_checkpoint(self, val_loss, model, path):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        self.best_model = True
        self.val_loss_min = val_loss
        paddle.save(model.state_dict(), path + '/best_model')

    def __call__(self, val_loss, model, path):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

class BaselineGruModel(nn.Layer):
    """
    Desc: GRU model
    """
    def __init__(self, settings):
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=False)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc_group):
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        samples = paddle.zeros([x_enc_group.shape[0], self.output_len, self.out])
        for seq_len in [144, 72, 36, 18, 12, 6, 3]:
            x_enc = x_enc_group[:, -seq_len:, :]
            out_one = int(seq_len * 2)
            x = paddle.zeros([x_enc.shape[0], out_one, x_enc.shape[2]])
            x_enc = paddle.concat((x_enc, x), 1)
            dec, _ = self.lstm(x_enc)
            sample = self.projection(self.dropout(dec))
            samples[:, :out_one, -self.out:] = sample[:, -out_one:, -self.out:]
        return samples


def max_of_day(df, cols=['Etmp']):
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
    df['minute'] = df['t1'].dt.minute
    df.loc[df['Etmp'] == 57, 'Etmp'] = df.loc[df['Etmp'] == 57, 'Itmp'] - 9
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

def build_df(df0):
    """
    build DataFrame
    :param df0: DataFrame
    :return: DataFrame
    """
    df = df0.copy()
    df.reset_index(drop=True, inplace=True)
    df['date'] = df['Day'].apply(lambda x: get_date(x))
    df = cols_concat(df, ["date", "Tmstamp"])
    df['t1'] = pd.to_datetime(df['t1'])
    df = engineering(df)
    df = max_of_day(df, cols=['Etmp'])

    id_col = 'TurbID'
    time_col = 't1'
    df_rolling_stat = fe_rolling_stat(df, id_col=id_col, time_col=time_col, time_varying_cols=['Etmp_Max'])
    df_rolling_stat.fillna(method='bfill', inplace=True)

    df['TurbID__Etmp_Max__1008__max'] = df_rolling_stat['TurbID__Etmp_Max__1008__max']
    return df

def stand_df(df):
    """
    stand dataframe
    :param df: DataFrame
    :return: df_x, df_y, df_x_std, df_x_mean
    """
    x_cols = ['hour', 'minute', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv',
              'Etmp_Max', 'TurbID__Etmp_Max__1008__max']
    df_x = df[x_cols]
    df_tids = pd.get_dummies(df[['TurbID']].astype('str'))

    df_x_std = df_x.std()
    df_x_mean = df_x.mean()
    df_x = (df_x - df_x_mean) / df_x_std
    df_y = df_x[['Patv']]

    df_x = pd.concat([df_tids, df_x], axis=1)
    return df_x, df_y, df_x_std, df_x_mean

def split_df(df):
    """
    split dataframe to train and validation dataset
    :param df: DataFrame
    :return: train DataFrame indexes, validation DataFrame indexes
    """
    train_indexes = np.asarray([])
    val_indexes = np.asarray([])
    indices = df.groupby(by='TurbID').indices
    for tid in indices.keys():
        arr = indices[tid]
        train_indexes = np.hstack((train_indexes, arr[:train_size]))
        val_indexes = np.hstack((val_indexes, arr[train_size:-(in_len + out_len)]))
    train_indexes = train_indexes.astype(np.int32)
    val_indexes = val_indexes.astype(np.int32)
    return train_indexes, val_indexes


def train(df_x, df_y, train_indexes, val_indexes, model_dir):
    '''
    train model
    :param df_x: model object
    :param df_y: validation DataFrame
    :param train_indexes: output columns
    :param val_indexes: output columns
    :param model_dir: model output directory
    '''
    train_dataset = WindTurbineDataset(df_x, df_y, train_indexes, in_len=in_len)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                  num_workers=5)  # drop_last=True, places=['cpu']

    val_dataset = WindTurbineDataset(df_x, df_y, val_indexes, in_len=in_len)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                                num_workers=5)  # drop_last=True, places=['cpu']

    settings = {
        "output_len": out_len,
        "in_var": len(df_x.columns),
        "out_var": 1,
        "dropout": 0.05,
        "lstm_layer": 2,
    }
    model = BaselineGruModel(settings)

    EPOCH_NUM = 100
    lr = 0.00005
    is_debug = True
    clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
    model_optim = paddle.optimizer.Adam(parameters=model.parameters(),
                                        learning_rate=lr,
                                        grad_clip=clip)
    criterion = nn.MSELoss(reduction='mean')
    all_train_loss = []
    all_val_loss = []

    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(EPOCH_NUM):
        epoch_start_time = time.time()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(f'{time_str} =====> epoch:{epoch} start')
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            iter_count += 1
            sample = model(batch_x)
            loss = criterion(sample, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.minimize(loss)
            model_optim.step()

            if iter_count % 2000 == 0:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(f'{time_str} =====> epoch:{epoch}, iter_count:{iter_count}, loss:{loss.item()}')
            # break
        val_loss = val(model, val_dataloader, criterion)
        # break

        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, time:{epoch_end_time - epoch_start_time}")
            epoch_start_time = epoch_end_time
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

        paddle.save(model.state_dict(), f'{model_dir}/{epoch}_model_{in_len}_{out_len}.pdparams')
        paddle.save(model_optim.state_dict(), f"{model_dir}/{epoch}_optim_{in_len}_{out_len}.pdopt")

        # Early Stopping if needed
        early_stopping(val_loss, model, model_dir)
        if early_stopping.early_stop:
            print("Early stopped! ")
            break

        adjust_learning_rate(model_optim, epoch + 1, lr)


def run(dataset_file, model_dir):
    """
    run and train
    :param df_all: DataFrame
    :return: train DataFrame, validation DataFrame
    """
    print('begin')
    df0 = pd.read_csv(dataset_file)
    print('read_csv finished')
    df = build_df(df0)
    print('build_df finished')
    train_indexes, val_indexes = split_df(df)
    print('split_df finished')
    df_x, df_y, df_x_std, df_x_mean = stand_df(df)
    print('stand_df finished')
    train(df_x, df_y, train_indexes, val_indexes, model_dir)
    print('train finished')


if __name__ == '__main__':
    dataset_file = '../datasets/wtbdata_245days.csv'
    model_dir = './gru-single-models/'
    run(dataset_file, model_dir)
    print('finished')
