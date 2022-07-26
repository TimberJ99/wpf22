# -*-Encoding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class Scaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class WindTurbineData(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
    """
    first_initialized = False
    df_raw = None

    def __init__(self, data_path,
                 filename='sdwpf_baidukddcup2022_full.csv',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Patv',
                 scale=True,
                 start_col=3,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 total_days=30,     # 30 days
                 farm_capacity=134,
                 is_test=False
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id
        self.farm_capacity = farm_capacity
        self.is_test = is_test
        # If needed, we employ the predefined total_size (e.g. one month)
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        #
        if self.is_test:
            if not WindTurbineData.first_initialized:
                self.__read_data__()
                WindTurbineData.first_initialized = True
        else:
            self.__read_data__()
        if not self.is_test:
            self.data_x, self.data_y = self.__get_data__(self.tid)
        #if WindTurbineData.scaler_collection is None:
        self.scaler_collection = []
        for i in range(self.farm_capacity):
            self.scaler_collection.append(None)

    def __read_data__(self):
        WindTurbineData.df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        WindTurbineData.df_raw.replace(to_replace=np.nan, value=0, inplace=True)
        WindTurbineData.df_raw = self.engineering(WindTurbineData.df_raw)
        # df = df.fillna(0)
        #self.df_raw = engineering_data


    def __get_turbine__(self, turbine_id):
        border1s = [turbine_id * self.total_size,
                    turbine_id * self.total_size + self.train_size - self.input_len
                    ]
        border2s = [turbine_id * self.total_size + self.train_size,
                    turbine_id * self.total_size + self.train_size + self.val_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'MS':
            cols = WindTurbineData.df_raw.columns[self.start_col:]
            df_data = WindTurbineData.df_raw[cols]
        elif self.task == 'S':
            df_data = WindTurbineData.df_raw[[turbine_id, self.target]]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        train_data = df_data[border1s[0]:border2s[0]]
        if self.scaler_collection[turbine_id] is None:
            scaler = Scaler()
            scaler.fit(train_data.values)
            self.scaler_collection[turbine_id] = scaler
        self.scaler = self.scaler_collection[turbine_id]
        res_data = df_data[border1:border2]
        if self.scale:
            res_data = self.scaler.transform(res_data.values)
        else:
            res_data = res_data.values
        return res_data

    def __get_data__(self, turbine_id):
        data_x = self.__get_turbine__(turbine_id)
        data_y = data_x
        return data_x, data_y

    def get_scaler(self, turbine_id):
        if self.is_test and self.scaler_collection[turbine_id] is None:
            self.__get_turbine__(turbine_id)
        return self.scaler_collection[turbine_id]

    def __getitem__(self, index):
        #
        # Rolling window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        # In our case, the rolling window is adopted, the number of samples is calculated as follows
        if self.set_type < 2:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise,
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    def engineering(self, df):
        '''
        engineering_keys = ["Wspd",  "Patv"]
        def add_features1(df, keys, step_length):
            for key in keys:
                for step in range(1, step_length + 1):
                    df[key + "_lag" + str(step)] = df.groupby('TurbID')[key].shift(step)
                    df[key + "_lag_back" + str(step)] = df.groupby('TurbID')[key].shift(-step)
            print("Step-1...Completed")
            return df

        def add_features3(df, keys, step_length):
            for key in keys:
                for step in range(1, step_length + 1):
                    df[key + "_diff" + str(step)] = df[key] - df[key + "_lag" + str(step)]
                    df[key + "_diffback" + str(step)] = df[key] - df[key + "_lag_back" + str(step)]
            print("Step-3...Completed")
            return df

        def add_features2(df, keys):
            for key in keys:
                df[["144_in_sum_" + key, "144_in_min_" + key, "144_in_max_" + key, "144_in_mean_" + key]] = \
                    (df.groupby('TurbID')[key].rolling(window=144, min_periods=1).agg(
                        {"144_in_sum_" + key: "sum",
                         "144_in_min_" + key: "min",
                         "144_in_max_" + key: "max",
                         "144_in_mean_" + key: "mean"
                         }).reset_index(level=0, drop=True))
            return df
            # df = df.fillna(0)

        def filter_ndir(ndir):
            if (ndir > 360):
                return ndir - 360
            if (ndir < 0):
                return ndir + 360
            return ndir

        def filter_wdir(wdir):
            if (wdir < 0):
                return -wdir
            return wdir

        def filter_tmp(tmp):
            if (tmp < -25):
                return -25
            return tmp

        df['Ndir'] = df.Ndir.apply(filter_ndir)
        df['Wdir'] = df.Wdir.apply(filter_wdir)
        df['Etmp'] = df.Etmp.apply(filter_tmp)
        df['Itmp'] = df.Itmp.apply(filter_tmp)
        df = df[["TurbID", "Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Prtv", "Patv"]]
        engineering_data = add_features1(df, engineering_keys, 2)

        engineering_data = engineering_data.fillna(0)
        engineering_data.head(10)
        engineering_data = add_features3(engineering_data, engineering_keys, 2)
        engineering_data = engineering_data.fillna(0)
        print(engineering_data[engineering_data['Patv_lag1'].isna()].Patv_lag1)
        return engineering_data
        '''
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
        df.loc[df['Etmp'] == 57,'Etmp'] = df.loc[df['Etmp'] == 57,'Itmp'] -9
        return df
