# -*-Encoding: utf-8 -*-

import numpy as np
import pandas as pd
from copy import deepcopy

pd.set_option('mode.chained_assignment', None)


class TestData(object):
    """
        Desc: Test Data
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,       # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data = deepcopy(self.df_raw)
        self.df_data = self.engineering(self.df_data)
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        '''
        get turbine dataset
        '''
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]
            data = self.df_data[cols]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]
        df = self.df_raw[border1:border2]
        return seq, df

    def get_all_turbines(self):
        '''
        get all turbines DataFrame
        '''
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs

    def engineering(self, df):
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
        df.loc[df['Etmp'] == 57,'Etmp'] = df.loc[df['Etmp'] == 57,'Itmp'] -9
        return df
