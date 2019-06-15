#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: data_utils.py
Helper functions to deal with data
"""


__author__ = 'Ethan'


import os
import warnings
import pickle
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool


def process_data(path='data/predict/'):
    """ Read the csv file and do some processing on it:
      - sort by geohash and timestamp
      - fill the missing data with 0

    :param path: path to the folder that contains the csv file
    :returns: processed dataframe
    :rtype: pandas.core.frame.DataFrame

    """
    glob_path = os.path.join(path, '*.csv')
    csv_file = glob.glob(glob_path)

    if not csv_file:
        raise FileNotFoundError(
            path + ' directory does not contain any csv files!')
    elif len(csv_file) > 1:
        warn = path + ' contains more than one csv files, using ' + csv_file[0]
        warnings.warn(warn)

    csv_file = csv_file[0]

    # load the dataframe fom csv file
    try:
        df = pd.read_csv(csv_file)
    except:
        raise RuntimeError('Cannot read ' + csv_file)

    # convert timestamp column to datetime format
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format='%H:%M') + \
        df.day.values * np.timedelta64(1, 'D')

    start_time = df.timestamp.values.min()
    end_time = df.timestamp.values.max()

    mulidx = pd.MultiIndex.from_product(
        [df['geohash6'].unique(), pd.date_range(start_time, end_time, freq='15T')],
        names=['geohash6', 'timestamp'])

    result = df.set_index(['geohash6', 'timestamp'])\
               .reindex(mulidx, fill_value=0).reset_index()

    return result
