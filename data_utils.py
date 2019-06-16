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

    start_time = df.timestamp.min()
    end_time = df.timestamp.max()

    mulidx = pd.MultiIndex.from_product(
        [df['geohash6'].unique(), pd.date_range(start_time, end_time, freq='15T')],
        names=['geohash6', 'timestamp'])

    result = df.set_index(['geohash6', 'timestamp'])\
               .reindex(mulidx, fill_value=0).reset_index()

    # re-fill the day the day column for zero filled data
    days = result['timestamp'].dt.date - start_time.date()
    days = list(map(lambda x: x.days + 1, days))
    result.loc[:, 'day'] = days

    return result


def get_last_nday(df, nday=14):
    """ Get the last nday number of days from the data frame

    :param df: the original dataframe
    :param nday: number of day the data will be extracted
    :returns: a dataframe contain data in the last nday days
    :rtype: pandas.core.frame.DataFrame

    """

    start_time = df.timestamp.values.max() - np.timedelta64(nday, 'D')

    if start_time < df.timestamp.values.min():
        # the dataframe has less than nday of data
        # zero padding it
        end_time = df.timestamp.values.max()
        mulidx = pd.MultiIndex.from_product(
            [df['geohash6'].unique(), pd.date_range(
                start_time, end_time, freq='15T')],
            names=['geohash6', 'timestamp'])

        df = df.set_index(['geohash6', 'timestamp'])\
               .reindex(mulidx, fill_value=0).reset_index()

    res = df[df['timestamp'] > start_time]
    return res


def extract_feature(df, ghs, with_batch_dim=True):
    """ extract features from the dataframe. The list of
    feature is as follows:
     - normalized time of day
     - demand

    :param df: the dataframe that contain data of all location
    :param ghs: list of geohash to extract
    :param with_batch_dim: does the feature array contain batch dimension
    :returns: features extracted from df
    :rtype: numpy array

    """
    features = []
    for gh in ghs:
        df_ = df.loc[df.geohash6 == gh, :].copy()
        new_series = (60 * df_['timestamp'].dt.hour +
                      df_['timestamp'].dt.minute) / (24 * 60)
        df_.loc[:, 'normalized_time'] = new_series.values
        ft = df_.loc[:, ['normalized_time', 'demand']].values

        if with_batch_dim:
            # expand the features array to contain batch dimension
            ft = np.expand_dims(ft, 0)
        features.append(ft)
    return features


def build_result_dataframe(gh, pred, df):
    """ Construct a datarame that contain the prediction.

    :param gh: the geohas6 code of the prediction
    :param pred: numpy array of prediction
    :param df: the dataframe used for prediction
    :returns: prediction dataframe
    :rtype: pandas.core.frame.DataFrame

    """
    # generate a sequence of timestamp
    start_time = df.timestamp.values.max() + np.timedelta64(15, 'm')
    timestamps = pd.date_range(start_time, periods=len(pred), freq='15T')

    # calulate 'day' colum of the dataframe
    dtdelta = (timestamps.date - df.timestamp.max().date())
    dtdelta = list(map(lambda x: x.days, dtdelta))
    days = dtdelta + df.day.max()

    # calulate time of day
    tod = list(map(lambda x: x.strftime('%H:%M'), timestamps.time))

    # construct the result dictionary
    res = {'geohash6': [gh] * len(pred),
           'day': days,
           'timestamp': tod,
           'demand': pred
           }

    return pd.DataFrame(res)
