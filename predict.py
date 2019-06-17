#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: predict.py
Code to run prediction for the project
"""


__author__ = 'Ethan'


import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from keras.models import load_model
from fbprophet import Prophet
from data_utils import (
    process_data, get_last_nday, extract_feature,
    build_result_dataframe
)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size for model prediction")
parser.add_argument("--num_thread", default=4, type=int,
                    help="batch size for model prediction")
parser.add_argument("--num_day", default=14, type=int,
                    help="number of days of data to use for prediction")
parser.add_argument('--use_prophet', action='store_true',
                    help="use facebook prophet instead of model for prediction")

args = parser.parse_args()


def prophet_predict(df, ghs):
    """ Run prediction using facebook's prophet

    :param ghs: list of geohash6 to predict
    :param df: the original dataframe that contains all location
    :returns: list of predictions
    :rtype: list of dataframe

    """

    res = []
    for gh in ghs:
        df_ = df.loc[df.geohash6 == gh, ['timestamp', 'demand']].copy()
        df_.rename(columns={'timestamp': 'ds', 'demand': 'y'}, inplace=True)
        df_.loc[:, 'y'] = np.log(df_.loc[:, 'y'] + 1)

        # train prophet model
        model = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False
        ).add_seasonality(
            name='weekly',
            period=7,
            fourier_order=15
        ).add_seasonality(
            name='daily',
            period=1,
            fourier_order=20
        )
        model.fit(df_)

        # run prediction
        future_times = model.make_future_dataframe(
            periods=5, freq='15T', include_history=False)

        forecast = model.predict(future_times)
        forecast = forecast.loc[:, ['ds', 'yhat']]
        forecast['yhat'] = np.exp(forecast['yhat']) - 1

        # build result dataframe
        forecast.rename(columns={'ds': 'timestamp',
                                 'yhat': 'demand'}, inplace=True)
        forecast['geohash6'] = [gh] * len(forecast)

        # calulate 'day' colum of the dataframe
        dtdelta = (forecast.timestamp.dt.date - df_.ds.max().date())
        dtdelta = list(map(lambda x: x.days, dtdelta))
        days = dtdelta + df.day.max()

        # calulate time of day
        tod = list(map(lambda x: x.strftime(
            '%H:%M'), forecast.timestamp.dt.time))
        forecast.loc[:, 'timestamp'] = tod
        forecast.loc[:, 'day'] = days

        res.append(forecast)

    return res


def predict(data_path='data/predict/', model_path='pretrained/model.h5', use_prophet=False):
    """ Main function to run the prediction

    """

    print('Loading and processing data ...')
    # get the processed dataframe
    df = process_data(data_path)

    # predict using only last 14 days of the time series data
    df = get_last_nday(df, args.num_day)
    print('done!')

    gh_list = list(df.geohash6.unique())
    chunk_size = len(gh_list) // (args.num_thread - 1)
    gh_chunks = [gh_list[chunk_size*i:chunk_size *
                         (i+1)] for i in range(args.num_thread)]
    gh_chunks = [ch for ch in gh_chunks if ch]

    pool = Pool(args.num_thread)

    if use_prophet:
        # use facebook's prophet to predict
        p_prophet_predict = partial(prophet_predict, df)

        predictions = []
        preds = pool.map(p_prophet_predict, gh_chunks)
        [predictions.extend(pred) for pred in preds]

        pred_df = pd.concat(predictions, ignore_index=True)

        pred_df.loc[:, 'demand'] = pred_df['demand'].clip_lower(
            0).clip_upper(1)

        return pred_df
    else:
        print('Predict using wavenet model...')
        # load model
        model = load_model(model_path)

        # extract feature for all locations
        print('Extracting features ...')
        p_extract_feature = partial(extract_feature, df)

        features = []

        fts = pool.map(p_extract_feature, gh_chunks)
        [features.extend(ft) for ft in fts]
        print('done!')

        # split features into batches
        feature_chunks = [features[args.batch_size*i:args.batch_size *
                                   (i+1)] for i in range(int(len(features)/args.batch_size) + 1)]
        feature_chunks = [ch for ch in feature_chunks if ch]

        # run prediction using keras model
        print('Predicting ...')
        predictions = []
        for each_chunk in feature_chunks:
            print(len(each_chunk))
            batch = np.concatenate(each_chunk, axis=0)
            pred = model.predict(batch)
            pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))
            predictions.extend(list(pred))
        print('done!')
        print(len(predictions))

        # create result dataframe from prediction results
        print('Constructing result dataframe ...')
        pred_df = []
        for i, pred in enumerate(predictions):
            res = build_result_dataframe(gh_list[i], pred, df)
            pred_df.append(res)

        pred_df = pd.concat(pred_df, ignore_index=True)
        print('done!')

        pred_df.loc[:, 'demand'] = pred_df['demand'].clip_lower(
            0).clip_upper(1)

        return pred_df


if __name__ == '__main__':
    res = predict(use_prophet=args.use_prophet)
    res.to_csv('out/result.csv')
