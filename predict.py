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


def predict(data_path='data/predict/', model_path='pretrained/model.h5', use_prophet=False):
    """ Main function to run the prediction

    """

    print('Loading and processing data ...')
    # get the processed dataframe
    df = process_data(data_path)

    # predict using only last 14 days of the time series data
    df = get_last_nday(df, args.num_day)
    print('done!')

    if use_prophet:
        # use facebook's prophet to predict
        pass
    else:
        print('Predict using wavenet model...')
        # load model
        model = load_model(model_path)

        # extract feature for all locations
        print('Extracting features ...')
        p_extract_feature = partial(extract_feature, df)

        features = []

        gh_list = list(df.geohash6.unique())
        chunk_size = len(gh_list) // (args.num_thread - 1)
        gh_chunks = [gh_list[chunk_size*i:chunk_size *
                             (i+1)] for i in range(args.num_thread)]
        gh_chunks = [ch for ch in gh_chunks if ch]

        pool = Pool(args.num_thread)
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

        return pred_df


if __name__ == '__main__':
    res = predict(use_prophet=args.use_prophet)
    res.to_csv('out/result.csv')
