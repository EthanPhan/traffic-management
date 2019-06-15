#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: model.py
Code to build model using keras and tensorflow
in this project, the model is a wavenet model
"""


__author__ = 'Ethan'


import keras.backend as K
import keras.layers
from keras.layers import Activation, Lambda, Multiply, Add
from keras.layers import Conv1D, Dropout
from keras.models import Input, Model


def process_dilations(dilations):
    """ helper function to validate and process the dilation rate of
    the wavenet model. The dilation rate at each layer must be a power of 2

    :param dilations: list of dilation rate at each layer
    :returns: valid dilation rate
    :rtype: list

    """
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]

        return new_dilations


def slice(x, seq_length):
    """ get seg_length number of timestep now, 1 day and 1 week before current
    timestep; concatenate them into 1 tensor

    :param x: tensor contains all timestep
    :param seq_length: the length of sequence we want to extract
    :returns: concatinated tensor of last seq_length timestep, seq_length
    timesteps 1 day before and 1 week before current timestep
    :rtype: tensor

    """
    x1 = x[:, -seq_length:, :]
    x2 = x[:, -96:-96+seq_length:, :]
    x3 = x[:, -96*7:-96*7+seq_length:, :]
    out = K.concatenate([x1, x2, x3], axis=-1)
    return out


def wavenet(num_feat,  # type: int
            nb_filters,  # type: int
            kernel_size,  # type: int
            dilations,  # type: List[int]
            max_len,  # type: int
            out_len,  # type: int
            padding='causal',  # type: str
            dropout_rate=0.2,  # type: float
            name='wavenet'):
    dilations = process_dilations(dilations)
    input_layer = Input(shape=(max_len, num_feat))
    x = keras.layers.Reshape((max_len, num_feat))(input_layer)

    skips = []
    for dr in dilations:

        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x)

        # filter convolution
        x_f = Conv1D(filters=nb_filters,
                     kernel_size=kernel_size,
                     padding=padding,
                     dilation_rate=dr)(x)

        # gating convolution
        x_g = Conv1D(filters=nb_filters,
                     kernel_size=kernel_size,
                     padding=padding,
                     dilation_rate=dr)(x)

        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)

        # residual connection
        x = Add()([x, z])

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs
    out = Activation('relu')(Add()(skips))

    # extract the last out_len time steps as the training target
    out = Lambda(slice, arguments={'seq_length': out_len})(out)

    # time-distributed dense layers
    out = Conv1D(128, 3, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(128, 3, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(128, 3, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)

    # final time-distributed dense layers for output
    out = Conv1D(16, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(1, 1, padding='same')(out)

    pred_seq_train = out

    model = Model(input_layer, pred_seq_train)

    return model
