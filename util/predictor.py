# Modified the provided standard predictor for prescriptor usage by
# making its weights untrainable

# @author Andrew Zhou

import numpy as np
import tensorflow as tf
import keras.backend as K
import numpy as np
import keras
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.models import Model
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
MODEL_WEIGHTS_FILE = os.path.join(MODEL_PATH, 'trained_model_weights.h5')

class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)

# Functions to be used for lambda layers in model
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)

# create predictor with trainable set to false
def get_predictor(weights_path=MODEL_WEIGHTS_FILE, nb_context=1, nb_action=12, lstm_size=32, nb_lookback_days=21):

    # Create context encoder
    context_input = Input(shape=(nb_lookback_days, nb_context),
                          name='context_input')
    x = LSTM(lstm_size, name='context_lstm', trainable=False)(context_input)
    context_output = Dense(units=1,
                           activation='softplus',
                           name='context_dense', trainable=False)(x)

    # Create action encoder
    # Every aspect is monotonic and nonnegative except final bias
    action_input = Input(shape=(nb_lookback_days, nb_action),
                         name='action_input')
    x = LSTM(units=lstm_size,
             kernel_constraint=Positive(),
             recurrent_constraint=Positive(),
             bias_constraint=Positive(),
             return_sequences=False,
             name='action_lstm', trainable=False)(action_input)
    action_output = Dense(units=1,
                          activation='sigmoid',
                          kernel_constraint=Positive(),
                          name='action_dense', trainable=False)(x)

    # Create prediction model
    model_output = Lambda(_combine_r_and_d, name='prediction')(
        [context_output, action_output])
    model = Model(inputs=[context_input, action_input],
                  outputs=[model_output])
    #model.compile(loss='mae', optimizer='adam')
    model.load_weights(weights_path)

    return model