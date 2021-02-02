# covid xprize prescriptor
# Andrew Zhou

import numpy as np
import tensorflow as tf
import keras.backend as K
import numpy as np
import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import Input, Concatenate, Reshape, Add
from tensorflow.keras.layers import LSTM, Multiply
from tensorflow.keras.layers import Lambda, Average
from tensorflow.keras.models import Model

from .predictor import get_predictor

class CalcCasesLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    # inputs are [ratio
    def call(self, inputs):
        ratio, prev_new_cases, curr_total_cases, pop_size = inputs
        pct_inf = tf.math.divide(curr_total_cases, pop_size)

        term_1 = tf.math.subtract(tf.math.multiply(ratio, tf.math.subtract(1.0, pct_inf)), 1.0)
        
        term_2 = tf.math.multiply(tf.reduce_mean(prev_new_cases), WINDOW_SIZE)
        
        return tf.math.maximum(0.0, tf.math.add(tf.math.multiply(term_1, term_2), prev_new_cases[0]))

# testing a stringency constraint
# class StringencyConstraintLayer(Layer):
#     def __init__(self, initial_max, name=None):
#         super().__init__(name=name)
#         self.max_stringency = tf.Variable(
#             initial_value=initial_max,
#             trainable=False
#         )

#     # input is just the avg stringency
#     def call(self, inputs):
#         return tf.math.maximum(0.0, tf.math.multiply(1000000000000.0, tf.math.subtract(inputs, self.max_stringency)))


BATCH_SIZE = 1
NB_LOOKBACK_DAYS = 21
NB_CONTEXT = 1
NB_ACTION = 12
WINDOW_SIZE = 7

# todo: try pushing the weights toward integers
class NPIConstraint(Constraint):
  def __init__(self):
    self.max_npis = tf.convert_to_tensor([3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4], dtype='float32')

  def __call__(self, w):
    clip = tf.clip_by_value(w, clip_value_min=tf.zeros(w.shape), clip_value_max=self.max_npis)
    return clip 

  def get_config(self):
    return {'max_npis': self.max_npis}

class NPILayer(Layer):
    def __init__(self, name=None, constraint=None):
        super().__init__(name=name)
        
        npis = tf.keras.initializers.Zeros()(shape=(NB_ACTION))
        
        self.npis = tf.Variable(
            initial_value=npis,
            trainable=True,
            constraint=constraint
        )

    def call(self, inputs):
        return tf.broadcast_to(self.npis, (tf.shape(inputs)[0], 1, NB_ACTION))

def construct_model(num_days):
    predict_layer = get_predictor()

    outer_context_input = Input(shape=(NB_LOOKBACK_DAYS, NB_CONTEXT), batch_size = BATCH_SIZE, name="outer_context_input")
    outer_action_input = Input(shape=(NB_LOOKBACK_DAYS, NB_ACTION), batch_size = BATCH_SIZE, name="outer_action_input")
    population_input = Input(shape=(), batch_size=BATCH_SIZE, name="population")
    total_cases_input = Input(shape=(), batch_size=BATCH_SIZE, name="total_cases_input")
    prev_new_cases_input = Input(shape=(WINDOW_SIZE,), batch_size=BATCH_SIZE, name="prev_new_cases_input")
    stringency_costs_input = Input(shape=(NB_ACTION,), batch_size=BATCH_SIZE, name="stringency_costs_input")


    calc_layer = CalcCasesLayer()
    add_layer = Add()


    context_layers = []
    for i in range(NB_LOOKBACK_DAYS):
        next_context = outer_context_input[:, i:i+1]
        context_layers.append(next_context)

    action_layers = []
    for i in range(NB_LOOKBACK_DAYS):
        next_action = outer_action_input[:, i:i+1]
        action_layers.append(next_action)


    npi_constraint = NPIConstraint()
    future_action_layers = []
    future_stringency_layers = []

    multiply_layer = Multiply()
    # pass population as a dummy input
    for i in range(num_days):
        future_action = NPILayer(name=f"future_npis_{i}", constraint=npi_constraint)(population_input)
        future_action_layers.append(future_action)

        weighted_stringencies = multiply_layer([future_action, stringency_costs_input])
        future_stringency = Lambda(lambda x: tf.reduce_sum(x), name=f"future_stringency_{i}")(weighted_stringencies)
        future_stringency_layers.append(future_stringency)

    # their code uses day-of actions in the prediction
    # could potentially throw away the first action layer
    action_layers.append(future_action_layers[0])

    concat_layer = Concatenate(axis=1)
    reshape_prediction_layer = Reshape((1,1))

    for i in range(num_days):
        context_grp = concat_layer(context_layers[-NB_LOOKBACK_DAYS:])
        action_grp = concat_layer(action_layers[-NB_LOOKBACK_DAYS:])

        predict = reshape_prediction_layer(predict_layer([context_grp, action_grp]))

        context_layers.append(predict)
        if i + 1 != num_days:
            action_layers.append(future_action_layers[i+1])

    prev_new_cases_layers = []
    for i in range(WINDOW_SIZE):
        prev_new_cases = prev_new_cases_input[:, i:i+1]
        prev_new_cases_layers.append(prev_new_cases)

    current_total_cases_layers = [total_cases_input]

    reshape_context_layer = Reshape(())

    for i in range(num_days):
        reshaped_context = reshape_context_layer(context_layers[NB_LOOKBACK_DAYS+i])
        next_new_cases = calc_layer([reshaped_context, prev_new_cases_layers[-WINDOW_SIZE:], current_total_cases_layers[-1], population_input])
        next_total_cases = add_layer([current_total_cases_layers[-1], next_new_cases])

        prev_new_cases_layers.append(next_new_cases)
        current_total_cases_layers.append(next_total_cases)

    total_new_cases = reshape_context_layer(add_layer(prev_new_cases_layers[WINDOW_SIZE:]))
    #total_cases = add_layer([total_new_cases, total_cases_input])
    avg_stringency = Average()(future_stringency_layers)

    return Model(inputs=[outer_context_input, outer_action_input, population_input, total_cases_input, prev_new_cases_input, stringency_costs_input], outputs=[total_new_cases, avg_stringency])