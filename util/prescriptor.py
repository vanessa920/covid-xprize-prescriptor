# Using the Keras functional API, constructs a network that functions as our
# prescriptor. For num_days days and NB_ACTION NPIs per day, we have 
# num_days*NB_ACTION trainable weights, each representing the value of a 
# specific NPI.
# 
# Note: NB_ACTION here should be 12, the number of NPIs per day. NB_LOOKBACK_DAYS
# is the number of lookback days the standard predictor uses, which should be 21. 
# WINDOW_SIZE is the standard predictor's window size for converting 
# PredictionRatio into NewCases, which should be 7. Note also that "context" 
# refers to the PredictionRatio for a a day and "action" refers to the set of 
# NB_ACTION NPIs for a day.
# 
# Our input is the starting conditions for a GeoID: the previous NB_LOOKBACK_DAYS
# PredictionRatios, the previous NB_LOOKBACK_DAYS sets of NPIs, the total 
# population, the number of confirmed cases for the day prior to the start date, 
# and the number of new cases for each of the WINDOW_SIZE prior days.
#
# NB_LOOKBACK days is the number of lookback days the standard predictor uses,
# which should be 21. WINDOW_SIZE is the standard predictor's window size, for
# converting PredictionRatio into NewCases, which should be 7.
#
# Our output is the number of new cases in the period of interest, assuming
# our model weights are the NPIs for each day.
#
# Here we take advantage of the predictor being a differentiable neural network,
# which allows us to efficiently calculuate the gradients of the case number
# with respect to the individual NPIs. We use these gradients in a custom
# gradient update function in our trainer.
#
# Note that our batch size will only ever be one, and we will only ever train
# a network on a single input. The process will then yield the NPIs for that
# specific case. If we want to prescribe for another GeoID, we will reset
# the weights to zero and start the process over. 
#
# All the layers of this network use tensorflow ops (rather than python control
# logic) so it can be compiled as a graph with @tf.function.
#
# The key here is we can reuse the compiled graph for both queries, as long
# as the number of days is the same. Thus we can build the graph on the first
# call and use the efficient static graph in future queries, avoiding the
# inefficiencies of the dynamic approach.
#
# (NB: This approach is extensible to handling different numbers of days
# with the same graph. These extensions have not been implemented.)

# @author Andrew Zhou

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


BATCH_SIZE = 1
NB_LOOKBACK_DAYS = 21
NB_CONTEXT = 1
NB_ACTION = 12
WINDOW_SIZE = 7

from .predictor import get_predictor

# Replicate the logic from the XPrizePredictor to transform an input of a
# a PredictionRatio, previous new cases, current total cases, and 
# population size into a case number corresponding to that 
# PredictionRatio. See _convert_ratio_to_new_cases.
class CalcCasesLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs):
        ratio, prev_new_cases, curr_total_cases, pop_size = inputs
        pct_inf = tf.math.divide(curr_total_cases, pop_size)

        term_1 = tf.math.subtract(tf.math.multiply(ratio, tf.math.subtract(1.0, pct_inf)), 1.0)
        
        term_2 = tf.math.multiply(tf.reduce_mean(prev_new_cases), WINDOW_SIZE)
        
        return tf.math.maximum(0.0, tf.math.add(tf.math.multiply(term_1, term_2), prev_new_cases[0]))

class NPIConstraint(Constraint):
    def __init__(self, max_val):
        self.max_val = max_val
        
    def __call__(self, w):
        clip = tf.clip_by_value(w, clip_value_min=tf.zeros(w.shape),
                               clip_value_max=self.max_val)
        return clip

    def get_config(self):
        return {'max_val': self.max_val}

# Layer for a single NPI. We create nodes for each NPI because it eases
# the gradient computation and application process.
class SingleNPI(Layer):
    def __init__(self, max_val = None, name=None, constraint=None):
        super().__init__(name=name)
        npi = tf.keras.initializers.Zeros()(shape=())
        
        self.npi = tf.Variable(
            initial_value=npi,
            trainable=True,
            constraint=constraint
        )
        
    def call(self, inputs):
        return tf.broadcast_to(self.npi, (tf.shape(inputs)[0], 1, 1))

# Take NB_ACTION SingleNPIs as input and reshape them into a shape that the
# prediction layer can accept.
class NPILayer(Layer):
    def __init__(self, name=None, constraint=None):
        super().__init__(name=name)
        self.max_values = tf.Variable(
            initial_value = tf.reshape([3., 3., 2., 4., 2., 3., 2., 4., 2., 3., 2., 4.],
                                      shape=(1,1,NB_ACTION)),
            trainable=False
        )

    # Unsure if the clip matters
    def call(self, inputs):    
        return tf.clip_by_value(tf.concat(inputs, axis=2), 
                                clip_value_min=tf.zeros((1,1,NB_ACTION)), 
                                clip_value_max=self.max_values)

# return a Model created with the functional API
def construct_model(num_days):
    # Create a "layer" with the standard predictor
    predict_layer = get_predictor()

    # Our five inputs
    outer_context_input = Input(shape=(NB_LOOKBACK_DAYS, NB_CONTEXT), 
                                batch_size = BATCH_SIZE, name="outer_context_input")
    outer_action_input = Input(shape=(NB_LOOKBACK_DAYS, NB_ACTION), 
                               batch_size = BATCH_SIZE, name="outer_action_input")
    population_input = Input(shape=(), batch_size=BATCH_SIZE, name="population")
    total_cases_input = Input(shape=(), batch_size=BATCH_SIZE, name="total_cases_input")
    prev_new_cases_input = Input(shape=(WINDOW_SIZE,), batch_size=BATCH_SIZE, name="prev_new_cases_input")



    # Each layer in this array is the context for a specific day. Start by
    # appending the input tensor contexts.
    context_layers = []
    for i in range(NB_LOOKBACK_DAYS):
        next_context = outer_context_input[:, i:i+1]
        context_layers.append(next_context)

    # Each layer in this array is the set of actions for a specific day. Start
    # by appending the input tensor actions.
    #
    # Note their code uses day-of actions in the prediction, so the first action
    # layer in the input is never used. We could potentially throw away the 
    # first action layer from NB_LOOKBACK_DAYS ago. Keep for now for simplicity's sake.
    action_layers = []
    for i in range(NB_LOOKBACK_DAYS):
        next_action = outer_action_input[:, i:i+1]
        action_layers.append(next_action)

    # Constraints for the layers
    max_npis = tf.convert_to_tensor([3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4], dtype='float32')
    
    npi_constraints = []
    for i in range(NB_ACTION):
        npi_constraints.append(NPIConstraint(max_npis[i]))
    
    # The crucial trainable weights. An array for single NPIs and an array
    # for the set of NB_ACTION NPIs for each day.
    future_action_single_npis = []
    future_action_layers = []

    # Reuseable layers for the logic below
    npi_concat_layer = NPILayer()
    calc_layer = CalcCasesLayer()
    multiply_layer = Multiply()
    add_layer = Add()  
    concat_layer = Concatenate(axis=1)
    reshape_prediction_layer = Reshape((1,1))
    reshape_context_layer = Reshape(())
    
    # Create our trainable weight layers. Append the resulting NPILayers to
    # action_layers one at a time as we proceed through the prediction rollout.
    for i in range(num_days):
        future_action_single_npis.append([])
        for j in range(NB_ACTION):
            # Pass population as a dummy input
            future_action_single_npis[-1].append(
                SingleNPI(name=f"day_{i}_npi_{j}", constraint=npi_constraints[j])(population_input)
            )
        future_action = npi_concat_layer(future_action_single_npis[-1])
        future_action_layers.append(future_action)

    # Looking at the rollout logic, the predictor actually uses the actions
    # from the day of.
    action_layers.append(future_action_layers[0])

    # For each day we need to prescribe for:
    # a) Get the context and action inputs by concatenating the last 
    #    NB_LOOKBACK_DAYs contexts and actions. 
    # b) Predict the next context and append it to the context list
    # c) Append our next set of (prescribed) NPIs
    for i in range(num_days):
        context_grp = concat_layer(context_layers[-NB_LOOKBACK_DAYS:])
        action_grp = concat_layer(action_layers[-NB_LOOKBACK_DAYS:])
        predict = reshape_prediction_layer(predict_layer([context_grp, action_grp]))

        context_layers.append(predict)
        if i + 1 != num_days:
            action_layers.append(future_action_layers[i+1])

    # We now have all the PredictionRatios we need, and must convert them to
    # case numbers
    
    # Construct a list of WINDOW_SIZE previous new cases
    prev_new_cases_layers = []
    for i in range(WINDOW_SIZE):
        prev_new_cases = prev_new_cases_input[:, i:i+1]
        prev_new_cases_layers.append(prev_new_cases)

    # Keep track of the current number of total cases as we roll out
    current_total_cases_layers = [total_cases_input]

    
    
    # For each day:
    # a) Get the ratio we want to convert
    # b) Calculate the number of cases from this ratio
    # c) Add that number of cases to the running total and the list of previous
    #    cases as we need to update those two inputs fro the next call to
    #    calc_layer
    for i in range(num_days):
        reshaped_context = reshape_context_layer(context_layers[NB_LOOKBACK_DAYS+i])
        next_new_cases = calc_layer([reshaped_context, 
                                     prev_new_cases_layers[-WINDOW_SIZE:], 
                                     current_total_cases_layers[-1], population_input])
        next_total_cases = add_layer([current_total_cases_layers[-1], next_new_cases])

        prev_new_cases_layers.append(next_new_cases)
        current_total_cases_layers.append(next_total_cases)

    # Sum the case numbers to get the total cases in the layer: here is our output
    total_new_cases = reshape_context_layer(add_layer(prev_new_cases_layers[WINDOW_SIZE:]))

    return Model(inputs=[outer_context_input, outer_action_input, population_input, 
                         total_cases_input, prev_new_cases_input], outputs=[total_new_cases])