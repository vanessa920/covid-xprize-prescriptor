# Trainer class to (re)train the neural network for each successive query
#
# Implements a custom gradient descent technique adapted to this specific
# problem: we want to generate a set of prescriptions (10 max) where each
# optimizes for a different caseload-stringency tradeoff, while also training
# the model and transitioning between different queries in a reasonable amount
# of time.

# @author Andrew Zhou

import numpy as np
import tensorflow as tf
import keras.backend as K
import numpy as np
import keras

from .prescriptor import construct_model


# normalize the weights?
BATCH_SIZE = 1
NB_LOOKBACK_DAYS = 21
NB_CONTEXT = 1
NB_ACTION = 12
WINDOW_SIZE = 7

# Our Trainer class initializes and stores a prescriptor and contains functions
# to run it through the gradient update loops. We are careful in both the design
# of the prescriptor Model and our loop logic to ensure that these functions
# can be decorated with @tf.function so they can run efficiently after the
# first compilation.
class Trainer(object):
    def __init__(self, num_days):
        self.num_days = num_days
        
        # After each query, we want to find the minimum possible caseload 
        # (with weights set to max_npis) and maximum (weights set to zero_npis).
        # We also want to reset to zero so we can start training from scratch.
        self.zero_npis = tf.convert_to_tensor([[[0.0] for i in range(NB_ACTION)] 
                                               for j in range(num_days)], dtype='float32')
        self.max_npis = tf.convert_to_tensor([[[3], [3], [2], [4], [2], [3], 
                                               [2], [4], [2], [3], [2], [4]] 
                                              for j in range(num_days)], dtype='float32')
        
        self.prescriptor = construct_model(num_days)
        
        # MAE as loss function
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        # Jury rig an SGD optimizer for our gradient descent
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
        
        
        # Initialize dummy inputs and minimum cases
        self.inputs = [tf.zeros((1, NB_LOOKBACK_DAYS, NB_CONTEXT)), 
                       tf.zeros((1, NB_LOOKBACK_DAYS, NB_ACTION)), 
                       tf.ones((1,)), tf.zeros((1,)), 
                       tf.zeros((1, WINDOW_SIZE)), 
                       tf.zeros((1, NB_ACTION))]
        
        self.min_cases = tf.constant([0.0], dtype='float32')
        
    def get_prescriptor(self):
        return self.prescriptor

    def get_weights(self):
        return self.prescriptor.get_trainable_weights()
    
    # Functions to manually alter the weights. Note that doing so does not require
    # a reload of the graph.
    def set_weights(self, new_weights):
        for i in range(self.num_days):
            for j in range(NB_ACTION):
                self.prescriptor.get_layer(f"day_{i}_npi_{j}").set_weights(new_weights[i][j])

    def zero_weights(self):
        self.set_weights(self.zero_npis)
        
    def max_weights(self):
        self.set_weights(self.max_npis)
        
    def weights(self):
        return self.prescriptor.trainable_weights
        
    # Inputs: context, action, population, current infected, prev cases
    def predict(self):
        return(self.prescriptor(self.inputs))
    
        
    # Prepare the trainer for a new query. Set the input, calculate the
    # minimum and maximum cases, and set the stringency costs. Make sure
    # the weights are reset to zero.
    def set_inputs_weights_and_goal(self, inputs, npi_weights):
        self.inputs = inputs
        self.max_weights()
        self.min_cases = self.prescriptor(self.inputs)
        self.zero_weights()
        self.max_cases = self.prescriptor(self.inputs)
        self.npi_weights = npi_weights
        
        self.max_reduction = tf.math.maximum(self.max_cases - self.min_cases, 
                                             tf.constant(1.0, dtype='float32'))

    # Get the gradient of the loss, defined as the MAE between the minimum
    # number of cases (when NPIs are maxed out) and the cases produced by
    # the current set of NPI weights.
    @tf.function#(experimental_compile=True)
    def get_gradient(self, inputs, goal):
        #print("tracing gradient")
        with tf.GradientTape() as tape:
            new_cases = self.prescriptor(inputs)
            cases_loss = self.loss_fn(goal, new_cases)
            
            gradients = tape.gradient(cases_loss, self.prescriptor.trainable_weights)
        
        return gradients, cases_loss
    
    # Here's where the magic happens. Given the gradient of the loss with
    # respect to each weight, the number of updates we want to conduct
    # in a given iteration, and the cost of incrementing each NPI, we 
    # weight the gradients by those costs and select the num_updates best
    # directions whose NPIs are not yet maxed. We take those NPIs and we 
    # increment them by one. 
    #
    # This technique guarantees a consistent progression through the spectrum
    # of NPI stringency (unlike regular floating point gradient descent, whose
    # speed is harder to predict), which is crucial to dynamically selecting 
    # multiple sets of predictions with different tradeoffs. The floating point
    # approach might prove superior with query-specific tuning but in a situation
    # with time constraints and dynamic queries, the incrementation approach
    # is much more reliable. Note that decreasing num_updates will improve the 
    # results due to more granular updates.
    @tf.function
    def get_update_tensor(self, g, npi_weights, num_updates):
        #print("tracing update")
        reshaped_g = tf.reshape(g, (self.num_days,NB_ACTION))
        # Weight gradient by NPI costs
        new_g = tf.math.multiply(tf.constant(-1.0, 'float32'), tf.math.divide(reshaped_g, npi_weights))
       
        # Find the directions that are not yet maxed
        neq = tf.math.not_equal(self.prescriptor.trainable_weights, tf.reshape(self.max_npis, (self.num_days*NB_ACTION,)))
        neq_float = tf.reshape(tf.cast(neq, 'float32'), (self.num_days, NB_ACTION))

        # Multiply the maxed directions by zero so we don't select them.
        # We may start selecting them in the very end but their NPIs aren't 
        # incremented due to our NPIConstraints
        remove_capped = tf.reshape(new_g*neq_float, (self.num_days*NB_ACTION,))

        # Get the best k directions
        top_k = tf.math.top_k(remove_capped, num_updates)
        top_k_idx = tf.reshape(top_k.indices, (num_updates, 1))
        
        updates = tf.repeat(-1.0, num_updates)

        # Return a tensor with negative ones in the best directions and
        # zeros elsewhere. Applying this gradient using SGD with a learning
        # rate of 1.0 will increment those NPIs by one.
        z = tf.zeros((self.num_days*NB_ACTION,))
        update = tf.tensor_scatter_nd_update(z, top_k_idx, updates)
        unstacked = tf.unstack(update, axis=0)
        return unstacked
    
    # Decorated function to simply apply the gradients
    @tf.function
    def apply_grads(self, u):
        self.optimizer.apply_gradients(zip(u, self.prescriptor.trainable_weights))
        return
    
    # Note: in the sandbox there was enough memory to decorate the three
    # called functions in train_loop but not enough to decorate train_loop
    # itself
    @tf.function 
    def train_loop(self, inputs, goal, npi_weights, num_updates):
        #print("tracing loop")
        g, c = self.get_gradient(inputs, goal)
        u = self.get_update_tensor(g, npi_weights, num_updates)
        self.apply_grads(u)
        return c, u