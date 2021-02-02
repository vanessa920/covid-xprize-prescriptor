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

class Trainer(object):
    def __init__(self, num_days):
        self.num_days = num_days
        
        self.zero_npis = tf.convert_to_tensor([[[0.0 for i in range(NB_ACTION)]] for j in range(num_days)], dtype='float32')
        self.max_npis = tf.constant(tf.convert_to_tensor([[[3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]] for j in range(num_days)], dtype='float32'))
        
        self.prescriptor = construct_model(num_days)
        
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        
        
        self.optimizer_zero_weights = [np.int64(0.0)] + [np.zeros((12,)) for i in range(2*num_days)]
        
        self.inputs = [tf.zeros((1, NB_LOOKBACK_DAYS, NB_CONTEXT)), tf.zeros((1, NB_LOOKBACK_DAYS, NB_ACTION)), tf.zeros((1,)), tf.zeros((1,)), tf.zeros((1, WINDOW_SIZE)), tf.zeros((1, NB_ACTION))]
        self.min_cases = tf.constant(0.0, dtype='float32')
        
    def get_prescriptor(self):
        return self.prescriptor

    def get_weights(self):
        return self.prescriptor.get_trainable_weights()
    
    def set_weights(self, new_weights):
        for i in range(self.num_days):
            self.prescriptor.get_layer(f"future_npis_{i}").set_weights(new_weights[i])

    def zero_weights(self):
        self.set_weights(self.zero_npis)
        
    def max_weights(self):
        self.set_weights(self.max_npis)
        
    # context, action, population, current infected, prev cases, weights
    def predict(self):
        return(self.prescriptor(self.inputs))
    
    def predict_round(self):
        return self.prescriptor(tf.math.round(self.inputs))
    
    def weights(self):
        return self.prescriptor.trainable_weights
        
    # sets inputs, calcs min cases, then sets weights to zero
    def set_inputs_and_goal(self, inputs):
        self.inputs = inputs
        self.max_weights()
        self.min_cases = tf.reshape(self.predict()[0], (1,))
        self.zero_weights()
        self.max_cases = tf.reshape(self.predict()[0], (1,))
        
        self.max_stringency = tf.reshape(tf.tensordot(inputs[-1], self.max_npis[0], axes=(1,1)), ())
        self.max_reduction = self.max_cases - self.min_cases
        
        # stringency penalty starts high - 1 increase in stringency is impossible since it's the max number of reducible cases
        # scale stringency/penalty relationship with max weighted stringency
            
    # could potentially constrain with case goal?
    #@tf.function
    
    def reset_optimizer(self):
        if len(self.optimizer.weights) != 0:
            self.optimizer.set_weights(self.optimizer_zero_weights)
            
    def change_lr(self, lr, beta_1 = 0.9, beta_2 = 0.999, decay=0.0):
        self.optimizer.lr.assign(lr)
        self.optimizer.beta_1.assign(beta_1)
        self.optimizer.beta_2.assign(beta_2)
        self.optimizer.decay.assign(decay)
    
    @tf.function#(experimental_compile=True)
    def train_loop(self, stringency_penalty):
        print("tracing...")
        with tf.GradientTape() as tape:
            new_cases, stringency = self.predict()
            
            
            cases_loss = self.loss_fn(self.min_cases, new_cases)
            
            stringency_loss = tf.math.multiply(stringency_penalty, tf.divide(stringency, self.max_stringency))
            
            total_loss = tf.math.add(cases_loss, stringency_loss)
            
            gradients = tape.gradient(total_loss, self.prescriptor.trainable_weights)
            
            self.optimizer.apply_gradients(zip(gradients, self.prescriptor.trainable_weights))
        return cases_loss, stringency, total_loss, gradients
    