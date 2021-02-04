# Custom prescriptor logic, to be invoked by prescribe.py

# @author Andrew Zhou

import numpy as np
from .preprocess import get_all_data
from .write_solutions import write_solutions
from .trainer import Trainer
import tensorflow as tf
import pandas as pd
import os
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')

NB_ACTION = 12
MAX_NPI_SUM = 34
NUM_PRESCRIPTIONS = 10
BASE_ITERATIONS = 200

NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

# Our prescription function.
#
# Our sandbox version had extra timing functionality that sped up predictions
# if we were running out of time which is omitted here.
def prescribe_help(start_date, end_date, ips_path, weights_path, output_csv):
    
    start = time.time()
    
    num_days = int(1 + (np.datetime64(end_date)-np.datetime64(start_date))/(np.timedelta64(1, 'D')))
    
    t = Trainer(num_days)

    data = get_all_data(start_date, DATA_FILE_PATH, ips_path, weights_path)
    
    geos = data['geos']
    
    dummy_df = pd.DataFrame(columns=NPI_COLUMNS + ["Date", "CountryName", "RegionName", "PrescriptionIndex"])
    dummy_df.to_csv(output_csv, index=False)
    
    num_iterations = BASE_ITERATIONS
    
    # Prescribe for each geo in turn
    for i in range(len(geos)):
        g = geos[i]
        print("Prescribing for", g)
        
        g_inputs = data['input_tensors'][g]
        npi_weights = data['npi_weights'][g]
        country_name, region_name = data['country_names'][g]
        
        seconds_elapsed = time.time() - start
        #print(seconds_elapsed/60.0, "minutes elapsed")
        
        prescriptions = prescribe_one_geo(t, g_inputs, npi_weights, num_iterations)
        write_solutions(prescriptions, country_name, region_name, start_date, num_days, output_csv)    
    return

# Perform a modified gradient descent on the NPI weights. More iterations 
# means more granular descent and should yield better results. 
#
# Begin with zero weights and increase selected weights with each iteration,
# choosing ten sets of weights from among all those snapshotted.
def prescribe_one_geo(t, inputs, npi_weights, num_iterations):
    t.set_inputs_weights_and_goal(inputs, npi_weights)
    
    # The total number of 1-step increments in weights we can take before
    # we reach full NPIs.
    total_updates = t.num_days * MAX_NPI_SUM
    
    # We prescribe 10 solutions, selecting the ones that first breach the
    # following barriers. These barriers indicate the percentage of reducible
    # cases that the prescriptor has failed to reduce, where total reducible 
    # cases is measured by the difference between the projected cases for
    # zero NPIs and the projected cases for max NPIs. (This logic relies
    # on the assumption that the predictor yields fewer cases for more 
    # stringent NPIs, which is reasonable to assume.)
    thresholds = [0.87, 0.72, 0.60, 0.5, 0.4, 0.33, 0.25, 0.17, 0.10, 0.07]#np.linspace(0.87, 0.07, 10)
    current_threshold_idx = 0
    
    weights_list = []
    
    # We want to go through the (almost) full spectrum of NPIs so we do 
    # update_stride increments at a time. We want to go all the way through
    # because the NPIs with the most case reductions could be weighted so
    # harshly that they won't be added until the end.
    
    # There are potential issues here given our current strategy of incrementing
    # NPIs one at a time. Potentially can be improved.
    update_stride = tf.cast(tf.math.floor(tf.math.divide(total_updates, num_iterations)), 'int32')
    update_stride = tf.math.maximum(update_stride, 1)
    
    #print("Iterations:", num_iterations)
    for i in range(num_iterations):
        # c is the number of cases over the targeted period with our current
        # set of weights. train_loop also updates the NPIs based on the
        # calculated gradients
        cases, _ = t.train_loop(t.inputs, t.min_cases, t.npi_weights, update_stride)
        p = tf.math.divide(cases, t.max_reduction)
        
        # If we've breached a threshold and not yet added a corresponding 
        # prescription, add the current one
        if p < thresholds[current_threshold_idx]:
            #print(f"found solution at {tf.constant(i)} with performance {p.numpy()[0]}")
            weights_list.append(tf.constant(tf.reshape(t.prescriptor.trainable_weights, (-1, NB_ACTION))))
            current_threshold_idx += 1
            
        # Stop early if we've found the desired number of prescriptions
        if current_threshold_idx == NUM_PRESCRIPTIONS:
            break
    
    # Some queries have anomalous data and cases end up being zero for every
    # day regardless of NPIs. Prescribe nothing here. Cameroon would be
    # an example here as some of its PredictionRatios at the end are 0, which
    # disturbs the prediction process.
    if t.min_cases == t.max_cases:
        #print("0 cases fixable, returning all zeros")
        # Wipe the weights list so the next part will fill with all zeros
        weights_list = [] 
        
    # If we failed to find the desired number of prescriptions, just pad our
    # list with empty prescriptions 
    if len(weights_list) < NUM_PRESCRIPTIONS:
        #print("fewer than 10 solutions found")
        for i in range(NUM_PRESCRIPTIONS-len(weights_list)):
            weights_list.append(tf.zeros((t.num_days, NB_ACTION)))
            
    return weights_list