
import numpy as np
from .preprocess import get_all_data
from .write_solutions import write_solutions
from .trainer import Trainer
import tensorflow as tf
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')

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

def prescribe_help(start_date, end_date, ips_path, weights_path, output_csv):
    
    num_days = int(1 + (np.datetime64(end_date)-np.datetime64(start_date))/(np.timedelta64(1, 'D')))
    
    t = Trainer(num_days)
    t.change_lr(1, decay=0.1)
    # todo - handle rolling out data if start date is in the future

    data = get_all_data(start_date, DATA_FILE_PATH, ips_path, weights_path)

    df = data['df']
    geos = data['geos']
    
    dummy_df = pd.DataFrame(columns=NPI_COLUMNS + ["Date", "CountryName", "RegionName", "PrescriptionIndex"])
    dummy_df.to_csv(output_csv, index=False)
    
    for g in geos:
        print("prescribing for", g)
        g_inputs = data['input_tensors'][g]
        country_name = df[df.GeoID == g].iloc[0]["CountryName"]
        region_name = df[df.GeoID == g].iloc[0]["RegionName"]
        
        prescriptions = prescribe_one_geo(t, g_inputs)
        
        write_solutions(prescriptions, country_name, region_name, start_date, output_csv)


def prescribe_one_geo(t, inputs):
    t.set_inputs_and_goal(inputs)

    initial_penalty = get_initial_penalty(t)
    
    prescriptions = []
    new_penalty = initial_penalty

    for i in range(151):
        if i % 15 == 0:
            new_penalty = initial_penalty - 1.5*initial_penalty/(1+np.floor(i/15))
            if i != 0:
                prescriptions.append(np.round(np.array(list(map(lambda x: list(x.numpy()), t.weights())))))
                t.reset_optimizer()
        cases_left, stringency, total_loss, gradients = t.train_loop(new_penalty)
    return prescriptions
    
def get_initial_penalty(t):
    zero_penalty = tf.constant(0.0, dtype='float32')
    _, _, total_loss, gradients = t.train_loop(zero_penalty)
    min_gradient = tf.math.reduce_min(gradients)
    initial_penalty = tf.constant(-min_gradient * t.max_stringency * t.num_days)
    t.zero_weights()
    t.reset_optimizer()
    return initial_penalty