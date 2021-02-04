# Modified the provided preprocessing code

# @author Andrew Zhou

# Original license
# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.


import os

# noinspection PyPep8Naming
import numpy as np
import pandas as pd
import tensorflow as tf
from .xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
MODEL_WEIGHTS_FILE = os.path.join(ROOT_DIR, "models", "trained_model_weights.h5")
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")
COUNTRIES_REGIONS_FILE = os.path.join(DATA_PATH, "countries_regions.csv")

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

CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']
NB_LOOKBACK_DAYS = 21
NB_TEST_DAYS = 14
WINDOW_SIZE = 7
US_PREFIX = "United States / "
NUM_TRIALS = 1
LSTM_SIZE = 32
MAX_NB_COUNTRIES = 20
NB_ACTION = 12

# also get previous case data
def get_all_data(start_date, data_file_path, ips_path, weights_file_path, window_size=WINDOW_SIZE):
    data = {}
    df = prepare_dataframe(data_file_path)
    df = df[df.Date < start_date]
    
    fill_starting = df.Date.max() + np.timedelta64(1, 'D')
    
    ips_df = _load_original_data(ips_path)
    

    required_geos = ips_df.GeoID.unique()
    
    df=df[df.GeoID.isin(required_geos)]
    
    
    # If the start date is in the future, use our historical IP file
    # as a base and project until the start date. Note that as specified
    # by the competition base IP file (OxCGRT_latest.csv) is frozen on
    # one of the 2021 January 11 releases. The exact version is in the 
    # competition repo.
    
    # Note that XPrizePredictor uses some redundant code which is replicated
    # in this file. May fix this issue in the future.
    xprize_predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, df)
    
    
    fill_ending = pd.to_datetime(start_date) - np.timedelta64(1, 'D')
    
    fill_df = xprize_predictor.predict(fill_starting, fill_ending, ips_df)
    add_geoid(fill_df)
    fill_df = do_calculations(fill_df, df)
    
    fill_df = fill_df.merge(ips_df, how='left', on=['GeoID', 'Date'], suffixes=['', '_r'])
    
    df = pd.concat([df, fill_df])
    
    df = df.sort_values(by=['Date'])
    npi_weights = prepare_weights_dict(weights_file_path, required_geos)
    
    initial_conditions, country_names, geos_and_infected = create_country_samples(df, required_geos, start_date, window_size)
    
    data["geos"] = df.GeoID.unique()
    data["input_tensors"] = initial_conditions
    data["npi_weights"] = npi_weights
    data["country_names"] = country_names
    
    return data

def add_geoid(df):
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                                  df["CountryName"],
                                  df["CountryName"] + ' / ' + df["RegionName"])

def prepare_weights_dict(weights_file_path, required_geos):
    
    npi_weights = {}
    
    weights_df = pd.read_csv(weights_file_path,
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    weights_df["GeoID"] = np.where(weights_df["RegionName"].isnull(),
                                  weights_df["CountryName"],
                                  weights_df["CountryName"] + ' / ' + weights_df["RegionName"])
    
    for g in required_geos:
        geo_weights = weights_df[weights_df.GeoID == g]

        if len(geo_weights) != 0:
            weights = geo_weights.iloc[0][NPI_COLUMNS].to_numpy()
            weights[weights==0] += 0.001 # so we don't divide by zero later  
        else:
            weights = np.array([1.0 for i in range(NB_ACTION)])
            
        weight_tensor = tf.convert_to_tensor(weights, dtype='float32')[tf.newaxis]
        npi_weights[g] = weight_tensor
            
    return npi_weights

def get_input_tensor(init_cond):
    context_0 = tf.convert_to_tensor(init_cond['context_0'], dtype='float32')[tf.newaxis, :, tf.newaxis]
    action_0 = tf.convert_to_tensor(init_cond['action_0'], dtype='float32')[tf.newaxis]
    population = tf.convert_to_tensor(init_cond['population'], dtype='float32')[tf.newaxis]
    total_cases_0 = tf.convert_to_tensor(init_cond['total_cases_0'], dtype='float32')[tf.newaxis]
    prev_new_cases_0 = tf.convert_to_tensor(init_cond['prev_new_cases'], dtype='float32')[tf.newaxis]
    
    return [context_0, action_0, population, total_cases_0, prev_new_cases_0]

def prepare_dataframe(data_url) -> pd.DataFrame:
    """
    Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
    loads the Johns Hopkins dataset and merges that in.
    :param data_url: the url containing the original data
    :return: a Pandas DataFrame with the historical data
    """
    # Original df from Oxford
    df1 = _load_original_data(data_url)

    # Additional context df (e.g Population for each country)
    df2 = _load_additional_context_df()
    
    # Merge the 2 DataFrames
    df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

    # Drop countries with no population data
    df.dropna(subset=['Population'], inplace=True)

    #  Keep only needed columns
    columns = CONTEXT_COLUMNS + NPI_COLUMNS
    df = df[columns]

    # Fill in missing values
    _fill_missing_values(df)

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

    # Compute smoothed versions of new cases and deaths each day
    df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
        WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
        WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

    # Compute percent change in new cases and deaths each day
    df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    
    df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1

    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])
    
    
    return df

# Calculate ConfirmedCases and ProportionInfected columns for the dates we
# need to fill in before the start date
def do_calculations(fill_df, hist_df):
    additional_context = _load_additional_context_df()
    new_fill_df = fill_df.merge(additional_context, on=['GeoID'], how='left', suffixes=('', '_y'))
    # Drop countries with no population data
    new_fill_df.dropna(subset=['Population'], inplace=True)
    new_fill_df = new_fill_df.rename(columns={"PredictedDailyNewCases": "NewCases"})
    new_fill_df['NewCases'] = new_fill_df['NewCases'].clip(lower=0)
    required_geos = new_fill_df.GeoID.unique()
  
    new_fill_df["ConfirmedSinceLastKnown"] = new_fill_df.groupby("GeoID")["NewCases"].cumsum()
    new_fill_df["LastKnown"] = 0

    for g in required_geos:
        previous_data = hist_df[hist_df.GeoID == g]
        # cases for the last day on record
        last_confirmed_cases = previous_data["ConfirmedCases"].iloc[-1]
        new_fill_df.loc[new_fill_df.GeoID == g, ["LastKnown"]] = last_confirmed_cases

    new_fill_df["ConfirmedCases"] = new_fill_df["ConfirmedSinceLastKnown"] + new_fill_df["LastKnown"]
    
    # Add column for proportion of population infected
    new_fill_df['ProportionInfected'] = new_fill_df['ConfirmedCases'] / new_fill_df['Population']
    
    return new_fill_df

    
def _load_original_data(data_url):
    latest_df = pd.read_csv(data_url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)

    add_geoid(latest_df)
    
    return latest_df

def _fill_missing_values(df):
    """
    # Fill missing values by interpolation, ffill, and filling NaNs
    :param df: Dataframe to be filled
    """
    df.update(df.groupby('GeoID').ConfirmedCases.apply(
        lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of cases is available
    df.dropna(subset=['ConfirmedCases'], inplace=True)
    df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
        lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of deaths is available
    df.dropna(subset=['ConfirmedDeaths'], inplace=True)
    for npi_column in NPI_COLUMNS:
        df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

def _load_additional_context_df():
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                        usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName']

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                          usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_uk_df)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_brazil_df)

    return additional_context_df

# Modified original logic. Now returns the input tensor for each GeoID,
# a GeoID to country/region name dict just for convenience, and a dict
# with the infection count for each GeoID.
def create_country_samples(df, geos, start_date, window_size) -> dict:
    context_column = 'PredictionRatio'
    action_columns = NPI_COLUMNS
    outcome_column = 'PredictionRatio'
    country_samples = {}
    country_names = {}
    
    geos_and_infected = {}
    df = df[df.Date < start_date]
    for g in geos:
        cdf = df[df.GeoID == g]
        cdf = cdf[cdf.ConfirmedCases.notnull()]
            
        context_data = np.array(cdf[context_column])
        action_data = np.array(cdf[action_columns])
      
        population = cdf['Population'].iloc[-1]
    
        context_samples = []
        action_samples = []
        
        initial_total_cases = np.array(cdf['ConfirmedCases'])[-1]
        prev_new_cases = np.array(cdf['NewCases'])[-window_size:]

        nb_total_days = context_data.shape[0] + 1
        for d in range(NB_LOOKBACK_DAYS, nb_total_days):
            context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
            action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
        if len(context_samples) > 0:
            X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
            X_action = np.stack(action_samples, axis=0)
            
            initial_conditions = {
                'context_0': X_context[-1].reshape((-1,)),
                'action_0': X_action[-1],
                'population': population,
                'total_cases_0': initial_total_cases,
                'prev_new_cases': prev_new_cases
            }
            input_tensor = get_input_tensor(initial_conditions)
            geos_and_infected[g] = initial_total_cases
            country_samples[g] = input_tensor
            country_names[g] = [cdf['CountryName'].iloc[0], cdf['RegionName'].iloc[0]]
    return country_samples, country_names, geos_and_infected

# Function for performing roll outs into the future
    def _roll_out_predictions(predictor, initial_context_input, initial_action_input, future_action_sequence):
        nb_roll_out_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_roll_out_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        for d in range(nb_roll_out_days):
            action_input[:, :-1] = action_input[:, 1:]
            # Use the passed actions
            action_sequence = future_action_sequence[d]
            action_input[:, -1] = action_sequence
            pred = predictor.predict([context_input, action_input])
            pred_output[d] = pred
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred
        return pred_output

    # Functions for converting predictions back to number of cases

    def _convert_ratio_to_new_cases(ratio,
                                    window_size,
                                    prev_new_cases_list,
                                    prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]

    def _convert_ratios_to_total_cases(self,
                                       ratios,
                                       window_size,
                                       prev_new_cases,
                                       initial_total_cases,
                                       pop_size):
        new_new_cases = []
        prev_new_cases_list = list(prev_new_cases)
        curr_total_cases = initial_total_cases
        for ratio in ratios:
            new_cases = self._convert_ratio_to_new_cases(ratio,
                                                         window_size,
                                                         prev_new_cases_list,
                                                         curr_total_cases / pop_size)
            # new_cases can't be negative!
            new_cases = max(0, new_cases)
            # Which means total cases can't go down
            curr_total_cases += new_cases
            # Update prev_new_cases_list for next iteration of the loop
            prev_new_cases_list.append(new_cases)
            new_new_cases.append(new_cases)
        return new_new_cases