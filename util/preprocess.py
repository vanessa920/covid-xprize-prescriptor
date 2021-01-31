# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

# modified preprocessing logic from Cognizant's original code
# Andrew Zhou

import os

# noinspection PyPep8Naming
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")

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


# todo: handle the case where the given start date is in the future & there's a gap between our last data
# would this impact the time needed to train? we'd have to roll out the predictions before starting
# could also toss out all the data prior to NB_LOOKBACK_DAYS 

# also get previous case data
def get_all_data(start_date, window_size=WINDOW_SIZE):
    data = {}
    
    df = prepare_dataframe(DATA_FILE_PATH)
    df = df[df.Date < start_date]
    geos = df.GeoID.unique()
    # start_date shouldn't matter here, but let's just be safe
    initial_conditions = create_country_samples(df, geos, start_date, window_size)
    
    data["df"] = df
    data["geos"] = geos
    data["initial_conditions"] = initial_conditions
    
    return data

# npi_weights is (NB_ACTION,)
def get_input_tensor(data, geo, npi_weights):
    init_cond = data['initial_conditions'][geo]
    context_0 = tf.convert_to_tensor(init_cond['context_0'], dtype='float32')[tf.newaxis, :, tf.newaxis]
    action_0 = tf.convert_to_tensor(init_cond['action_0'], dtype='float32')[tf.newaxis]
    population = tf.convert_to_tensor(init_cond['population'], dtype='float32')[tf.newaxis]
    total_cases_0 = tf.convert_to_tensor(init_cond['total_cases_0'], dtype='float32')[tf.newaxis]
    prev_new_cases_0 = tf.convert_to_tensor(init_cond['prev_new_cases'], dtype='float32')[tf.newaxis]
    return [context_0, action_0, population, total_cases_0, prev_new_cases_0, npi_weights[tf.newaxis]]

def prepare_dataframe(data_url: str) -> pd.DataFrame:
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

def _load_original_data(data_url):
    latest_df = pd.read_csv(data_url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                  latest_df["CountryName"],
                                  latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
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

def create_country_samples(df, geos, start_date, window_size) -> dict:
    """
    For each country, creates numpy arrays for Keras
    :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
    :param geos: a list of geo names
    :param start_date: the first day to prescribe for
    :return: a dictionary of train and test sets, for each specified country
    """
    context_column = 'PredictionRatio'
    action_columns = NPI_COLUMNS
    outcome_column = 'PredictionRatio'
    country_samples = {}
    
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
            country_samples[g] = {
                'context_0': X_context[-1].reshape((-1,)),
                'action_0': X_action[-1],
                'population': population,
                'total_cases_0': initial_total_cases,
                'prev_new_cases': prev_new_cases
            }
    return country_samples