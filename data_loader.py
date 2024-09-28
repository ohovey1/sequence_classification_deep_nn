import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic


'''
Method that reads the data from the data directory into a pandas dataframe.

Input:
    Data_dir: a string, the path to the data directory.
Output:
    Data: a pandas dataframe containing the data.
'''
def load_data(data_dir):
    # Gets all csv files from data directory
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []
    
    # Reads all csv files into a pandas dataframe
    for file in all_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    # Combines all dataframes into one, sorted by the 'time' column
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data
    
'''
Method that preprocesses the data.

Input:
    Data: a pandas dataframe containing the data.
Output:
    Data: a new pandas dataframe containing the preprocessed data.

The output will be ready to be converted into PyTorch tensors
and fed to our model.

Preprocessing steps:
    1. Sorting and Grouping data by driver and day
    2. Handling Status column
    3. Extracting Features
    4. Normalizing Features

This function uses helper functions for each step, which are defined below.
'''
def preprocess_data(df):
    df = sort_and_group_data(df)
    df = handle_status_column(df)
    df = extract_features(df)
    df = normalize_features(df)
    return df

'''
Helper functions for the preprocessing steps.
'''
def sort_and_group_data(df):
    # Sort and group data by driver and day
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    df['date'] = df['time'].dt.date  # Extract the date part from the timestamp
    df_sorted = df.sort_values(by=['plate', 'date', 'time'])
    return df_sorted

def handle_status_column(df):
    # Handling the 'status' column
    # To do so, we will create sub-trajectories based on the status column
    # This way, we can treat each status (active or not) separately.
    # This preserves the temporal order of the data while also considering
    # the impact of whether a driver is active or not
    df['trajectory_id'] = (df['status'] != df['status'].shift()).cumsum()
    
    # Grouping by plate, date, and trajectory_id to create sub-trajectories
    #df_grouped = df.groupby(['plate', 'date', 'trajectory_id'])
    return df

def extract_features(df):
    # Extracting distance feature from latitude and longitude
    # We will calculate the distance between each consecutive pair of points
    # for each driver and day

    # replace invalid latitudes and longitudes with NaN
    df.loc[(df['latitude'] < -90) | (df['latitude'] > 90), 'latitude'] = np.nan
    df.loc[(df['longitude'] < -180) | (df['longitude'] > 180), 'longitude'] = np.nan

    # Create a column for latitude and longitude pairs
    df['lat_lon'] = list(zip(df['latitude'], df['longitude']))

    # Define a lambda function to calculate the distance between consecutive rows
    df['distance'] = df.apply(
        lambda row: geodesic(df['lat_lon'].shift(1)[row.name], row['lat_lon']).meters 
        if row.name > 0 else 0, 
        axis=1
    )

    # Set the distance to 0 where a new trajectory begins (when trajectory_id changes)
    df.loc[df['trajectory_id'] != df['trajectory_id'].shift(1), 'distance'] = 0

    # Extracting speed feature.
    # Getting speed by taking the distance divided by the time difference
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['time_diff'].fillna(0, inplace=True)

    # Calculate speed as distance / time difference
    # We handle cases where time_diff is 0 by setting speed to 0 (to avoid division by zero)
    df['speed'] = df.apply(lambda row: row['distance'] / row['time_diff'] if row['time_diff'] > 0 else 0, axis=1)

    return df

def normalize_features(df):
    # This function will normalize the features
    # and remove features that are not necessary for the model
    # We will normalize the following features:
    # - longitude
    # - latitude
    # - distance
    # - speed
    # - time (convert to sin/cos cyclic representation)

    # Normalize columns manually using pandas
    for feature in ['longitude', 'latitude', 'distance', 'speed']:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val)

    # Transforming time transformation to sin/cos representation, 
    # To do so, we will first convert time to seconds from midnight,
    # to preserve exact time of each entry
    # Then, convert the time to sin/cos representation
    # Convert 'time' to seconds since midnight
    df['seconds_since_midnight'] = pd.to_datetime(df['time']).dt.hour * 3600 + pd.to_datetime(df['time']).dt.minute * 60 + pd.to_datetime(df['time']).dt.second

    # Normalize 'seconds_since_midnight' to [0, 1] 
    df['normalized_time'] = df['seconds_since_midnight'] / 86400  # 86400 seconds in a day

    # Add cyclical time features (sin and cos based on normalized time)
    df['sin_time'] = np.sin(2 * np.pi * df['normalized_time'])  # capture cyclical behavior
    df['cos_time'] = np.cos(2 * np.pi * df['normalized_time'])  # capture cyclical behavior

    # Drop columns that are not necessary for the model
    df.drop(columns=['time', 'date', 'trajectory_id', 'lat_lon', 'time_diff', 'seconds_since_midnight', 'normalized_time'], inplace=True)

    return df
