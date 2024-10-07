import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
        # print length of each dataframe
        print(f'Length of {file}: {len(df)}')
    
    # Combines all dataframes into one, sorted by the 'time' column
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # drop rows with missing values
    combined_data.dropna(inplace=True)

    # drop dataframe with length of 0
    combined_data = combined_data[combined_data['latitude'] != 0]

    # print the length of the combined dataframe
    print(f'Length of combined dataframe: {len(combined_data)}')

    return combined_data
    
'''
Method that preprocesses the data.

Input:
    Data: a pandas dataframe containing the data.
Output:
    Data: a new pandas dataframe containing the preprocessed data.
          dataset includes following normalized features:
            - longitude
            - latitude
            - distance
            - speed
            - sin_time
            - cos_time

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
    # print length of the dataframe
    print(f'Length of dataframe: {len(df)}')
    # drop rows with missing values
    df.dropna(inplace=True)

    # drop rows with values of 0 
    df = df[df['latitude'] != 0]

    print(f'Length of dataframe: {len(df)}')

    df = sort_and_group_data(df)
    print("Sorted and grouped data:")

    df = handle_status_column(df)
    print("Data with status column handled:")

    df = extract_features(df)
    print("Data with features extracted:")

    df = normalize_features(df)
    print("Data with features normalized:")

    # Make sure every value in the dataframe is an int, float, etc.
    # Drop any columns that are not numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

'''
Helper functions for the preprocessing steps.
'''
def sort_and_group_data(df):
    # Sort and group data by driver and day
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    df['date'] = df['time'].dt.date  # Extract the date part from the timestamp
    # If training data ('plate' column exists), group by 'plate', 'date', and 'time'
    if 'plate' in df.columns:
        df_sorted = df.sort_values(by=['plate', 'date', 'time'])
    else:
        # If test_data (no 'plate' column), group by 'date' and 'time'
        df_sorted = df.sort_values(by=['date', 'time'])
    return df_sorted

def handle_status_column(df):
    # Handling the 'status' column
    # To do so, we will create sub-trajectories based on the status column
    # This way, we can treat each status (active or not) separately.
    # This preserves the temporal order of the data while also considering
    # the impact of whether a driver is active or not
    df['trajectory_id'] = (df['status'] != df['status'].shift()).cumsum()
    return df

def extract_features(df):
    # Replace invalid latitude and longitude values with 0
    df.loc[(df['latitude'] < -90) | (df['latitude'] > 90), 'latitude'] = 0
    df.loc[(df['longitude'] < -180) | (df['longitude'] > 180), 'longitude'] = 0

    # Create a column for latitude and longitude pairs
    df['lat_lon'] = list(zip(df['latitude'], df['longitude']))

    # Define a column for distance between consecutive points
    df['distance'] = 0.0

    # Calculate the distance between consecutive rows using geographic distance
    for i in range(1, len(df)):
        df.at[i, 'distance'] = geodesic(df['lat_lon'][i-1], df['lat_lon'][i]).meters
        # Reset distance to 0 if it's the beginning of a new trajectory
        if df['trajectory_id'].iloc[i] != df['trajectory_id'].iloc[i - 1]:
            df.at[i, 'distance'] = 0.0
        if i % 10000 == 0:
            print(f'Processed {i} rows')

    # Extracting time difference feature for speed calculation
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['time_diff'].fillna(0, inplace=True)

    # Calculate speed as distance / time difference
    df['speed'] = df.apply(lambda row: row['distance'] / row['time_diff'] if row['time_diff'] > 0 else 0, axis=1)

    # Calculate total distance and average speed for each trajectory
    total_distance = df.groupby('trajectory_id')['distance'].sum().rename('total_distance')
    average_speed = df.groupby('trajectory_id')['speed'].mean().rename('average_speed')

    # Merge the aggregated features back into the original DataFrame
    df = df.merge(total_distance, on='trajectory_id', how='left')
    df = df.merge(average_speed, on='trajectory_id', how='left')

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
    for feature in ['longitude', 'latitude', 'distance', 'speed', 'total_distance', 'average_speed']:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val)

    # Transforming time transformation to sin/cos representation.
    # This is done to capture cyclical nature of time.
    # To do so, we will first convert time to seconds from midnight
    # to preserve exact time of each entry
    # Then, convert the time to sin/cos representation

    # Converting 'time' to seconds since midnight
    df['seconds_since_midnight'] = pd.to_datetime(df['time']).dt.hour * 3600 + pd.to_datetime(df['time']).dt.minute * 60 + pd.to_datetime(df['time']).dt.second

    # Normalize 'seconds_since_midnight' to [0, 1] 
    df['normalized_time'] = df['seconds_since_midnight'] / 86400  # 86400 seconds in a day

    # Add cyclical time features (sin and cos based on normalized time)
    df['sin_time'] = np.sin(2 * np.pi * df['normalized_time'])  # capture cyclical behavior
    df['cos_time'] = np.cos(2 * np.pi * df['normalized_time'])  # capture cyclical behavior

    # Drop columns that are not necessary for the model
    df.drop(columns=['time', 'date', 'lat_lon', 'time_diff', 'seconds_since_midnight', 'normalized_time'], inplace=True)

    return df


'''
Creating sequences.

For this task, we will create sequences which will be fed to the model.
After looking at the trajectory lengths for each driver, we notice that sequence length is 
heavily right-skewed -- meaning that most trajectories are short, but there are a few very long ones.
For a sample dataset from the data directory, we see the following attributes of trajectory lengths:

count    512.000000
mean      52.568359
std       69.840770
min        1.000000
25%       14.000000
50%       31.500000
75%       67.000000
max      633.000000

Here, we see an average trajectory length of 52.57, with a standard deviation of 69.84, and a maximum length of 633.

Since trajectories are of varying lengths, we will use padding to handle variable-length sequences. 
We will pad sequences to the 95h percentile of the trajectory length distribution. 
This way, we'll preserve most of the data while keeping the sequence length manageable.

Padding - adding zeros to the end of sequences to make them all the same length.

Features:
    - longitude
    - latitude
    - status
    - distance
    - speed
    - sin_time
    - cos_time
'''
def create_sequences(df):
    # Get trajectory lengths
    trajectory_lengths = df.groupby('trajectory_id').size()

    # Get 95th percentile of trajectory lengths
    percentile_95 = trajectory_lengths.quantile(0.95)

    # Set maximum sequence length
    max_seq_length = int(percentile_95)

    # Group by 'trajectory_id' and create list of sequences
    sequences = []
    trajectory_ids = df['trajectory_id'].unique()

    # Creating sequences including selected feature data
    features = ['longitude', 'latitude', 'status', 'distance', 'speed', 'total_distance', 'average_speed', 'sin_time', 'cos_time']

    for traj_id in trajectory_ids:
        traj_data = df[df['trajectory_id'] == traj_id][features].values
        traj_tensor = torch.tensor(traj_data, dtype=torch.float32)

        # Truncate the sequence if it's longer than the max_seq_length
        if traj_tensor.shape[0] > max_seq_length:
            traj_tensor = traj_tensor[:max_seq_length]
        
        sequences.append(traj_tensor)

    # Create targets tensor, which is the 'plate' column
    if 'plate' in df.columns:
        targets = df.groupby('trajectory_id')['plate'].first().values
        targets_tensor = torch.tensor(targets, dtype=torch.long)
    else:
        targets_tensor = None
    # targets = df.groupby('trajectory_id')['plate'].first().values
    # targets_tensor = torch.tensor(targets, dtype=torch.long)

    return sequences, targets_tensor

# padded_sequences, targets_tensor = create_sequences(df)

# Real data
# data_dir = 'data'
# df = load_data(data_dir)
# print('Data successfully loaded.')

# df = preprocess_data(df)
# print('Data successfully preprocessed.')
# print(df.head(10))

# # Save preprocessed data
# df.to_csv('preprocessed_data.csv', index=False)

# # Split into training and validation set, using .8/.2 split
# df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Sample data
# data_dir = 'sample_data'
# df = load_data(data_dir)
# print('Data successfully loaded.')