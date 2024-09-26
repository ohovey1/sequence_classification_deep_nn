import os
import pandas as pd

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

Preprocessing steps:
    1. Sorting and Grouping data by driver and day
    2. Handling Status column
    3. Creating Sequences
    4. Extracting Features
    5. Normalizing Features
    6. Embedding trajectories
'''
def preprocess_data(data):
    #TODO: Implement
    pass