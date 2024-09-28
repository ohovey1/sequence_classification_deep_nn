'''
Define models for evaluation using PyTorch.

For sequence classification, we use a simple LSTM model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


from data_loader import load_data, preprocess_data

# data_dir = 'data'
# df = load_data(data_dir)
# df = preprocess_data(df)
# print(df.head(20))

df2 = pd.read_csv('data/2016_07_02.csv')
df2 = preprocess_data(df2)
print(df2.head(10))




