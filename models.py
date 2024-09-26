'''
Define models for evaluation using PyTorch.

For sequence classification, we use a simple LSTM model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from data_loader import * 