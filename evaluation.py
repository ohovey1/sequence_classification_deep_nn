import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from data_loader import load_data, preprocess_data, create_sequences
from models import LSTMClassifier, GRUClassifier, evaluate

def process_data(traj):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
            [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.
    
    """
    # Convert data to pandas df
    df = pd.DataFrame(traj, columns=['longitude', 'latitude', 'time', 'status'])

    # Preprocess data using preprocess_data function (from data_loader.py)
    df = preprocess_data(df)

    # Creating sequences:
    sequences, _ = create_sequences(df)
    print(data)

    # `sequences` is a list of tensors, take the first (and only) tensor for this single trajectory
    input_tensor = sequences[0]

    # Add a batch dimension to make it compatible with the model's expected input shape
    # The model expects input of shape: [batch_size, sequence_length, input_size]
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, seq_len, feature_size]

    return input_tensor


def run(data,model):
    """
    
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    
    """

    # Move  tensor to cuda
    input_tensor = data.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations 
    with torch.no_grad():
        # Forward pass: get predictions
        output = model(input_tensor)

        # Get the predicted class (index of the max logit/probability)
        _, predicted_class = torch.max(output, dim=1)

    # Return the predicted class as an int
    return predicted_class.item()


# Recreating  model architecture
input_size = 9  # Number of input features
hidden_size = 256  # Hidden size used during training
output_size = 5  # Number of driver classes
num_layers = 3  # Number of GRU layers
dropout = 0.2  # Dropout rate used during training

# Initialize model with the same parameters
model = GRUClassifier(input_size, hidden_size, output_size, num_layers, dropout)
#model = LSTMClassifier(input_size, hidden_size, output_size, num_layers, dropout)


# Load the saved state dictionary
model.load_state_dict(torch.load('gru_model.pth'))
#model.load_state_dict(torch.load('lstm_model.pth'))

# Set the model to evaluation mode
model.eval()

# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data = [[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
            [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]

data = process_data(data)
prediction = run(data, model)

print('------------------')
print(f"Prediciton: {prediction}")