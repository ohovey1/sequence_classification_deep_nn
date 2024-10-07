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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

from data_loader import load_data, preprocess_data, create_sequences

'''
LSTM Construction.

Considerations for model construction:
    1. Input data shape (batch_size, sequence_length, input_size)
    2. LSTM layer
    3. Fully connected layer
    4. Regularization layer

Inputs:
    - input_size: number of features in input data
    - hidden_size: number of features in hidden state
    - output_size: number of features in output
    - num_layers: number of LSTM layers
    - dropout: dropout rate for regularization

Output:
    - LSTM model
'''

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, packed_input):
        packed_output, (hn, cn) = self.lstm(packed_input)  # LSTM expects packed sequence
        #unpacked_output, lengths = pad_packed_sequence(packed_output, batch_first=True)  # Unpack 
        out = self.fc(self.dropout(hn[-1]))  # Apply dropout and fully connected layer, getting final hidden state
        return out
    
'''
GRU Construction.

Inputs:
    - input_size: number of features in input data
    - hidden_size: number of features in hidden state
    - output_size: number of features in output
    - num_layers: number of LSTM layers
    - dropout: dropout rate for regularization

Output:
    - GRU model
'''
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(GRUClassifier, self).__init__()
        # Define the GRU layer instead of LSTM
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, packed_input):
        # GRU forward pass (returns packed output and the last hidden state)
        packed_output, hn = self.gru(packed_input)
        out = self.fc(self.dropout(hn[-1]))  # Shape: [batch_size, output_size]. 
        return out

# Defining model parameters
input_size = 9  # Number of features in input data
hidden_size = 256 # Using 128 as starting point for # of hidden units
output_size = 5 # Output size for binary classification
num_layers = 3 # Number of stacked LSTM layers, using 2 for now
dropout = 0.2 # Dropout rate for regularization

# Initialize model
model = GRUClassifier(input_size, hidden_size, output_size, num_layers, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Model successfully initialized.')
print(model)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Using Adam optimizer
criterion = nn.CrossEntropyLoss(reduction='mean') # CrossEntropyLoss for multi-class classification

'''
Step 3: Training the model.

For model training, we will create DataLoader to handle batch processing.
Then, feed the model through the training loop and monitor loss
'''
# Load full preprocessed df
#df = pd.read_csv('preprocessed_data.csv')
#df = df[1:5000]

# df = load_data('data')
# df = preprocess_data(df)
# print(df.head())
# df.to_csv('preprocessed_data.csv', index=False)

# # Get traj ids and split into training and validation set
# trajectory_ids = df['trajectory_id'].unique()
# train_ids, val_ids = train_test_split(trajectory_ids, test_size=0.2, random_state=42)

# # Define training and validation dataframes
# train_df = df[df['trajectory_id'].isin(train_ids)]
# test_df = df[df['trajectory_id'].isin(val_ids)]

# # # Create sequences and target tensors for train and test data
# print('Creating sequences...')
# X_train, y_train = create_sequences(train_df)
# X_test, y_test = create_sequences(test_df)
# print('Sequences successfully created.')

'''
Data Loading.

Creating a custom sequence dataset which we can use for our DataLoader.
Utilyzes some helper functions that help with padding.
'''
# Defining some helper functions for training loop to handle padding and masking
def pad_sequences(sequences):
    # Pad sequences to match the longest sequence in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Create a tensor containing the lengths of each sequence before padding
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    return padded_sequences, lengths

def pack_for_lstm(padded_sequences, lengths):
    # Pack the padded sequences
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
    return packed_sequences


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def collate_fn(batch):
    # Separate sequences and targets from the batch
    sequences, targets = zip(*batch)
    
    # Convert sequences and targets to tensors
    sequences = [torch.tensor(seq) for seq in sequences]
    targets = torch.tensor(targets)
    
    # Step 1: Pad the sequences and get their lengths
    padded_sequences, lengths = pad_sequences(sequences)

    # Step 2: Return the padded sequences, targets, and lengths
    return padded_sequences, targets, lengths

# Create DataLoader objects for batch handling, batch size 32
batch_size = 32

# train_dataset = SequenceDataset(X_train, y_train)
# test_dataset = SequenceDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Training loop
def train(model, train_loader):
    train_losses = []
    model.train()
    # Train the model
    num_epochs = 250

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for padded_sequences, targets, lengths in train_loader:  # Loop over train_loader
            padded_sequences, targets, lengths = padded_sequences.to(device), targets.to(device), lengths.to('cpu')
            # Step 1: Pack the sequences
            packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

            # Step 2: Forward pass through the model
            optimizer.zero_grad()  # Reset gradients
            outputs = model(packed_sequences)  # Model returns outputs directly

            # Step 3: Loss calculation
            # `outputs` shape: [batch_size, num_classes]
            # `targets` shape: [batch_size]
            loss = criterion(outputs, targets)  # Compute loss 

            # Step 4: Backpropagation and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Accumulate loss
            epoch_loss += loss.item()

        # Print loss
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")
    
    return train_losses

#train_losses = train(model, train_loader)

# After training, plot the loss
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time')
# plt.legend()
# plt.show()

# Save the model
#torch.save(model.state_dict(), 'gru_model.pth')

print('Model successfully trained and saved.')

'''
Step 4: Evaluation.

Using test_df, we will preprocess the data and create sequences.
Then, we will evaluate the model using the validation set and get the accuracy of the model
'''
# Load the model
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()
model.to(device)

# Evaluate the model
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation
        for padded_sequences, targets, lengths in test_loader:
            # Move data to the same device as the model
            padded_sequences, targets = padded_sequences.to(device), targets.to(device)
            lengths = lengths.to('cpu')

            # Step 1: Pack the sequences
            packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

            # Step 2: Forward pass through the model
            outputs = model(packed_sequences)  # Outputs will have shape [batch_size, num_classes]

            # Step 3: Get predictions
            _, predicted_classes = torch.max(outputs, dim=1)  # Get the index of the max log-probability

            # Step 4: Count correct predictions
            correct_predictions += (predicted_classes == targets).sum().item()
            total_predictions += targets.size(0)

            # Store predictions and targets for further analysis if needed
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    total = 0

    for i, j in zip(all_predictions, all_targets):
        if i == j:
            total += 1
        
            
    print(f"Total correct: {total}\nTotal: {len(all_targets)}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Return predictions and targets for further analysis or reporting
    return all_predictions, all_targets

print(model)
#all_predictions, all_targets = evaluate(model, test_loader)
# Save predictions and targets to a CSV file for further analysis
# results_df = pd.DataFrame({
#     'Predictions': all_predictions,
#     'Targets': all_targets
# })

#results_df.to_csv('predictions_targets.csv', index=False)
print('Predictions and targets successfully saved to predictions_targets.csv.')#