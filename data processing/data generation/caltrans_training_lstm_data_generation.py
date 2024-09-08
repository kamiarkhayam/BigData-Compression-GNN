import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os


def load_data(file_path):
    """
    Loads data from a CSV file, handles missing values in the 'Station Length' column, and checks for remaining NaNs.

    Args:
        file_path (str): Path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The processed DataFrame with NaN values handled in the 'Station Length' column.
    """
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Check for NaN values in the 'Station Length' column and replace them with the mean
    if data['Station Length'].isnull().any():
        mean_station_length = data['Station Length'].mean()
        data['Station Length'].fillna(mean_station_length, inplace=True)
        print(f"NaN values in 'Station Length' have been replaced with the mean value: {mean_station_length}")
    
    # After replacing 'Station Length' NaNs, check for any remaining NaNs in the entire DataFrame
    if data.isnull().any().any():
        total_nans = data.isnull().sum().sum()
        print(f"Warning: After replacement, the data still contains {total_nans} NaN values.")
        # Print the count of NaN values in each column for further investigation
        print(data.isnull().sum())
    else:
        print("No NaN values found in the DataFrame after processing.")
    
    return data


def generate_sequences(data, sequence_length):
    """
    Generates input sequences and corresponding targets from time-series data.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data, where each column represents a timestamp.
        sequence_length (int): Number of time steps in each input sequence.

    Returns:
        tuple: 
            - inputs (np.array): Input sequences shaped as (num_sequences, sequence_length, num_nodes).
            - targets (np.array): Corresponding target values shaped as (num_sequences, num_nodes).
    """
    # Extract columns with time-series data (assuming date format in headers with a colon, e.g., 'HH:MM' or 'YYYY-MM-DD')
    consumption_columns = [col for col in data.columns if ':' in col]
    
    # Initialize lists to store input sequences and target values
    inputs = []
    targets = []
    
    # Loop to generate input sequences and corresponding target values
    for i in range(data[consumption_columns].shape[1] - sequence_length):
        # Extract input sequence of length 'sequence_length'
        input_seq = data[consumption_columns[i:i + sequence_length]].values  # Shape: [sequence_length, num_nodes]
        
        # Extract the target, which is the next time step following the input sequence
        target_seq = data[consumption_columns[i + sequence_length]].values  # Shape: [num_nodes]
        
        # Append input sequence and target to their respective lists
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Convert lists to numpy arrays for easier handling in machine learning models
    return np.array(inputs), np.array(targets)


def create_lstm_data(inputs, targets):
    """
    Converts input sequences and corresponding targets into PyTorch Geometric Data objects.

    Args:
        inputs (np.array): Input sequences (shape: [num_sequences, sequence_length, num_features]).
        targets (np.array): Corresponding target values (shape: [num_sequences, num_features]).

    Returns:
        list: A list of PyTorch Geometric Data objects, where each object contains input features (x) and target features (y).
    """
    all_data = []
    
    # Loop through each input sequence and corresponding target
    for input_seq, target in zip(inputs, targets):
        # Convert input and target sequences to PyTorch tensors
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Create a Data object with input features and target, and add it to the list
        all_data.append(Data(x=input_features, y=target_features))
    
    return all_data


def save_lstm_data(data, sequence_length, data_name, split, folder_path="generated_data"):
    """
    Saves LSTM data (PyTorch Geometric Data objects) to disk in the specified folder.

    Args:
        data (list): List of PyTorch Geometric Data objects.
        sequence_length (int): Length of the input sequences.
        data_name (str): Name of the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        folder_path (str, optional): Directory where the data will be saved. Defaults to "generated_data".
    """
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the file name using the sequence length and dataset split
    file_name = f"{data_name}_{split}_seq{sequence_length}.pt"
    file_path = os.path.join(folder_path, file_name)
    
    # Save the data to disk using PyTorch's torch.save function
    torch.save(data, file_path)
    
    # Print a message indicating successful save
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input and target data into training, validation, and test sets.

    Args:
        inputs (np.array): Input sequences (shape: [num_sequences, sequence_length, num_features]).
        targets (np.array): Target values corresponding to the inputs.
        train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of data to use for validation. Defaults to 0.15.

    Returns:
        tuple: Splits of the input and target data for training, validation, and testing.
    """
    # Determine the total number of sequences
    total_sequences = inputs.shape[0]
    
    # Calculate the indices for splitting the data
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # Split the data into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]
    
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)

# Define file paths and parameters
file_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cleaned_speed_data_full.csv'
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans Traffic Data\\lstm data'
sequence_length = 4
data_name = 'caltrans_speed_norm_data'

# Process and save the data
data = load_data(file_path)
inputs, targets = generate_sequences(data, sequence_length)

# Assume inputs and targets have been generated using generate_sequences
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_data(inputs, targets)

# Convert to PyTorch Geometric Data objects
train_data = create_lstm_data(train_inputs, train_targets)
val_data = create_lstm_data(val_inputs, val_targets)
test_data = create_lstm_data(test_inputs, test_targets)

# Save the datasets
save_lstm_data(train_data, sequence_length, data_name, 'train', output_folder_path)
save_lstm_data(val_data, sequence_length, data_name, 'val', output_folder_path)
save_lstm_data(test_data, sequence_length, data_name, 'test', output_folder_path)