import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os

def load_data(folder_path, state_abbreviations):
    """
    Loads and concatenates data from multiple CSV files based on state abbreviations.

    Args:
        folder_path (str): The path to the folder containing the data files.
        state_abbreviations (list): A list of state abbreviations representing the files to load.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing data from all specified states.

    Note:
        This function assumes each state's data is stored in a file named '<state>_data.csv'.
        It checks for and warns about the presence of NaN values in the data files.
    """
    df_list = []  # Initialize a list to hold dataframes loaded from each file

    for state in state_abbreviations:
        file_path = os.path.join(folder_path, f"{state}_data.csv")  # Construct full file path
        
        if os.path.exists(file_path):  # Check if the file exists
            data = pd.read_csv(file_path)  # Load the data from CSV
            
            # Check for NaN values in the DataFrame
            if data.isnull().values.any():
                print(f"Warning: The data in {file_path} contains NaN values.")
                print(data.isnull().sum())  # Print the count of NaNs in each column
            else:
                print(f"No NaN values found in the data from {file_path}.")
            
            df_list.append(data)  # Append the DataFrame to the list
        else:
            print(f"No data file found for {state.upper()}.")

    return pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames into a single DataFrame and return


def normalize_outputs(data, sequence_length):
    """
    Normalizes the time-series data by 'CapacityMW' and generates input sequences and corresponding targets for LSTM.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data with columns prefixed with 'Minute_' and 'CapacityMW'.
        sequence_length (int): Number of time steps to include in each input sequence.

    Returns:
        tuple: A tuple containing:
            - inputs (np.array): Input sequences shaped as (num_sequences, sequence_length, num_nodes).
            - targets (np.array): Target sequences shaped as (num_sequences, num_nodes).
    """
    
    # Identify columns that represent the time-series data ('Minute_' prefixed columns)
    output_columns = [col for col in data.columns if col.startswith('Minute_')]
    
    # Clean and convert the 'CapacityMW' column, removing 'MW' and converting to float
    data['CapacityMW'] = data['CapacityMW'].str.replace('MW', '').astype(float)
    
    # Normalize the time-series columns by the 'CapacityMW' column for each node
    normalized_data = data[output_columns].div(data['CapacityMW'], axis=0)
    
    # Initialize lists to store input sequences and target values
    inputs = []
    targets = []
    
    # Loop through the time steps to generate input sequences and targets
    for i in range(normalized_data[output_columns].shape[1] - sequence_length):
        # Extract a sequence of 'sequence_length' time steps as inputs
        input_seq = normalized_data[output_columns[i:i + sequence_length]].values  # Shape: [sequence_length, num_nodes]
        
        # The target is the next time step immediately following the input sequence
        target_seq = normalized_data[output_columns[i + sequence_length]].values  # Shape: [num_nodes]
        
        # Append the input sequence and target to the respective lists
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Convert the lists to numpy arrays for easier handling in machine learning models
    return np.array(inputs), np.array(targets)


def create_lstm_data(inputs, targets):
    """
    Creates a list of PyTorch Geometric Data objects for LSTM training.

    Args:
        inputs (np.array): Input sequences (shape: [num_sequences, sequence_length, num_nodes]).
        targets (np.array): Target sequences corresponding to the inputs (shape: [num_sequences, num_nodes]).

    Returns:
        list: A list of Data objects, each containing the input features (x) and target values (y).
    """
    all_data = []
    
    for input_seq, target in zip(inputs, targets):
        # Convert input and target sequences into PyTorch tensors
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Create a Data object and append to the list
        all_data.append(Data(x=input_features, y=target_features))
    
    return all_data


def save_lstm_data(data, sequence_length, data_name, split, folder_path="generated_data"):
    """
    Saves LSTM data to disk in a specified folder.

    Args:
        data (list): A list of Data objects containing inputs and targets.
        sequence_length (int): The length of the input sequences.
        data_name (str): A name to identify the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        folder_path (str, optional): The directory where the data will be saved. Defaults to "generated_data".
    """
    # Ensure the output directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the filename with relevant details (data_name, split, sequence length)
    file_name = f"{data_name}_{split}_seq{sequence_length}_c.pt"
    file_path = os.path.join(folder_path, file_name)
    
    # Save the data using PyTorch's save functionality
    torch.save(data, file_path)
    
    # Print a confirmation message
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the inputs and targets into training, validation, and test sets based on the provided ratios.

    Args:
        inputs (np.array): Input sequences (shape: [num_sequences, sequence_length, num_nodes]).
        targets (np.array): Corresponding target sequences (shape: [num_sequences, num_nodes]).
        train_ratio (float, optional): Proportion of the data to use for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of the data to use for validation. Defaults to 0.15.

    Returns:
        tuple: Three tuples containing the training, validation, and test sets in the format:
            (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)
    """
    total_sequences = inputs.shape[0]  # Total number of sequences
    
    # Calculate the indices for training and validation splits
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # Split the data into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]
    
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


# Configuration
folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data'
state_abbreviations = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 
    'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 
    'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
]
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data'
sequence_length = 4
data_name = 'full_data'

# Process and save the data
data = load_data(folder_path, state_abbreviations)

# Normalize the inputs and targets
inputs, targets = normalize_outputs(data, sequence_length)

# Split the data
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_data(inputs, targets)

# Create cnn data objects for train, validation, and test sets
train_data = create_lstm_data(train_inputs, train_targets)
val_data = create_lstm_data(val_inputs, val_targets)
test_data = create_lstm_data(test_inputs, test_targets)

# Save the lstm data
save_lstm_data(train_data, sequence_length, data_name, 'train', output_folder_path)
save_lstm_data(val_data, sequence_length, data_name, 'val', output_folder_path)
save_lstm_data(test_data, sequence_length, data_name, 'test', output_folder_path)
