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
    Normalizes time-series data and structures it into input sequences and targets for a CNN.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data with 'Minute_' prefixed columns and 'CapacityMW'.
        sequence_length (int): Number of time steps to include in each input sequence.

    Returns:
        tuple: A tuple containing:
            - inputs (np.array): Array of input sequences shaped as (num_sequences, 1, sequence_length, num_nodes).
            - targets (np.array): Array of target output values shaped as (num_sequences, num_nodes).

    Notes:
        - 'Minute_' columns represent time-series data at different time steps for each node.
        - 'CapacityMW' is assumed to contain capacity values, potentially with 'MW' as a suffix that needs to be removed.
    """
    # Identify the output columns that contain the time-series data ('Minute_' prefixed columns)
    output_columns = [col for col in data.columns if col.startswith('Minute_')]
    
    # Convert 'CapacityMW' to a float by removing any 'MW' suffix
    data['CapacityMW'] = data['CapacityMW'].str.replace('MW', '').astype(float)
    
    # Normalize the time-series data (Minute_ columns) by the 'CapacityMW' values
    # Transpose the normalized data so that each row corresponds to a time step, and each column corresponds to a node
    normalized_data = data[output_columns].div(data['CapacityMW'], axis=0).values.T  # shape: (time_steps, num_nodes)
    
    # Initialize lists to store inputs and targets
    inputs = []
    targets = []
    
    # Generate input sequences and corresponding targets
    for i in range(normalized_data.shape[0] - sequence_length):
        # For each sequence, we take `sequence_length` time steps as inputs
        # Reshape the input to fit the CNN expected input shape: [1, sequence_length, num_nodes]
        inputs.append(normalized_data[i:i+sequence_length].reshape((1, sequence_length, normalized_data.shape[1])))
        # The target is the value immediately following the input sequence
        targets.append(normalized_data[i + sequence_length])
    
    # Convert inputs and targets to numpy arrays
    return np.array(inputs), np.array(targets)


def create_cnn_data(inputs, targets):
    """
    Creates a list of PyTorch Geometric Data objects containing input features and corresponding target values.

    Args:
        inputs (np.array): Input sequences for the CNN (expected shape: [num_sequences, 1, sequence_length, num_nodes]).
        targets (np.array): Target values corresponding to the input sequences (expected shape: [num_sequences, num_nodes]).

    Returns:
        list: A list of PyTorch Geometric Data objects, each containing input features (x) and target values (y).
    """
    all_data = []
    for input_seq, target in zip(inputs, targets):
        # Convert input and target into torch tensors
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Create a Data object and append to the list
        all_data.append(Data(x=input_features, y=target_features))

    return all_data


def save_cnn_data(data, sequence_length, split, data_name="cnn_data", folder_path="generated_data"):
    """
    Saves CNN data to a file in the specified folder.

    Args:
        data (list): List of PyTorch Geometric Data objects containing inputs and targets.
        sequence_length (int): Length of the input sequences.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        data_name (str, optional): Base name for the saved file. Defaults to "cnn_data".
        folder_path (str, optional): Directory where the data will be saved. Defaults to "generated_data".

    Notes:
        The function constructs a file name based on the data name, sequence length, and split, and saves the data using torch.save().
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the file name and full path
    file_name = f"{data_name}_{split}_seq{sequence_length}_c.pt"
    file_path = os.path.join(folder_path, file_name)
    
    # Save the data
    torch.save(data, file_path)
    
    # Notify the user
    print(f"{split.capitalize()} data saved to {file_path}")


def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input and target data into training, validation, and test sets based on the provided ratios.

    Args:
        inputs (np.array): Input sequences for the CNN.
        targets (np.array): Target values corresponding to the input sequences.
        train_ratio (float, optional): Proportion of data used for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of data used for validation. Defaults to 0.15.

    Returns:
        tuple: Three tuples containing the inputs and targets for training, validation, and testing:
            (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets).
    """
    total_sequences = inputs.shape[0]  # Get the total number of sequences
    
    # Calculate the indices for splitting the dataset
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
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\cnn data'
sequence_length = 4
data_name = 'full_data'

# Process and save the data
data = load_data(folder_path, state_abbreviations)

# Normalize the inputs and targets
inputs, targets = normalize_outputs(data, sequence_length)

# Split the data
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_data(inputs, targets)

# Create cnn data objects for train, validation, and test sets
train_data = create_cnn_data(train_inputs, train_targets)
val_data = create_cnn_data(val_inputs, val_targets)
test_data = create_cnn_data(test_inputs, test_targets)

# Save the cnn data
save_cnn_data(train_data, sequence_length,  data_name, 'train', output_folder_path)
save_cnn_data(val_data, sequence_length,  data_name, 'val', output_folder_path)
save_cnn_data(test_data, sequence_length,  data_name, 'test', output_folder_path)
