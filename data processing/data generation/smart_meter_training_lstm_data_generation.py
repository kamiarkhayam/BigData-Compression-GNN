import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os


def load_data(file_path):
    """
    Loads the data from a CSV file, removes any rows with missing values (NaNs), 
    and resets the index to ensure sequential indexing.

    Args:
        file_path (str): The file path to the CSV file containing the data.

    Returns:
        pd.DataFrame: The cleaned data as a Pandas DataFrame with no missing values.
    """
    
    # Load the data from the CSV file into a Pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Remove rows with missing values (NaNs)
    data = data.dropna()
    
    # Reset the index of the DataFrame after dropping rows, ensuring sequential indexing
    data.reset_index(drop=True, inplace=True)
    
    return data


def generate_sequences(data, sequence_length):
    """
    Generates sequences of consumption data for time-series modeling, such as CNN or LSTM models.
    
    Args:
        data (pd.DataFrame): The data containing consumption columns, assumed to be timestamped (e.g., columns contain time info).
        sequence_length (int): The number of time steps in each input sequence.

    Returns:
        np.array: Inputs (shape: [num_sequences, sequence_length, num_nodes]).
        np.array: Targets (shape: [num_sequences, num_nodes]).
    """
    
    # Extract columns that represent consumption data (those containing a ':' indicating time or date format)
    consumption_columns = [col for col in data.columns if ':' in col]  # Date format in headers
    
    # Initialize empty lists to hold the input sequences and target sequences
    inputs = []
    targets = []
    
    # Loop over the data to create sequences and their corresponding targets
    for i in range(data[consumption_columns].shape[1] - sequence_length):
        # Extract the input sequence of length 'sequence_length'
        input_seq = data[consumption_columns[i:i + sequence_length]].values  # Shape: [sequence_length, num_nodes]
        
        # Extract the target sequence, which is the value immediately following the input sequence
        target_seq = data[consumption_columns[i + sequence_length]].values  # Shape: [num_nodes]
        
        # Append the input sequence and target sequence to the respective lists
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Convert the lists of inputs and targets into NumPy arrays for efficient processing
    return np.array(inputs), np.array(targets)


def create_lstm_data(inputs, targets):
    """
    Converts input sequences and targets into a list of PyTorch Geometric Data objects for LSTM-based models.
    
    Args:
        inputs (np.array): Input sequences of shape (num_sequences, sequence_length, num_nodes).
        targets (np.array): Target sequences of shape (num_sequences, num_nodes).
        
    Returns:
        list: A list of PyTorch Geometric Data objects, where each object contains input features (x) and targets (y).
    """
    
    # Initialize an empty list to store the Data objects
    all_data = []
    
    # Loop over each input sequence and its corresponding target
    for input_seq, target in zip(inputs, targets):
        # Convert the input sequence to a PyTorch tensor of type float
        input_features = torch.tensor(input_seq, dtype=torch.float)
        
        # Convert the target sequence to a PyTorch tensor of type float
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Create a PyTorch Geometric Data object and store the input as 'x' and target as 'y'
        all_data.append(Data(x=input_features, y=target_features))
    
    # Return the list of Data objects
    return all_data


def save_lstm_data(data, sequence_length, data_name, split, folder_path="generated_data"):
    """
    Saves the LSTM data (a list of PyTorch Geometric Data objects) to a specified directory in .pt format.
    
    Args:
        data (list): A list of PyTorch Geometric Data objects containing the input features and targets.
        sequence_length (int): The length of the input sequences, used for naming the output file.
        data_name (str): The name of the dataset, used in the file naming.
        split (str): The data split (e.g., 'train', 'val', or 'test'), used in the file naming.
        folder_path (str): The folder where the data will be saved. Defaults to "generated_data".
    """
    
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the file name using data_name, split (train/val/test), and sequence_length
    file_name = f"{data_name}_{split}_seq{sequence_length}.pt"
    
    # Create the full file path by combining the folder path and the file name
    file_path = os.path.join(folder_path, file_name)
    
    # Save the list of PyTorch Geometric Data objects to the file path using torch.save
    torch.save(data, file_path)
    
    # Print confirmation message to indicate successful save
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input sequences and corresponding targets into training, validation, and test sets.
    
    Args:
        inputs (np.array): Input sequences, typically of shape (num_sequences, sequence_length, num_nodes).
        targets (np.array): Target sequences, typically of shape (num_sequences, num_nodes).
        train_ratio (float): The proportion of data to be used for training. Defaults to 0.7 (70%).
        val_ratio (float): The proportion of data to be used for validation. Defaults to 0.15 (15%).

    Returns:
        tuple: Three tuples, each containing inputs and targets for:
            - Training set
            - Validation set
            - Test set
    """
    
    # Determine the total number of sequences in the dataset
    total_sequences = inputs.shape[0]
    
    # Calculate the index where the training data ends
    train_end = int(total_sequences * train_ratio)
    
    # Calculate the index where the validation data ends
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # Split the inputs and targets into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]  # Training data
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]  # Validation data
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]  # Test data
    
    # Return the three sets: (train, validation, test)
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


# File path for the input data (smart meter data in CSV format)
file_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit\\cumulative_data_9digit_master.csv'

# Folder path to save the processed LSTM data
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\lstm data'

# Define the sequence length (number of timesteps in each input sequence)
sequence_length = 4

# Define the name for the dataset (used for saving files)
data_name = 'smart_meter_data_9digit'

# Load the raw smart meter data from the specified file path
data = load_data(file_path)

# Generate input sequences and corresponding targets from the data
inputs, targets = generate_sequences(data, sequence_length)

# Split the data into training, validation, and test sets
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_data(inputs, targets)

# Create LSTM-friendly data objects (PyTorch Geometric Data) for each split
train_data = create_lstm_data(train_inputs, train_targets)  # Training data
val_data = create_lstm_data(val_inputs, val_targets)        # Validation data
test_data = create_lstm_data(test_inputs, test_targets)      # Test data

# Save the processed LSTM data to the output folder, organized by train, val, and test splits
save_lstm_data(train_data, sequence_length, data_name, 'train', output_folder_path)
save_lstm_data(val_data, sequence_length, data_name, 'val', output_folder_path)
save_lstm_data(test_data, sequence_length, data_name, 'test', output_folder_path)
