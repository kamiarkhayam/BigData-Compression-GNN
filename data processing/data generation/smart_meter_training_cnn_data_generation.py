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
        np.array: Inputs (shape: [num_sequences, 1, sequence_length, num_nodes]) suitable for CNN models.
        np.array: Targets (shape: [num_sequences, num_nodes]) corresponding to the values immediately following each input sequence.
    """
    
    # Extract consumption columns from the DataFrame (columns containing ':' are assumed to represent timestamps)
    consumption_columns = [col for col in data.columns if ':' in col]  # Assuming date format in headers
    
    # Initialize empty lists to store the inputs and targets
    inputs = []
    targets = []
    
    # Loop through the data to generate sequences of consumption data
    for i in range(data[consumption_columns].shape[1] - sequence_length):
        # Extract a sequence of length 'sequence_length' and reshape it to fit CNN input shape [1, sequence_length, num_nodes]
        inputs.append(data[consumption_columns[i:i+sequence_length]].values.T.reshape((1, sequence_length, -1)))
        
        # The target is the consumption value right after the input sequence
        targets.append(data[consumption_columns[i+sequence_length]].values)
    
    # Convert the lists of inputs and targets to NumPy arrays
    return np.array(inputs), np.array(targets)


def create_cnn_data(inputs, targets):
    """
    Converts input sequences and targets into a list of PyTorch Geometric Data objects.

    Args:
        inputs (np.array): Input sequences of shape (num_sequences, 1, sequence_length, num_nodes).
        targets (np.array): Target values of shape (num_sequences, num_nodes).

    Returns:
        list: A list of PyTorch Geometric Data objects where each object contains the input features and the target values.
    """
    
    # Initialize an empty list to store the Data objects
    all_data = []
    
    # Loop through each input sequence and its corresponding target
    for input_seq, target in zip(inputs, targets):
        
        # Convert the input sequence and target to PyTorch tensors (float type)
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Create a PyTorch Geometric Data object, where `x` is the input and `y` is the target
        all_data.append(Data(x=input_features, y=target_features))

    # Return the list of Data objects
    return all_data


def save_cnn_data(data, sequence_length, data_name, split, folder_path="generated_data"):
    """
    Saves the CNN data (a list of PyTorch Geometric Data objects) to a specified directory in .pt format.
    
    Args:
        data (list): A list of PyTorch Geometric Data objects containing the input features and targets.
        sequence_length (int): The length of the input sequences, used for naming the output file.
        data_name (str): The name of the dataset, used in the file naming.
        split (str): The data split (e.g., 'train', 'val', or 'test'), used in the file naming.
        folder_path (str): The folder where the data will be saved. Defaults to "generated_data".
    """
    
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the file name using the data_name, split, and sequence_length for clarity
    file_name = f"{data_name}_{split}_seq{sequence_length}.pt"
    
    # Create the full file path by combining the folder path and the file name
    file_path = os.path.join(folder_path, file_name)
    
    # Save the data (list of Data objects) to the specified file path using PyTorch's save function
    torch.save(data, file_path)
    
    # Print confirmation message to indicate successful save
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input sequences and targets into training, validation, and test sets based on the provided ratios.

    Args:
        inputs (np.array): Input sequences, typically of shape (num_sequences, sequence_length, num_nodes).
        targets (np.array): Target sequences, typically of shape (num_sequences, num_nodes).
        train_ratio (float): The proportion of data to be used for training. Defaults to 0.7 (70%).
        val_ratio (float): The proportion of data to be used for validation. Defaults to 0.15 (15%).

    Returns:
        tuple: Returns three tuples, each containing inputs and targets for:
            - Training set
            - Validation set
            - Test set
    """
    
    # Determine the total number of sequences
    total_sequences = inputs.shape[0]
    
    # Calculate the indices for splitting the data
    train_end = int(total_sequences * train_ratio)  # End index for the training set
    val_end = int(total_sequences * (train_ratio + val_ratio))  # End index for the validation set
    
    # Split the data into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]  # Training set
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]  # Validation set
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]  # Test set
    
    # Return the split data
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


# Define file paths and sequence length
file_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit\\cumulative_data_9digit_master.csv'
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\cnn data'
sequence_length = 4  # Number of time steps in each input sequence

# Name of the dataset used in file names
data_name = 'smart_meter_data_9digit'

# Load the data from the CSV file
data = load_data(file_path)

# Generate input sequences and their corresponding targets based on sequence length
inputs, targets = generate_sequences(data, sequence_length)

# Split the data into training, validation, and test sets
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_data(inputs, targets)

# Create CNN-compatible data objects for training, validation, and testing sets
train_data = create_cnn_data(train_inputs, train_targets)  # Create training data
val_data = create_cnn_data(val_inputs, val_targets)        # Create validation data
test_data = create_cnn_data(test_inputs, test_targets)     # Create test data

# Save the processed data to disk in .pt format for easy re-use
save_cnn_data(train_data, sequence_length, data_name, 'train', output_folder_path)  # Save training data
save_cnn_data(val_data, sequence_length, data_name, 'val', output_folder_path)      # Save validation data
save_cnn_data(test_data, sequence_length, data_name, 'test', output_folder_path)    # Save test data
