import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



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
    Generates input sequences and corresponding target values from a time-series dataset.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data, where each column represents a timestamp.
        sequence_length (int): The number of time steps in each input sequence.

    Returns:
        tuple: A tuple containing:
            - inputs (np.array): Input sequences shaped as (num_sequences, sequence_length, num_nodes).
            - targets (np.array): Target values shaped as (num_sequences, num_nodes).
    """
    # Extract columns with consumption data (assuming date format in headers, e.g., 'YYYY-MM-DD' or 'HH:MM')
    consumption_columns = [col for col in data.columns if ':' in col]
    
    # Calculate the number of sequences that can be generated given the sequence length
    num_sequences = len(consumption_columns) - sequence_length
    
    # Initialize lists to store inputs and targets
    inputs = []
    targets = []
    
    # Loop to generate sequences and corresponding target values
    for i in range(num_sequences):
        # Extract a sequence of 'sequence_length' time steps for inputs
        input_sequence = data[consumption_columns[i:i+sequence_length]].values  # Shape: [num_nodes, sequence_length]
        
        # The target is the next time step following the input sequence
        target_sequence = data[consumption_columns[i+sequence_length]].values  # Shape: [num_nodes]
        
        # Append input sequence and target to their respective lists
        inputs.append(input_sequence)
        targets.append(target_sequence)
    
    # Convert lists to numpy arrays for easier handling in machine learning models
    return np.array(inputs), np.array(targets)


def construct_graph(data, k=3):
    """
    Constructs a graph from time-series data using cosine similarity and k-nearest neighbors.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data, where each column represents a timestamp.
        k (int, optional): The number of nearest neighbors to connect in the graph. Defaults to 3.

    Returns:
        torch.Tensor: Edge index representing the graph structure as a tensor of shape (2, num_edges).
    """
    # Extract columns with traffic or time-series data (assuming columns with date-time format)
    feature_columns = [col for col in data.columns if ':' in col]  # Assuming date and time format in headers
    
    # Extract feature matrix (nodes x features)
    features = data[feature_columns].values
    
    # Normalize features along each node to prepare for cosine similarity computation
    normalized_features = normalize(features, axis=1)
    
    # Compute the cosine similarity matrix (nodes x nodes)
    similarity_matrix = cosine_similarity(normalized_features)
    
    # For each node, find the k most similar nodes based on cosine similarity
    # argsort returns indices that would sort the similarity scores; take top k (excluding the node itself)
    neighbors = np.argsort(-similarity_matrix, axis=1)[:, 1:k+1]  # Exclude self-comparison by skipping the first index
    
    # Build the edge index based on k-nearest neighbors
    rows = np.repeat(np.arange(neighbors.shape[0]), k)  # Node indices for each row
    cols = neighbors.flatten()  # Corresponding k nearest neighbors for each node
    
    # Combine rows and columns to create the edge index
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    
    # Convert to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    return edge_index


def create_graph_data(data, inputs, targets, edge_index, start_idx, end_idx):
    """
    Creates graph data objects for each input sequence, combining node features, input sequences, and edge index.

    Args:
        data (pd.DataFrame): DataFrame containing node information like 'Lane Type' and 'Station Length'.
        inputs (np.array): Input sequences for the graph model (shape: [num_sequences, sequence_length, num_nodes]).
        targets (np.array): Target sequences corresponding to the inputs (shape: [num_sequences, num_nodes]).
        edge_index (torch.Tensor): Edge index defining the connectivity of the graph (shape: [2, num_edges]).
        start_idx (int): The starting index for the time steps.
        end_idx (int): The ending index for the time steps.

    Returns:
        list: A list of PyTorch Geometric Data objects, each containing node features, input sequences, and targets.
    """
    # Initialize LabelEncoder for 'Lane Type' column
    label_encoder = LabelEncoder()

    # Encode 'Lane Type' to integer labels
    lane_type_encoded = label_encoder.fit_transform(data['Lane Type'])

    # Replace 'Lane Type' in the data with its encoded version
    data['Lane Type'] = lane_type_encoded

    # Extract node-level features: 'Lane Type' and 'Station Length'
    node_features = np.array(data[['Lane Type', 'Station Length']].values)

    # Convert the node features to a PyTorch tensor
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Initialize a list to store the graph data objects
    graph_data = []

    i = start_idx
    
    # Loop through each input sequence and corresponding target sequence
    for input_seq, target in zip(inputs, targets):
        # Convert input sequence and target to PyTorch tensors
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)

        # Concatenate node features with input features along the feature dimension
        full_features = torch.cat([node_features, input_features], dim=1)

        # Create a PyTorch Geometric Data object and add to the list
        graph_data.append(Data(x=full_features, edge_index=edge_index, y=target_features, time_idx=i))
        
        i += 1  # Increment time index

    return graph_data


def save_graph_data(graph_data, sequence_length, method, k, data_name, split, folder_path="generated_data"):
    """
    Saves graph data to a file in the specified folder.

    Args:
        graph_data (list): A list of PyTorch Geometric Data objects containing graph data for a time-series problem.
        sequence_length (int): The length of the input sequences.
        method (str): The method used to construct the graph (e.g., 'knn', 'delaunay').
        k (int): The number of nearest neighbors used in the graph construction (for 'knn' method).
        data_name (str): A name to identify the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        folder_path (str, optional): Directory where the graph data will be saved. Defaults to "generated_data".
    """
    # Ensure the output folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the filename dynamically, incorporating the sequence length, method, and k value
    file_name = f"{data_name}_{split}_seq{sequence_length}_{method}_k{k}.pt"
    file_path = os.path.join(folder_path, file_name)
    
    # Save the graph data using torch.save()
    torch.save(graph_data, file_path)
    
    # Print a confirmation message
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input and target data into training, validation, and test sets based on the specified ratios.

    Args:
        inputs (np.array): Input sequences (shape: [num_sequences, sequence_length, num_nodes]).
        targets (np.array): Corresponding target sequences (shape: [num_sequences, num_nodes]).
        train_ratio (float, optional): Proportion of the data to be used for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of the data to be used for validation. Defaults to 0.15.

    Returns:
        tuple: Three tuples containing the input and target splits for training, validation, and testing:
            - (train_inputs, train_targets)
            - (val_inputs, val_targets)
            - (test_inputs, test_targets)
        int: The index of the last training sample.
        int: The index of the last validation sample.
        int: The total number of sequences.
    """
    # Total number of sequences
    total_sequences = inputs.shape[0]
    
    # Calculate the indices for splitting the dataset
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # Split the data into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]
    
    # Return the splits and the indices
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_end, val_end, total_sequences


# Define file paths and parameters
file_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cleaned_speed_data_full.csv'
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans Traffic Data\\graph data'
sequence_length = 4  # Number of time steps in each sequence
k = 3  # Number of nearest neighbors for graph construction
data_name = 'caltrans_speed_norm_data'  # Name to identify the dataset
method = 'knn'  # Graph construction method

# Load the raw data from the CSV file
data = load_data(file_path)

# Generate sequences of inputs and corresponding targets
inputs, targets = generate_sequences(data, sequence_length)

# Construct the edge index for the graph using the k-nearest neighbors method
edge_index = construct_graph(data, k=k)

# Split the data into training, validation, and test sets
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_last_idx, val_last_idx, last_idx = split_data(inputs, targets)

# Create graph data objects for the training, validation, and test sets
train_data = create_graph_data(data, train_inputs, train_targets, edge_index, 0, train_last_idx)
val_data = create_graph_data(data, val_inputs, val_targets, edge_index, train_last_idx, val_last_idx)
test_data = create_graph_data(data, test_inputs, test_targets, edge_index, val_last_idx, last_idx)

# Save the graph data for each split (training, validation, test) to disk
save_graph_data(train_data, sequence_length, method, k, data_name, 'train', output_folder_path)
save_graph_data(val_data, sequence_length, method, k, data_name, 'val', output_folder_path)
save_graph_data(test_data, sequence_length, method, k, data_name, 'test', output_folder_path)

