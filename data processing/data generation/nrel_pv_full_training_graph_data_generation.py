import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from sklearn.preprocessing import LabelEncoder
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
    Normalizes time-series data and structures it into input sequences and targets for model training.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data with columns named with 'Minute_' as prefix and a 'CapacityMW' column.
        sequence_length (int): The number of time steps to include in each input sequence.

    Returns:
        tuple: A tuple containing:
            - inputs (np.array): Input sequences shaped as (num_sequences, nodes, sequence_length).
            - targets (np.array): Target output values for each sequence shaped as (num_sequences, nodes).

    Notes:
        Assumes that each 'Minute_' prefixed column represents a different time step and that 'CapacityMW'
        can contain non-numeric values with 'MW' suffix which are cleaned and converted to float.
    """
    # Identify output columns that start with 'Minute_'
    output_columns = [col for col in data.columns if col.startswith('Minute_')]
    
    # Clean and convert the 'CapacityMW' column from string to float after removing 'MW'
    data['CapacityMW'] = data['CapacityMW'].str.replace('MW', '').astype(float)
    
    # Normalize the output columns by the 'CapacityMW' column
    normalized_data = data[output_columns].div(data['CapacityMW'], axis=0).values
    
    # Transpose the data to align time steps as rows, shape becomes (time_steps, nodes)
    normalized_data = normalized_data.T  # Example shape (105120, 405)

    # Calculate the number of sequences that can be generated given the sequence length
    num_sequences = normalized_data.shape[0] - sequence_length
    
    # Create input sequences by iterating over normalized data
    inputs = np.array([normalized_data[i:i+sequence_length] for i in range(num_sequences)])

    # The targets are the values immediately following each sequence
    targets = normalized_data[sequence_length:]

    # Reshape inputs to fit the model's expected input shape: (num_sequences, nodes, sequence_length)
    inputs = inputs.transpose(0, 2, 1)
    
    return inputs, targets


def construct_graph(data, method='delaunay', k=5):
    """
    Constructs a graph from geographical coordinates using Delaunay triangulation or k-nearest neighbors.

    Args:
        data (pd.DataFrame): DataFrame containing 'Latitude' and 'Longitude' columns.
        method (str, optional): Method to construct the graph. Options are 'delaunay' or 'knn'. Defaults to 'delaunay'.
        k (int, optional): Number of neighbors for the k-nearest neighbors graph. Defaults to 5.

    Returns:
        torch.Tensor: Edge index tensor representing the graph connectivity in COO format.

    Notes:
        - The 'delaunay' method uses the scipy.spatial.Delaunay for triangulation.
        - The 'knn' method uses sklearn's NearestNeighbors for finding k-nearest neighbors.
    """
    coordinates = data[['Latitude', 'Longitude']].values
    
    if method == 'delaunay':
        # Perform Delaunay triangulation
        tri = Delaunay(coordinates)
        # Extract edges from Delaunay triangulation
        edges = np.vstack({tuple(sorted(item)) for simplex in tri.simplices for item in zip(simplex[:-1], simplex[1:])})
        # Convert edge list to PyTorch tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif method == 'knn':
        # Initialize NearestNeighbors with k+1 because it includes self-loops
        neighbors = NearestNeighbors(n_neighbors=k + 1)
        neighbors.fit(coordinates)
        # Generate adjacency matrix and remove self-loops
        adjacency_matrix = neighbors.kneighbors_graph(coordinates, mode='connectivity').toarray()
        np.fill_diagonal(adjacency_matrix, 0)
        # Convert adjacency matrix to edge index tensor
        edge_index = torch.tensor(np.nonzero(adjacency_matrix), dtype=torch.long)
    
    return edge_index

def create_graph_data(data, inputs, targets, edge_index, start_idx, end_idx):
    """
    Prepares graph data by combining node features with input sequences and targets for each graph snapshot.

    Args:
        data (pd.DataFrame): DataFrame containing node attributes including 'PV Type', 'Latitude', 'Longitude', and 'CapacityMW'.
        inputs (np.array): Array of input features for each node over a sequence of time steps.
        targets (np.array): Array of target output values corresponding to the input sequences.
        edge_index (torch.Tensor): Tensor describing the edges of the graph in COO format.
        start_idx (int): Starting index for assigning time indices to graph snapshots.
        end_idx (int): Ending index (not inclusive) for time indices.

    Returns:
        list: A list of Data objects from PyTorch Geometric, each representing a graph with features and targets.

    Notes:
        Each Data object contains node features combined from static attributes and time-varying inputs,
        connectivity information via `edge_index`, target values, and a time index for tracking.
    """
    # Encode 'PV Type' using label encoding
    label_encoder = LabelEncoder()
    pv_type_encoded = label_encoder.fit_transform(data['PV Type'])
    
    # Stack static node features (Latitude, Longitude, CapacityMW) with the encoded PV Type
    node_features = np.column_stack([data[['Latitude', 'Longitude', 'CapacityMW']].values, pv_type_encoded])
    
    # Convert node features to a tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Initialize list to hold graph data objects
    graph_data = []
    i = start_idx
    for input_seq, target in zip(inputs, targets):
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Concatenate static and time-varying features for the complete feature set
        full_features = torch.cat([node_features, input_features], dim=1)
        
        # Create a graph data object for this snapshot
        graph_data.append(Data(x=full_features, edge_index=edge_index, y=target_features, time_idx=i))
        i += 1
    return graph_data


def save_graph_data(graph_data, sequence_length, method, k, data_name, split, folder_path="generated_data"):
    """
    Saves processed graph data to a file in a specified directory.

    Args:
        graph_data (list): A list of PyTorch Geometric Data objects containing the graph information.
        sequence_length (int): The length of the input sequences used in the graph data.
        method (str): The method used to construct the graph (e.g., 'delaunay' or 'knn').
        k (int): The number of neighbors considered in the k-nearest neighbors graph construction.
        data_name (str): A base name for the data, typically indicating the source or type of data.
        split (str): Specifies the dataset split (e.g., 'train', 'val', 'test').
        folder_path (str, optional): The directory where the graph data files will be saved. Defaults to "generated_data".

    Notes:
        The function constructs a filename using the provided parameters to reflect the contents and settings used
        to generate the data. It ensures the directory exists before saving and informs the user of the file location.
    """
    # Ensure the output directory exists; create it if necessary
    os.makedirs(folder_path, exist_ok=True)
    
    # Format the filename to include data characteristics for easy identification
    file_name = f"{data_name}_{split}_seq{sequence_length}_{method}_k{k}_c.pt"
    file_path = os.path.join(folder_path, file_name)
    
    # Save the graph data to the specified file path using PyTorch's serialization utility
    torch.save(graph_data, file_path)
    
    # Notify the user of the successful save operation
    print(f"{split.capitalize()} data saved to {file_path}")

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into training, validation, and testing sets based on specified ratios.

    Args:
        inputs (np.array or torch.Tensor): The input features array.
        targets (np.array or torch.Tensor): The corresponding target values array.
        train_ratio (float, optional): The proportion of data to be used for training. Defaults to 0.7.
        val_ratio (float, optional): The proportion of data to be used for validation. Defaults to 0.15.

    Returns:
        tuple: Contains three tuples, each with inputs and targets for training, validation, and testing:
               (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)
    """
    total_sequences = inputs.shape[0]  # Get the total number of sequences available

    # Calculate the indices for the end of each data segment
    train_end = int(total_sequences * train_ratio)  # Index for the last training data item
    val_end = int(total_sequences * (train_ratio + val_ratio))  # Index for the last validation data item

    # Slice the data into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]

    # Return the dataset splits as tuples
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_end, val_end, total_sequences


# Configuration
folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data'
state_abbreviations = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 
    'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 
    'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
]

output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\graph data'
sequence_length = 4
method = 'delaunay'
k = 10
data_name = 'full_data'

# Process and load the data
data = load_data(folder_path, state_abbreviations)

# Extract the required columns
selected_columns = data[['Latitude', 'Longitude', 'CapacityMW', 'PV Type']]

# Normalize the inputs and targets
inputs, targets = normalize_outputs(data, sequence_length)
# Construct the graph (using Delaunay or k-nearest neighbors)
edge_index = construct_graph(data, method=method, k=k)

# Split the data
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_last_idx, val_last_idx, last_idx = split_data(inputs, targets)

# Create graph data objects for train, validation, and test sets
train_data = create_graph_data(data, train_inputs, train_targets, edge_index, 0, train_last_idx)
val_data = create_graph_data(data, val_inputs, val_targets, edge_index, train_last_idx, val_last_idx)
test_data = create_graph_data(data, test_inputs, test_targets, edge_index, val_last_idx, last_idx)

# Save the graph data
save_graph_data(train_data, sequence_length, method, k, data_name, 'train', output_folder_path)
save_graph_data(val_data, sequence_length, method, k, data_name, 'val', output_folder_path)
save_graph_data(test_data, sequence_length, method, k, data_name, 'test', output_folder_path)
