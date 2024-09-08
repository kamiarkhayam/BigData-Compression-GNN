import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from pyzipcode import ZipCodeDatabase
import os

def load_data(file_path):
    """
    Loads a CSV file, processes ZIP codes to retrieve corresponding coordinates, 
    and attaches the coordinates to the DataFrame. Drops rows with NaN values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame with added latitude and longitude columns.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Drop rows containing NaN values and reset the index
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    
    # Process ZIP codes (ensure they are strings and keep only the first 5 digits)
    data['ZIP_CODE'] = data['ZIP_CODE'].astype(str).str[:5]
    
    # Extract ZIP codes and fetch their coordinates using a custom function
    zip_codes = data['ZIP_CODE']
    coordinates = get_zip_code_coordinates(zip_codes)  # This function should return coordinates for each ZIP code
    
    # Convert the coordinates list to a DataFrame with appropriate columns
    coord_df = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude'])
    
    # Attach the coordinates to the original DataFrame by adding latitude and longitude columns
    data['Latitude'] = coord_df['Latitude']
    data['Longitude'] = coord_df['Longitude']
    
    # Check if there are any NaN values in the DataFrame and print a warning if necessary
    if data.isnull().values.any():
        print("Warning: The data contains NaN values.")
        print(data.isnull().sum())  # Print the count of NaNs in each column
    else:
        print("No NaN values found in the data.")
    
    return data


def get_zip_code_coordinates(zip_codes):
    """
    Fetches latitude and longitude coordinates for a list of ZIP codes.

    Args:
        zip_codes (pd.Series or list): A list or Pandas Series of ZIP codes.

    Returns:
        list: A list of tuples, where each tuple contains the latitude and longitude of the corresponding ZIP code.
               If a ZIP code is not found, a default coordinate is returned.
    """
    zcdb = ZipCodeDatabase()  # Initialize the ZipCodeDatabase object
    coordinates = []
    
    # Loop through each ZIP code and fetch its coordinates
    for zip_code in zip_codes:
        try:
            # Convert ZIP code to integer (in case it's a string) and fetch from the database
            zipcode = zcdb[int(zip_code)]
            coordinates.append((zipcode.latitude, zipcode.longitude))
        except Exception as e:
            # Handle missing or invalid ZIP codes by appending a default coordinate (e.g., for 60418)
            print(f"ZIP code {zip_code} not found in the database. Using default coordinates. Error: {e}")
            coordinates.append((41.65469091300716, -87.75262407840017))  # 60418 coordinates

    return coordinates


def generate_sequences(data, sequence_length):
    """
    Generates input sequences and corresponding targets from time-series data.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data, where each column represents a timestamp.
        sequence_length (int): The length of the input sequences.

    Returns:
        tuple: 
            - inputs (np.array): Input sequences shaped as (num_sequences, sequence_length, num_features).
            - targets (np.array): Corresponding target values shaped as (num_sequences, num_features).
    """
    # Extract columns that contain time-series data, assuming they have a ':' in the header
    consumption_columns = [col for col in data.columns if ':' in col]
    
    # Calculate the number of sequences to generate
    num_sequences = len(consumption_columns) - sequence_length
    
    # Initialize lists to hold the inputs and targets
    inputs = []
    targets = []
    
    # Generate sequences and corresponding targets
    for i in range(num_sequences):
        # Create an input sequence of 'sequence_length'
        input_sequence = data[consumption_columns[i:i + sequence_length]].values  # Shape: [sequence_length, num_features]
        
        # The target is the next time step after the input sequence
        target_sequence = data[consumption_columns[i + sequence_length]].values  # Shape: [num_features]
        
        # Append the input sequence and target to their respective lists
        inputs.append(input_sequence)
        targets.append(target_sequence)
    
    # Convert lists to numpy arrays for easier handling in machine learning models
    return np.array(inputs), np.array(targets)


def construct_graph(data, method='delaunay', k=5):
    """
    Constructs a graph from geographical coordinates using Delaunay triangulation or k-nearest neighbors (k-NN).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Latitude' and 'Longitude' columns.
        method (str, optional): Method for constructing the graph. Can be 'delaunay' or 'knn'. Defaults to 'delaunay'.
        k (int, optional): Number of nearest neighbors for k-NN graph. Defaults to 5.

    Returns:
        torch.Tensor: The edge index (2D tensor) representing the graph's edges.
    """
    # Extract coordinates (latitude and longitude) from the data
    coordinates = data[['Latitude', 'Longitude']].values
    
    if method == 'delaunay':
        # Use Delaunay triangulation to create a graph where edges connect neighboring triangles
        tri = Delaunay(coordinates)
        
        # Extract edges from the Delaunay triangulation simplices (unique sorted pairs of vertices)
        edges = np.vstack(list({tuple(sorted(item)) for simplex in tri.simplices for item in zip(simplex[:-1], simplex[1:])}))
        
        # Convert the edges to a PyTorch tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    elif method == 'knn':
        # Use k-nearest neighbors to connect each point to its k closest neighbors
        neighbors = NearestNeighbors(n_neighbors=k+1)  # k+1 because the closest point is the point itself
        neighbors.fit(coordinates)
        
        # Create an adjacency matrix where the value is 1 if there is a connection between two points
        adjacency_matrix = neighbors.kneighbors_graph(coordinates, mode='connectivity').toarray()
        
        # Remove self-loops (diagonal entries should be zero)
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Extract the non-zero entries as edges (where there is a connection between nodes)
        edge_index = torch.tensor(np.nonzero(adjacency_matrix), dtype=torch.long)
    
    return edge_index


def create_graph_data(data, inputs, targets, edge_index, start_idx, end_idx):
    """
    Creates a list of PyTorch Geometric Data objects for graph-based learning tasks.

    Args:
        data (pd.DataFrame): The DataFrame containing node features like 'Latitude' and 'Longitude'.
        inputs (np.array): The input sequences (shape: [num_sequences, sequence_length, num_features]).
        targets (np.array): The target values corresponding to the input sequences (shape: [num_sequences, num_features]).
        edge_index (torch.Tensor): The edge index tensor representing graph edges (shape: [2, num_edges]).
        start_idx (int): The starting index for the time step (e.g., for time-aware models).
        end_idx (int): The ending index for the time step.

    Returns:
        list: A list of PyTorch Geometric `Data` objects where each object contains the node features, edge index, input features, and target values.
    """
    
    # Extract latitude and longitude as node features and reshape them into a 2D array
    node_features = np.array([data[['Latitude', 'Longitude']].values]).reshape(-1, 2)
    
    # Convert node features to a PyTorch tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Initialize an empty list to store the graph data objects
    graph_data = []
    
    # Iterate over input sequences and their corresponding target values
    i = start_idx
    for input_seq, target in zip(inputs, targets):
        # Convert the input sequence and target to PyTorch tensors
        input_features = torch.tensor(input_seq, dtype=torch.float)
        target_features = torch.tensor(target, dtype=torch.float)
        
        # Combine node features (latitude, longitude) with input features
        full_features = torch.cat([node_features, input_features], dim=1)
        
        # Create a PyTorch Geometric Data object and append it to the list
        graph_data.append(Data(x=full_features, edge_index=edge_index, y=target_features, time_idx=i))
        
        # Increment the time index for the next sequence
        i += 1
    
    return graph_data


def save_graph_data(graph_data, sequence_length, method, k, data_name, split, folder_path="generated_data"):
    """
    Saves the graph data to a specified file in .pt format.

    Args:
        graph_data (list): List of graph data objects (usually instances of torch_geometric.data.Data).
        sequence_length (int): The length of input sequences (used for naming the file).
        method (str): The method used to construct the graph (e.g., 'delaunay', 'knn').
        k (int): Number of neighbors (used when method is k-NN).
        data_name (str): The name of the dataset (used for naming the file).
        split (str): The data split ('train', 'val', or 'test').
        folder_path (str): The folder where the data will be saved. Defaults to 'generated_data'.
    """
    
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the file name using the parameters
    file_name = f"{data_name}_{split}_seq{sequence_length}_{method}_k{k}.pt"
    
    # Join the folder path and file name to create the full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Save the graph data using PyTorch's save function
    torch.save(graph_data, file_path)
    
    # Print confirmation message after saving
    print(f"{split.capitalize()} data saved to {file_path}")
    

def split_data(inputs, targets, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the input and target data into training, validation, and test sets.

    Args:
        inputs (np.array): The input sequences (shape: [num_sequences, sequence_length, num_features]).
        targets (np.array): The target sequences (shape: [num_sequences, num_features]).
        train_ratio (float): The proportion of the data to use for training. Defaults to 0.7 (70%).
        val_ratio (float): The proportion of the data to use for validation. Defaults to 0.15 (15%).

    Returns:
        tuple: 
            - (train_inputs, train_targets): The training inputs and targets.
            - (val_inputs, val_targets): The validation inputs and targets.
            - (test_inputs, test_targets): The test inputs and targets.
            - train_end (int): The index where the training data ends.
            - val_end (int): The index where the validation data ends.
            - total_sequences (int): The total number of sequences.
    """
    
    # Calculate the total number of sequences
    total_sequences = inputs.shape[0]
    
    # Calculate the indices for splitting the data
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # Split the inputs and targets into training, validation, and test sets
    train_inputs, train_targets = inputs[:train_end], targets[:train_end]
    val_inputs, val_targets = inputs[train_end:val_end], targets[train_end:val_end]
    test_inputs, test_targets = inputs[val_end:], targets[val_end:]
    
    # Return the split data and indices
    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_end, val_end, total_sequences


# File paths and settings
file_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit\\cumulative_data_5digit_master.csv'
output_folder_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data'
sequence_length = 4  # Length of input sequences (time steps)
method = 'delaunay'  # Method for graph construction (can be 'delaunay' or 'knn')
k = 10  # Number of neighbors for k-NN method (only relevant if 'knn' is used)
data_name = 'smart_meter_data_ma'  # Name for the dataset used in file names

# Process and save the data
data = load_data(file_path)  # Load the data from the CSV file

# Generate input sequences and corresponding targets from the data
inputs, targets = generate_sequences(data, sequence_length)

# Construct the graph (either with Delaunay triangulation or k-NN)
edge_index = construct_graph(data, method=method, k=k)

# Split the data into training, validation, and test sets
(train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets), train_last_idx, val_last_idx, last_idx = split_data(inputs, targets)

# Create graph data objects for training, validation, and test sets
train_data = create_graph_data(data, train_inputs, train_targets, edge_index, 0, train_last_idx)
val_data = create_graph_data(data, val_inputs, val_targets, edge_index, train_last_idx, val_last_idx)
test_data = create_graph_data(data, test_inputs, test_targets, edge_index, val_last_idx, last_idx)

# Save the training, validation, and test graph data to the specified folder
save_graph_data(train_data, sequence_length, method, k, data_name, 'train', output_folder_path)
save_graph_data(val_data, sequence_length, method, k, data_name, 'val', output_folder_path)
save_graph_data(test_data, sequence_length, method, k, data_name, 'test', output_folder_path)
