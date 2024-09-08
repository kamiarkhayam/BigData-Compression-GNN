import pywt
import scipy.fftpack
from zfpy import compress_numpy, decompress_numpy, _decompress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import torch
import numpy as np
import pandas as pd

def load_data(path, batch_size):
    """
    Loads a dataset from a file and creates a DataLoader for batch processing.

    Args:
        path (str): The file path to the saved dataset.
        batch_size (int): The size of each batch for the DataLoader.
    
    Returns:
        DataLoader: A PyTorch DataLoader for iterating over the dataset.
    """
    # Load the dataset from the specified file path
    dataset = torch.load(path)
    
    # Create a DataLoader to batch the data for efficient processing
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader

def count_parameters(model):
    """
    Counts the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    
    Returns:
        int: The total number of parameters in the model.
    """
    # Sum the number of elements (parameters) in each parameter tensor of the model
    return sum(p.numel() for p in model.parameters())

def make_first_points(results, actuals, test_loader, model_type):
    """
    Extracts the initial points from the first batch of the test loader and appends them to the results and actuals arrays.
    
    Args:
        results (np.array): The array where predicted results will be stored.
        actuals (np.array): The array where actual ground truth values will be stored.
        test_loader (DataLoader): The DataLoader for the test dataset.
        model_type (str): The type of model ('gcn', 'graph_transformer', 'lstm', 'mlp', etc.).

    Returns:
        tuple: Updated results and actuals arrays with the initial points added.
    """
    # Fetch the first batch of data from the test loader
    first_batch = next(iter(test_loader))

    # Check model type and extract the relevant features
    if model_type == 'gcn' or model_type == 'graph_transformer':
        # For GCN and Graph Transformer, extract the node features 'x', skipping the first two columns
        intial_points = first_batch.x[:, 2:].T
    elif model_type == 'lstm' or model_type == 'mlp':
        # For LSTM and MLP, take the entire 'x' feature set and transpose it
        intial_points = first_batch.x.T
    else:
        # For other models (e.g., CNN), squeeze the batch dimension and extract the features
        intial_points = first_batch.x.squeeze(0)
    
    # Append the initial points to the results and actuals arrays
    results = np.append(results, intial_points.cpu().detach().numpy(), axis=0)
    actuals = np.append(actuals, intial_points.cpu().detach().numpy(), axis=0)
    
    return results, actuals


def find_compression_ratio_zfp(matrix, abs_error_threshold, tolerance_range=(2, -3)):
    """
    Finds the compression ratio for a matrix using ZFP compression, given a threshold for the absolute error.
    
    Args:
        matrix (np.array): The input matrix to be compressed.
        abs_error_threshold (float): The maximum allowed absolute error between the original and decompressed matrix.
        tolerance_range (tuple): The range of tolerance values (in log space) to be searched for compression. Defaults to (2, -3).
    
    Returns:
        float: The compression ratio (compressed size / original size) that meets the error threshold.
        None: If no compression tolerance meets the error threshold.
    """
    
    # Define a range of tolerance values to test for compression, using logarithmic spacing
    tolerances = np.logspace(tolerance_range[0], tolerance_range[1], num=1000)  # More points for finer granularity

    # Iterate over each tolerance value to find the first one that meets the absolute error threshold
    for tol in tolerances:
        # Compress the matrix using the specified tolerance
        compressed = compress_numpy(matrix, tolerance=tol)
        
        # Decompress the matrix to recover the original values
        recovered = decompress_numpy(compressed)

        # Calculate the maximum absolute error between the recovered and original matrix
        max_abs_error = np.max(np.abs(recovered - matrix))

        # Check if the maximum absolute error is below the specified threshold
        if max_abs_error < abs_error_threshold:
            # Calculate the size of the compressed data
            compressed_size = len(compressed)
            
            # Calculate the original size of the matrix in bytes
            original_size = len(matrix.tobytes())
            
            # Compute the compression ratio (compressed size / original size)
            compression_ratio = compressed_size / original_size
            
            # Return the compression ratio if the error condition is met
            return compression_ratio

    # If no tolerance value meets the error threshold, return None
    return None


def wavelet_transform_compression(matrix, wavelet, compression_ratio):
    """
    Compresses a matrix using wavelet transformation based on a specified compression ratio.
    
    Args:
        matrix (np.array): The input matrix to be compressed.
        wavelet (str): The type of wavelet to use for the compression (e.g., 'db1', 'haar').
        compression_ratio (float): The compression ratio, where 1 means no compression and 0 means maximum compression.
    
    Returns:
        np.array: The compressed matrix after applying wavelet transformation and thresholding.
    """
    # Perform 2D wavelet decomposition on the matrix
    coeffs = pywt.wavedec2(matrix, wavelet)
    
    # Convert the coefficients into a single array
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Determine the threshold value based on the compression ratio
    threshold = np.percentile(np.abs(coeff_arr), 100 - compression_ratio * 100)
    
    # Zero out coefficients that are below the threshold (compress the matrix)
    coeff_arr[abs(coeff_arr) < threshold] = 0
    
    # Reconstruct the coefficients array into wavelet coefficients format
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    
    # Reconstruct the compressed matrix from the wavelet coefficients
    compressed_matrix = pywt.waverec2(coeffs, wavelet)
    
    # Return the compressed matrix, ensuring its shape matches the original
    return compressed_matrix[:matrix.shape[0], :matrix.shape[1]]


def find_compression_ratio_wavelet(data, wavelet, error_threshold, step_size=0.005):
    """
    Finds the optimal compression ratio using wavelet transformation that satisfies the given error threshold.
    
    Args:
        data (np.array): The input data matrix.
        wavelet (str): The wavelet to use for compression (e.g., 'db1', 'haar').
        error_threshold (float): The maximum allowable error between the original and compressed matrix.
        step_size (float): The step size to increment the compression ratio, for finer searching.
    
    Returns:
        float: The optimal compression ratio that satisfies the error threshold.
        None: If no ratio is found that meets the error threshold.
    """
    # Generate a range of compression ratios from 0 to 1, inclusive, with the specified step size
    ratios = np.arange(0, 1 + step_size, step_size)  # Ensure 1 is included in the range
    best_ratio = None  # Variable to hold the best compression ratio
    
    # Iterate over the range of compression ratios
    for ratio in ratios:
        # Compress the matrix with the current ratio
        compressed_matrix = wavelet_transform_compression(data, wavelet, ratio)
        
        # Calculate the maximum absolute error between the original and compressed matrix
        error = np.max(np.abs(data - compressed_matrix))

        # If the error is within the threshold, update the best ratio and stop searching
        if error < error_threshold:
            best_ratio = ratio
            break

    return best_ratio


def dct_transform_compression(matrix, compression_ratio):
    """
    Compresses a matrix using the 2D Discrete Cosine Transform (DCT) based on a specified compression ratio.
    
    Args:
        matrix (np.array): The input matrix to be compressed.
        compression_ratio (float): The compression ratio, where 1 means no compression and 0 means maximum compression.
    
    Returns:
        np.array: The compressed matrix after applying DCT and thresholding.
    """
    # Perform 2D DCT (Discrete Cosine Transform) on the input matrix
    dct_matrix = scipy.fftpack.dct(scipy.fftpack.dct(matrix.T, norm='ortho').T, norm='ortho')
    
    # Flatten the DCT coefficients to apply thresholding
    dct_flat = dct_matrix.flatten()
    
    # Determine the threshold based on the compression ratio
    threshold = np.percentile(np.abs(dct_flat), 100 - compression_ratio * 100)
    
    # Zero out coefficients below the threshold to compress the matrix
    dct_flat[abs(dct_flat) < threshold] = 0
    
    # Reshape the flattened array back to the original DCT matrix shape
    dct_matrix = dct_flat.reshape(dct_matrix.shape)
    
    # Perform inverse DCT to reconstruct the compressed matrix
    compressed_matrix = scipy.fftpack.idct(scipy.fftpack.idct(dct_matrix.T, norm='ortho').T, norm='ortho')
    
    # Return the compressed matrix, ensuring the shape matches the original matrix
    return compressed_matrix[:matrix.shape[0], :matrix.shape[1]]


def find_compression_ratio_dct(matrix, error_threshold, step_size=0.005):
    """
    Finds the optimal compression ratio using DCT transformation that satisfies the given error threshold.
    
    Args:
        matrix (np.array): The input matrix to be compressed.
        error_threshold (float): The maximum allowable error between the original and compressed matrix.
        step_size (float): The step size to increment the compression ratio, for finer searching.
    
    Returns:
        float: The optimal compression ratio that satisfies the error threshold.
        None: If no ratio is found that meets the error threshold.
    """
    # Generate a range of compression ratios from 0 to 1, inclusive, with the specified step size
    ratios = np.arange(0, 1 + step_size, step_size)  # Ensure it includes 1
    best_ratio = None  # Variable to hold the best compression ratio
    
    # Iterate over each ratio and check the error after compression
    for ratio in ratios:
        # Compress the matrix using the current ratio
        compressed_matrix = dct_transform_compression(matrix, ratio)
        
        # Calculate the maximum absolute error between the original and compressed matrix
        error = np.max(np.abs(matrix - compressed_matrix))

        # If the error is below the threshold, update the best ratio and stop searching
        if error < error_threshold:
            best_ratio = ratio
            break

    return best_ratio

def compression_amount_with_pca(matrix, threshold):
    """
    Finds the optimal number of principal components (PCA) to compress the matrix 
    while maintaining the error below the specified threshold. 

    Args:
        matrix (np.array): The input data matrix to be compressed.
        threshold (float): The maximum allowable error between the original and reconstructed matrix.
    
    Returns:
        float: The compression ratio based on the number of selected components.
    """
    
    matrix = matrix.T
    # Standardize the data (optional but common practice for PCA)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(matrix)
    
    # Define the search bounds for binary search (1 to total number of rows/components)
    low = 1
    high = matrix.shape[0]  # The maximum number of components equals the number of rows (data points)
    optimal_components = high  # Initialize with maximum components as default

    # Perform a binary search to find the optimal number of components
    while low <= high:
        mid = (low + high) // 2  # Check the midpoint
        print(mid)
        
        # Apply PCA with 'mid' number of components
        pca = PCA(n_components=mid)
        pca.fit(standardized_data)
        
        # Transform the data using PCA and then reconstruct it back
        transformed_data = pca.transform(standardized_data)
        reconstructed = scaler.inverse_transform(pca.inverse_transform(transformed_data))
        
        # Calculate the maximum absolute error between original and reconstructed matrix
        max_abs_error = np.max(np.abs(matrix - reconstructed))
        
        # If the error is within the threshold, search in the lower half (fewer components)
        if max_abs_error <= threshold:
            optimal_components = mid  # Record the current best number of components
            high = mid - 1  # Narrow search to the lower half
        else:
            low = mid + 1  # Narrow search to the upper half (more components)

    # Calculate the compression ratio (components used / total possible components)
    compression_ratio = optimal_components / matrix.shape[0]
    
    return compression_ratio