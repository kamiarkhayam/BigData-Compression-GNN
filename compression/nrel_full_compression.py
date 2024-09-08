import numpy as np

from matplotlib import pyplot as plt
import os
import torch
from ..models import GCN, CNN, LSTM, MLP, TemporalGraphTransformer
from utils import *



def compress(test_loader, model, device, num_epoch, num_time_steps, threshold, model_type):
    """
    Perform refinement and compression on the model's output by adjusting predictions based on error thresholds.
    
    Args:
        test_loader (DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The trained model to use for predictions.
        device (torch.device): Device on which the model and data are processed.
        num_time_steps (int): Number of time steps to consider for the model input.
        threshold (float): Threshold for refining the predictions.
        model_type (str): The type of model (e.g., 'gcn', 'cnn', 'lstm', etc.).
    
    Returns:
        int: The number of failed predictions that exceed the threshold.
        np.array: The refined results after threshold-based corrections.
        np.array: The actual values used for comparison.
    """   
    # Initialize empty arrays to store results, actuals, and predictions
    actuals = np.zeros((0, 5166))  # Store actual values
    results = np.zeros((0, 5166))  # Store the refined results
    predicted = np.zeros((0, 5166))  # Store predicted values
    
    # Get the first batch of data
    first_batch = next(iter(test_loader))
    
    # Handle different model types and set up edge_index, node features, and batch if using GNN models
    if model_type == 'gcn' or model_type == 'graph_transformer':
        edge_index = first_batch.edge_index.to(device)  # Move edge_index to the device (GPU or CPU)
        node_features = first_batch.x[:, :4]  # Use the first four features as node features
        batch = torch.zeros(node_features.size(0), dtype=torch.long).to(device)  # Batch initialization (assuming batch size 1)
       
    # Initialize results and actuals with the first points from the test_loader
    results, actuals = make_first_points(results, actuals, test_loader)
    
    failed = 0  # Track the number of failed predictions
    i = 0  # Loop counter for debugging and logging
    
    # Iterate through the test loader
    for data in test_loader:
        if i % 1000 == 0:  # Print progress every 1000 iterations
            print(i)
        
        # Prepare data for prediction by selecting the last `num_time_steps` from results
        data_to_predict = results[-num_time_steps:, :]
        data_to_predict = torch.tensor(data_to_predict).T  # Transpose for correct shape
        
        # Handle different model types to perform predictions
        if model_type == 'graph_transformer':
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch, data.time_idx)
        
        elif model_type == 'gcn':
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch)
            
        elif model_type == 'cnn':
            x = torch.tensor(data_to_predict).reshape(1, num_time_steps, 5166).to(device).float()
            predicted_point = model(x)
        
        elif model_type == 'lstm' or model_type == 'mlp':
            x = torch.tensor(data_to_predict).to(device).float()
            predicted_point = model(x)
        
        # Store predicted values
        predicted = np.append(predicted, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        results = np.append(results, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        actuals = np.append(actuals, data.y.detach().cpu().numpy().reshape(1, -1), axis=0)

        # Compute error indices based on the threshold for each element
        error_idx = (np.abs(actuals[-1, :] - predicted[-1, :]) > threshold)
        
        # Update results for the current time step based on error threshold
        time_step_results = np.zeros((5166,))
        time_step_results[error_idx == True] = actuals[-1, :][error_idx == True]
        time_step_results[error_idx == False] = predicted[-1, :][error_idx == False]
        
        results[-1, :] = time_step_results.reshape((1, 5166))

        # Update the number of failed predictions
        failed += np.sum(error_idx)

        i += 1  # Increment loop counter

    return failed, results, actuals  # Return the number of failed predictions, refined results, and actual values
        
# =============================================================================
#  ↓↓↓ Main ↓↓↓
# =============================================================================
model_types = ['graph_transformer', 'gcn', 'cnn', 'lstm', 'mlp']

for model_type in model_types:

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model and optimizer
    model_address = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\models'
    
    if model_type == 'graph_transformer':
        #Loading GCN Model
        input_dim = 8
        hidden_dim = 64
        output_dim = 1
        num_layers = 3
        num_heads = 8
        dropout = 0.1
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'nrel_full_c_graph_transformer_model_final.pth'))
        
        model = TemporalGraphTransformer(input_dim=input_dim, 
                                         hidden_dim=hidden_dim, 
                                         output_dim=output_dim, 
                                         num_layers=num_layers,
                                         num_time_steps=105115, #105115 for nrel, 35034 for smart meter with seq4, 104991 for caltrans with seq4
                                         num_heads=num_heads,
                                         dropout=dropout)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\graph data\\full_data_test_seq4_delaunay_k10_c.pt"
    
        
    elif model_type == 'gcn':
        #Loading GCN Model
        input_dim = 8
        hidden_dim = 64
        output_dim = 1
        num_layers = 3
        num_time_steps = input_dim - 4
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'nrel_full_c_gcn_model_final_mre.pth'))
        
        model = GCN(input_dim=input_dim , 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\graph data\\full_data_test_seq4_delaunay_k10_c.pt"
    
    
    elif model_type == 'cnn':
        #Loading GCN Model
        sequence_length = 4
        filters = [16, 32]
        kernel_sizes = [3, 3]
        pool_sizes = [2, 2]
        num_nodes = 5166
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'nrel_full_c_cnn_model_final.pth'))
        
        model = CNN(
                    num_nodes=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    sequence_length=sequence_length,
                    input_channels=1,  # Each time step as a separate channel
                    num_filters=filters,  # Two layers with 16 and 32 filters
                    kernel_sizes=kernel_sizes,  # 3x3 kernels for both layers
                    pool_sizes=pool_sizes,  # Pooling with window size of 2 for both layers
                    output_dim=num_nodes)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\cnn data\\full_data_test_seq4_c.pt"
    
    
    elif model_type == 'lstm':
        #Loading GCN Model
        num_nodes = 5166
        hidden_dim = 64
        num_layers = 3
        num_epoch = 10
        lr = 0.001
        
        checkpoint = torch.load(os.path.join(model_address, 'nrel_full_c_lstm_model_final.pth'))
        
        model = LSTM(input_dim=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    hidden_dim=hidden_dim, 
                    output_dim=num_nodes, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data\\full_data_test_seq4_c.pt"
    
        
    elif model_type == 'mlp':
        #Loading GCN Model
        input_dim = 4
        hidden_dim = 16
        output_dim = 1
        num_layers = 2
        num_epoch = 10
        lr = 0.0001
        
        checkpoint = torch.load(os.path.join(model_address, 'nrel_full_c_mlp_model_final.pth'))
        
        model = MLP(input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data\\full_data_test_seq4_c.pt"
    
    
    # Move the model to the specified device (GPU or CPU)
model.to(device)

# Set the number of time steps to use in predictions
num_time_steps = 4


# Calculate the total number of parameters in the model and print it
total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')

# Load the test dataset with a batch size of 1
test_loader = load_data(test_set_path, 1)

# Initialize an empty list to store the final results
final_results = []

# Select random indices for plotting purposes
rand_indices = np.random.choice(range(0, 5166), size=1, replace=False)

# Iterate through a list of thresholds for refining and compressing the data
for threshold in [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]:
    
    # Call the refine_and_compress function to perform predictions and get the results
    failed, results, actuals = compress(test_loader, model, device, num_epoch, num_time_steps, threshold, model_type)
        
    # Calculate the storage requirement based on failed predictions and model size
    storage_need = (failed + total_params) / (len(test_loader) * 5166)
    
    # Append the threshold and storage requirement to final_results
    final_results.append([threshold, storage_need])
    
    # Plot the actual vs predicted results for a randomly selected index
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:, rand_indices], label='Actual', alpha=1)
    plt.plot(results[:, rand_indices], label='Predicted', alpha=0.5)
    plt.title(f'Actual vs Results of {rand_indices}')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    #plt.show()  # Commented out plot display
    
    # Save the actual and predicted results to .npy files
    np.save(f'NREL_full_c_{model_type}_{threshold}_actuals', actuals)
    np.save(f'NREL_full_c_{model_type}_{threshold}_results', results)

# Save the final compression results to a .npy file
np.save(f'NREL_full_c_{model_type}_compression_results', np.array(final_results))

# Write the final compression results to a text file
with open(f'nrel_full_c_{model_type}_results_abs.txt', 'a') as f:
    for result in final_results:
        f.write(str(result))
        f.write('\n')

# Save the actuals to a .npy file
np.save('nrel_full_c_actuals', actuals)

# Load the actuals for traditional compression method comparison
actuals = np.load('nrel_full_c_actuals.npy')

# Initialize empty lists to store compression results for different methods
zfp_compressions = []
wavelet_compressions = []
dct_compressions = []
pca_compressions = []

# Iterate through a list of thresholds to find compression ratios with traditional methods
for threshold in [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]:
    # Find compression ratio using ZFP (commented out)
    #zfp_compression = find_compression_ratio_zfp(actuals, threshold)
    #zfp_compressions.append([threshold, zfp_compression])
    
    # Find compression ratio using Wavelet transform
    wavelet_compression = find_compression_ratio_wavelet(actuals, 'db1', threshold)
    wavelet_compressions.append([threshold, wavelet_compression])
    
    # Find compression ratio using DCT transform
    dct_compression = find_compression_ratio_dct(actuals, threshold)
    dct_compressions.append([threshold, dct_compression])
    
    # Find compression ratio using PCA
    pca_compression = compression_amount_with_pca(actuals, threshold)
    pca_compressions.append([threshold, pca_compression])


# Save zfp compression results to a .npy file
#np.save('nrel_full_zfp_compression_results', np.array(zfp_compressions))
#with open('nrel_full_zfp_results_abs.txt', 'a') as f:
#    for result in zfp_compressions:
#        f.write(str(result))
#        f.write('\n')


# Save wavelet compression results to a .npy file
np.save('nrel_fulL_wavelet_compression_results', np.array(wavelet_compressions))
with open('nrel_full_wavelet_results_abs.txt', 'a') as f:
    for result in wavelet_compressions:
        f.write(str(result))
        f.write('\n')
        
# Save DCT compression results to a .npy file
np.save('nrel_full_dct_compression_results', np.array(dct_compressions))
with open('nrel_full_dct_results_abs.txt', 'a') as f:
    for result in dct_compressions:
        f.write(str(result))
        f.write('\n')

# Save PCA compression results to a .npy file
np.save('nrel_fulL_pca_compression_results', np.array(pca_compressions))
with open('nrel_fulL_pca_results_abs.txt', 'a') as f:
    for result in pca_compressions:
        f.write(str(result))
        f.write('\n')

   