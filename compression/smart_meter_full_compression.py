import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from ..models import GCN, CNN, LSTM, MLP, TemporalGraphTransformer
from utils import *



def compress(test_loader, model, device, num_time_steps, threshold, model_type):
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
    
    # Initialize arrays for storing actuals, results, and predictions
    actuals = np.zeros((0, 21672))
    results = np.zeros((0, 21672))
    predicted = np.zeros((0, 21672))
    
    # Get the first batch of data from the test loader
    first_batch = next(iter(test_loader))
    
    # Handling different model types to set up the required inputs
    if model_type == 'gcn' or model_type == 'graph_transformer':
        edge_index = first_batch.edge_index.to(device)
        node_features = first_batch.x[:, :2]  # Use the first two features as node features
        batch = torch.zeros(node_features.size(0), dtype=torch.long).to(device)  # Batch size of 1
    
    # Initialize results and actuals with the first points
    results, actuals = make_first_points(results, actuals, test_loader)
    
    failed = 0  # Track the number of failed predictions
    i = 0  # Iteration counter
    
    # Iterate over the test data
    for data in test_loader:
        if i % 1000 == 0:
            print(i)  # Print progress every 1000 iterations
        
        # Prepare data for prediction (last 'num_time_steps' points)
        data_to_predict = results[-num_time_steps:, :]
        data_to_predict = torch.tensor(data_to_predict).T
        
        # Different handling based on model type
        if model_type == 'graph_transformer':
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch, data.time_idx)
        elif model_type == 'gcn':
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch)
        elif model_type == 'cnn':
            x = torch.tensor(data_to_predict).reshape(1, num_time_steps, 21672).to(device).float()
            predicted_point = model(x)
        elif model_type == 'lstm' or model_type == 'mlp':
            x = torch.tensor(data_to_predict).to(device).float()
            predicted_point = model(x)
        
        # Store the predicted values and append to results and actuals
        predicted = np.append(predicted, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        results = np.append(results, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        actuals = np.append(actuals, data.y.detach().cpu().numpy().reshape(1, -1), axis=0)


        # Find the indices where the error exceeds the threshold
        error_idx = (np.abs(actuals[-1, :] - predicted[-1, :]) > threshold)
       
        # Refine the predictions based on the error threshold
        time_step_results = np.zeros((21672,))
        time_step_results[error_idx == True] = actuals[-1, :][error_idx == True]  # Use actual values for large errors
        time_step_results[error_idx == False] = predicted[-1, :][error_idx == False]  # Use predicted values for small errors
        
        # Update the last row of results with the refined values
        results[-1, :] = time_step_results.reshape((1, 21672))

        # Increment the number of failed predictions (errors exceeding the threshold)
        failed += np.sum(error_idx)

        i += 1  # Increment the iteration counter

    # Return the total number of failed predictions, refined results, and actual values
    return failed, results, actuals
        
# =============================================================================
#  ↓↓↓ Main ↓↓↓
# =============================================================================
model_types = ['graph_transformer', 'gcn', 'cnn', 'lstm', 'mlp']

for model_type in model_types:

    #test_data_address = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit\\cumulative_data_9digit_master.csv'
    #data = pd.read_csv(test_data_address)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model and optimizer
    model_address = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\models'
    
    if model_type == 'graph_transformer':
        #Loading GCN Model
        input_dim = 6
        hidden_dim = 64
        output_dim = 1
        num_layers = 3
        num_heads = 8
        dropout = 0.1
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'smart_meter_9digit_graph_transformer_model_final.pth'))
        
        model = TemporalGraphTransformer(input_dim=input_dim, 
                                         hidden_dim=hidden_dim, 
                                         output_dim=output_dim, 
                                         num_layers=num_layers,
                                         num_time_steps=35034, #105115 for nrel, 35034 for smart meter with seq4, 104991 for caltrans with seq4
                                         num_heads=num_heads,
                                         dropout=dropout)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data\\smart_meter_data_9digit_train_seq4_delaunay_k10.pt"
    
            
    elif model_type == 'gcn':
        #Loading GCN Model
        input_dim = 6
        hidden_dim = 64
        output_dim = 1
        num_layers = 3
        num_time_steps = input_dim - 2
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'smart_meter_9digit_gcn_model_final_mre.pth'))
        
        model = GCN(input_dim=input_dim , 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data\\smart_meter_data_9digit_train_seq4_delaunay_k10.pt"
    
    
    elif model_type == 'cnn':
        #Loading GCN Model
        sequence_length = 4
        filters = [16, 32]
        kernel_sizes = [3, 3]
        pool_sizes = [2, 2]
        num_nodes = 21672
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'smart_meter_9digit_cnn_model_final.pth'))
        
        model = CNN(
                    num_nodes=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    sequence_length=sequence_length,
                    input_channels=1,  # Each time step as a separate channel
                    num_filters=filters,  # Two layers with 16 and 32 filters
                    kernel_sizes=kernel_sizes,  # 3x3 kernels for both layers
                    pool_sizes=pool_sizes,  # Pooling with window size of 2 for both layers
                    output_dim=num_nodes)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\cnn data\\smart_meter_data_9digit_train_seq4.pt"
    
    
    elif model_type == 'lstm':
        #Loading GCN Model
        num_nodes = 21672
        hidden_dim = 64
        num_layers = 3
        num_epoch = 10
        lr = 0.001
        
        checkpoint = torch.load(os.path.join(model_address, 'smart_meter_9digit_lstm_model_final.pth'))
        
        model = LSTM(input_dim=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    hidden_dim=hidden_dim, 
                    output_dim=num_nodes, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\lstm data\\smart_meter_data_9digit_train_seq4.pt"
    
    elif model_type == 'mlp':
        #Loading GCN Model
        input_dim = 4
        hidden_dim = 16
        output_dim = 1
        num_layers = 2
        num_epoch = 10
        lr = 0.0001
        
        checkpoint = torch.load(os.path.join(model_address, 'smart_meter_9digit_mlp_model_final.pth'))
        
        model = MLP(input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\lstm data\\smart_meter_data_9digit_train_seq4.pt"
    
    # Move model to the device (GPU or CPU)
model.to(device)

# Define the number of time steps for prediction
num_time_steps = 4

# Count and print the total number of parameters in the model
total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')

# Load test data with two different batch sizes
test_loader = load_data(test_set_path, 1)  # Batch size 1 for regular prediction

# Initialize list to store the final results
final_results = []

# Randomly choose indices to visualize predictions later
rand_indices = np.random.choice(range(0, 21672), size=1, replace=False)

# Loop through different error thresholds for refinement
for threshold in [0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    
    # Refine and compress the predictions for each threshold
    failed, results, actuals = compress(test_loader, model, device, num_time_steps, threshold, model_type)
        
    # Calculate the storage need (failed points + model parameters) relative to the total data
    storage_need = (failed + total_params) / (len(test_loader) * 21672)
    
    # Append the threshold and storage need to the final results list
    final_results.append([threshold, storage_need])

    
    # Plot actual vs predicted values for the chosen random indices
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:, rand_indices], label='Actual', alpha=1)
    plt.plot(results[:, rand_indices], label='Predicted', alpha=0.5)
    plt.title(f'Actual vs Results of {rand_indices}')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    #plt.show()  # Optionally show the plot
    
    # Save the actuals and results arrays for each threshold
    np.save(f'smart_meter_full_{model_type}_{threshold}_actuals', actuals)
    np.save(f'smart_meter_full_{model_type}_{threshold}_results', results)

# Save the final compression results across thresholds
np.save(f'smart_meter_full_{model_type}_compression_results', np.array(final_results))

# Write the final results to a text file
with open(f'smart_meter_full_{model_type}_results_train.txt', 'a') as f:
    for result in final_results:
        f.write(str(result))
        f.write('\n')

# Save actual values for future reference
np.save('smart_meter_full_actuals_train', actuals)

# Load the saved actuals data
actuals = np.load('smart_meter_full_actuals_train.npy')

# Initialize lists for storing compression results
zfp_compressions = []
wavelet_compressions = []
dct_compressions = []
pca_compressions = []

# Loop through thresholds to compute compression ratios using various methods
for threshold in [0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # Uncomment to compute ZFP compression ratios
    # zfp_compression = find_compression_ratio_zfp(actuals, threshold)
    # zfp_compressions.append([threshold, zfp_compression])
    
    # Compute wavelet compression ratio for each threshold
    wavelet_compression = find_compression_ratio_wavelet(actuals, 'db1', threshold)
    wavelet_compressions.append([threshold, wavelet_compression])
    
    # Compute DCT compression ratio for each threshold
    dct_compression = find_compression_ratio_dct(actuals, threshold)
    dct_compressions.append([threshold, dct_compression])
    
    # Compute PCA compression ratio for each threshold
    pca_compression = compression_amount_with_pca(actuals.T, threshold)
    pca_compressions.append([threshold, pca_compression])
    
# Print and save zfp compression results
#print(zfp_compressions)
#np.save('smart_meter_full_zfp_compression_results', np.array(zfp_compressions))
#with open('smart_meter_full_zfp_results_f.txt', 'a') as f:
#    for result in zfp_compressions:
#        f.write(str(result))
#        f.write('\n')

# Print and save wavelet compression results
print(wavelet_compressions)        
np.save('smart_meter_full_wavelet_compression_results', np.array(wavelet_compressions))
with open('smart_meter_full_wavelet_results_train.txt', 'a') as f:
    for result in wavelet_compressions:
        f.write(str(result))
        f.write('\n')

# Print and save DCT compression results
print(dct_compressions)         
np.save('smart_meter_full_dct_compression_results', np.array(dct_compressions))
with open('smart_meter_full_dct_results_train.txt', 'a') as f:
    for result in dct_compressions:
        f.write(str(result))
        f.write('\n')

# Print and save PCA compression results
print(pca_compressions)         
np.save('smart_meter_full_pca_compression_results', np.array(pca_compressions))
with open('smart_meter_full_pca_results_train.txt', 'a') as f:
    for result in pca_compressions:
        f.write(str(result))
        f.write('\n')
