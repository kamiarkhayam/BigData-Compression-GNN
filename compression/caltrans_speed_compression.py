import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from utils import *
from ..models import GCN, CNN, LSTM, MLP, TemporalGraphTransformer

np.random.seed(23) 



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
    # Initialize arrays to store actuals, predicted, and result data points
    actuals = np.zeros((0, 697))
    results = np.zeros((0, 697))
    predicted = np.zeros((0, 697))
    
    # Get the first batch from the test loader
    first_batch = next(iter(test_loader))
    
    # Handle GCN and Graph Transformer models (special case for their input)
    if model_type == 'gcn' or model_type == 'graph_transformer':
        edge_index = first_batch.edge_index.to(device)  # Get edge index for GNN
        node_features = first_batch.x[:, :2]  # Extract the first two features (e.g., coordinates)
        batch = torch.zeros(node_features.size(0), dtype=torch.long).to(device)  # Assume batch size of 1 for GNN

    # Append the first points to results and actuals
    results, actuals = make_first_points(results, actuals, test_loader)
    
    failed = 0  # Counter for failed predictions
    i = 0  # Initialize step counter

    # Loop over the test_loader to process each batch of data
    for data in test_loader:
        if i % 1000 == 0:
            print(i)  # Print progress every 1000 iterations
        
        # Prepare the data for prediction (last 'num_time_steps' results)
        data_to_predict = results[-num_time_steps:, :]
        data_to_predict = torch.tensor(data_to_predict).T  # Transpose for correct shape

        # Model-specific data prediction process
        if model_type == 'graph_transformer':
            # For Graph Transformer models
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch, data.time_idx)
        
        elif model_type == 'gcn':
            # For GCN models
            x = torch.cat((node_features, data_to_predict), dim=1).to(device).float()
            predicted_point = model(x, edge_index, batch)
            
        elif model_type == 'cnn':
            # For CNN models (reshape data accordingly)
            x = torch.tensor(data_to_predict).reshape(1, num_time_steps, 697).to(device).float()
            predicted_point = model(x)
        
        elif model_type == 'lstm' or model_type == 'mlp':
            # For LSTM or MLP models
            x = torch.tensor(data_to_predict).to(device).float()
            predicted_point = model(x)

        # Optionally, zero out small predicted values
        # predicted_point[predicted_point < 0.01] = 0

        # Record the predicted results by appending them
        predicted = np.append(predicted, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        results = np.append(results, predicted_point.detach().cpu().numpy().reshape(1, -1), axis=0)
        actuals = np.append(actuals, data.y.detach().cpu().numpy().reshape(1, -1), axis=0)

        # Calculate the error based on the threshold
        error_idx = (np.abs(actuals[-1, :] - predicted[-1, :]) > threshold)

        # Update results for the current time step based on the thresholded error
        time_step_results = np.zeros((697,))
        time_step_results[error_idx == True] = actuals[-1, :][error_idx == True]  # Use actual values where error is large
        time_step_results[error_idx == False] = predicted[-1, :][error_idx == False]  # Use predicted values where error is small
        
        # Update the results array for the current time step
        results[-1, :] = time_step_results.reshape((1, 697))

        # Increment the number of failed predictions
        failed += np.sum(error_idx)

        # Move to the next iteration
        i += 1

    # Return the number of failed predictions, final results, and actuals
    return failed, results, actuals
        
# =============================================================================
#  ↓↓↓ Main ↓↓↓
# =============================================================================
model_types = ['graph_transformer', 'gcn', 'cnn', 'lstm', 'mlp']

for model_type in model_types:

    #test_data_address = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cleaned_speed_data_full.csv'
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
        
        checkpoint = torch.load(os.path.join(model_address, 'caltrans_speed_graph_transformer_model_final.pth'))
        
        model = TemporalGraphTransformer(input_dim=input_dim, 
                                         hidden_dim=hidden_dim, 
                                         output_dim=output_dim, 
                                         num_layers=num_layers,
                                         num_time_steps=35034, #105115 for nrel for seq4, 35034 for smart meter with seq4, 104991 for caltrans with seq4
                                         num_heads=num_heads,
                                         dropout=dropout)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\graph data\\caltrans_speed_data_test_seq4_knn_k3.pt"
    
        
    elif model_type == 'gcn':
        #Loading GCN Model
        input_dim = 6
        hidden_dim = 64
        output_dim = 1
        num_layers = 3
        num_time_steps = input_dim - 2
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'caltrans_speed_gcn_model_final_mre.pth'))
        
        model = GCN(input_dim=input_dim , 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\graph data\\caltrans_speed_data_test_seq4_knn_k3.pt"
    
    
    elif model_type == 'cnn':
        #Loading GCN Model
        sequence_length = 4
        filters = [16, 32]
        kernel_sizes = [3, 3]
        pool_sizes = [2, 2]
        num_nodes = 697 #901 for flow
        lr = 0.00001
        num_epoch = 10
        
        checkpoint = torch.load(os.path.join(model_address, 'caltrans_speed_cnn_model_final.pth'))
        
        model = CNN(
                    num_nodes=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    sequence_length=sequence_length,
                    input_channels=1,  # Each time step as a separate channel
                    num_filters=filters,  # Two layers with 16 and 32 filters
                    kernel_sizes=kernel_sizes,  # 3x3 kernels for both layers
                    pool_sizes=pool_sizes,  # Pooling with window size of 2 for both layers
                    output_dim=num_nodes)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cnn data\\caltrans_speed_data_test_seq4.pt"
    
    
    elif model_type == 'lstm':
        #Loading GCN Model
        num_nodes = 697
        hidden_dim = 64
        num_layers = 3
        num_epoch = 10
        lr = 0.001
        
        checkpoint = torch.load(os.path.join(model_address, 'caltrans_speed_lstm_model_final.pth'))
        
        model = LSTM(input_dim=num_nodes, #405 for nrel, 349 for smart meter, 901 for caltrans, 21672 for smart meter data 9 digit
                    hidden_dim=hidden_dim, 
                    output_dim=num_nodes, 
                    num_layers=num_layers)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\lstm data\\caltrans_speed_data_test_seq4.pt"
    
        
    elif model_type == 'mlp':
        #Loading GCN Model
        input_dim = 4
        hidden_dim = 16
        output_dim = 1
        num_layers = 2
        num_epoch = 10
        lr = 0.0001
        
        checkpoint = torch.load(os.path.join(model_address, 'caltrans_speed_mlp_model_final.pth'))
        
        model = MLP(input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint)
        test_set_path = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\lstm data\\caltrans_speed_data_test_seq4.pt"
    
    
    model.to(device)  # Move the model to the specified device (CPU or GPU)
        
    num_time_steps = 4  # Number of time steps used in prediction
    
    # Calculate and print the total number of parameters in the model
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')
        
    # Load test data
    test_loader = load_data(test_set_path, 1)  # Load the test set with batch size 1 for prediction
        
    final_results = []  # Initialize list to store results
    rand_indices = np.random.choice(range(0, 697), size=1, replace=False)  # Randomly select an index for plotting
    
    # Iterate over different error thresholds
    for threshold in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]:
            
        # Refine and compress the data based on the threshold and model type
        failed, results, actuals = compress(test_loader, model, device, num_time_steps, threshold, model_type)
            
        # Calculate the storage need based on the number of failed predictions and total parameters
        storage_need = (failed + total_params) / (len(test_loader) * 697)
        
        # Store the threshold and corresponding storage need in the final results
        final_results.append([threshold, storage_need])
        
        # Plot actual vs predicted results for the selected random index
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[:, rand_indices], label='Actual', alpha=0.7)
        plt.plot(results[:, rand_indices], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Results of {rand_indices}')
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.legend()
        #plt.show()  # Plotting is currently disabled, can be enabled by removing the comment
        
        # Print the current final results after each threshold iteration
        print(final_results)
        
        # Save the actual and predicted results for the current threshold
        np.save(f'caltrans_speed_{model_type}_{threshold}_actuals', actuals)
        np.save(f'caltrans_speed_{model_type}_{threshold}_results', results)
    
    # Save the final compression results
    np.save(f'caltrans_speed_{model_type}_compression_results', np.array(final_results))
    
    # Save the final results to a text file
    with open(f'caltrans_speed_{model_type}_results_rel.txt', 'a') as f:
        for result in final_results:
            f.write(str(result))
            f.write('\n')

# Load actuals for compression analysis
actuals = np.load('caltrans_speed_actuals.npy')

# Lists to store compression results for different methods
zfp_compressions = []
wavelet_compressions = []
dct_compressions = []
pca_compressions = []

# Iterate over the same thresholds to find compression ratios for different methods
for threshold in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]:
    # Uncomment to find ZFP compression ratio
    #threshold *= 0.811  # Apply scaling factor if needed
    #zfp_compression = find_compression_ratio_zfp(actuals, threshold)
    #zfp_compressions.append([threshold, zfp_compression])
    
    # Find wavelet compression ratio and store the result
    wavelet_compression = find_compression_ratio_wavelet(actuals, 'db1', threshold)
    wavelet_compressions.append([threshold, wavelet_compression])
    
    # Find DCT compression ratio and store the result
    dct_compression = find_compression_ratio_dct(actuals, threshold)
    dct_compressions.append([threshold, dct_compression])
    
    # Find PCA compression ratio and store the result
    pca_compression = compression_amount_with_pca(actuals, threshold)
    pca_compressions.append([threshold, pca_compression])

# Uncomment to save ZFP compression results
#np.save('caltrans_speed_zfp_compression_results', np.array(zfp_compressions))
#with open('caltrans_speed_zfp_results_rel.txt', 'a') as f:
#    for result in zfp_compressions:
#        f.write(str(result))
#        f.write('\n')

# Save wavelet compression results
print(wavelet_compressions)
np.save('caltrans_speed_wavelet_compression_results', np.array(wavelet_compressions))
with open('caltrans_speed_wavelet_results_abs.txt', 'a') as f:
    for result in wavelet_compressions:
        f.write(str(result))
        f.write('\n')

# Save DCT compression results
print(dct_compressions)
np.save('caltrans_speed_dct_compression_results', np.array(dct_compressions))
with open('caltrans_speed_dct_results_abs.txt', 'a') as f:
    for result in dct_compressions:
        f.write(str(result))
        f.write('\n')

# Save PCA compression results
print(pca_compressions)
np.save('caltrans_speed_pca_compression_results', np.array(pca_compressions))
with open('caltrans_speed_pca_results_abs.txt', 'a') as f:
    for result in pca_compressions:
        f.write(str(result))
        f.write('\n')


