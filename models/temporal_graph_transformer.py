import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
import argparse
import math


def setup_logging():
    """
    Configures and returns a custom logger with file and console handlers.

    This function creates a log directory if it doesn't exist, sets up logging to both file and console with a uniform format, and attaches these handlers to a custom logger. It ensures each session's logs are uniquely named by appending a timestamp to the log file's name.

    Returns:
        logging.Logger: A custom logger with file and console handlers.
    """
    # Specify the directory where log files will be stored
    log_directory = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\runs\\smart meter 9digit"
    # Ensure the directory exists; create it if it does not
    os.makedirs(log_directory, exist_ok=True)

    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the full path for the new log file
    log_file_path = os.path.join(log_directory, f'training_log_graph_transformer_{timestamp}.log')

    # Initialize a custom logger
    logger = logging.getLogger('CustomLogger')
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Set up logging to a file
    file_handler = logging.FileHandler(log_file_path)
    # Set up logging to the console
    console_handler = logging.StreamHandler()

    # Define a uniform format for logs
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    # Attach the file and console handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Return the configured logger
    return logger


class TemporalGraphTransformer(torch.nn.Module):
    """
    Combines GCNs for spatial dependency modeling and transformers for temporal dependency modeling in graph data.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        output_dim (int): Dimensionality of the output.
        num_layers (int): Number of layers for both GCN and transformer encoder.
        num_time_steps (int): Number of time steps for positional encoding.
        num_heads (int): Number of heads in the transformer.
        dropout (float): Dropout rate in the transformer.

    Attributes:
        hidden_dim (int): Hidden dimension size used across all layers.
        gnn_layers (ModuleList): List of GCN layers.
        transformer_encoder (TransformerEncoder): Transformer encoder for temporal modeling.
        out_proj (Linear): Output projection layer.
    """
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads=8, dropout=0.1):
        super(TemporalGraphTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GNN for Spatial Dependency Modeling
        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GCNConv(input_dim if _ == 0 else hidden_dim, hidden_dim))
        
        # Transformer Encoder for Temporal Dependency Modeling
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer for each node
        self.out_proj = Linear(hidden_dim, output_dim)

        
    def generate_positional_encoding(self, time_indices, hidden_dim):
        """
        Generates sinusoidal positional encoding for time indices.
        
        Args:
            time_indices (torch.Tensor): Time indices for encoding.
            hidden_dim (int): Dimensionality for encoding generation.
        
        Returns:
            torch.Tensor: Sinusoidal positional encoding.
        """
        # Ensure position and div_term are on the same device as time_indices
        device = time_indices.device  # Get the device from time_indices
    
        # Prepare div_term for broadcasting
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim)).to(device)
    
        # Generate sinusoidal positional encodings
        pe_sin = torch.sin(time_indices[:, None] * div_term)
        pe_cos = torch.cos(time_indices[:, None] * div_term)
    
        # Combine sin and cos into one tensor
        pe = torch.zeros((time_indices.size(0), hidden_dim), device=device)
        pe[:, 0::2] = pe_sin
        pe[:, 1::2] = pe_cos
        return pe

    
    def forward(self, x, edge_index, batch, time_idx):
        """
        Processes input through the network and predicts the next time step's features.

        Args:
            x (torch.Tensor): Node features for all graphs and all time steps.
            edge_index (torch.LongTensor): Graph connectivity in COO format.
            batch (torch.Tensor): Batch vector, which assigns each node to a graph.
            time_idx (torch.Tensor): Current time index for positional encoding.

        Returns:
            torch.Tensor: Predicted features for the next time step.
        """
        # x shape: (batch_size * num_nodes, input_dim)
        batch_size = batch.max().item() + 1  # Determine batch size from batch tensor
        num_features = x.shape[1]  # num_features is input_dim
        num_nodes = x.shape[0] // batch_size  # Determine the number of nodes per graph
    
        x_last = x.view(batch_size, num_nodes, num_features)[:, :, -1]  # (batch_size, num_nodes, num_features)
        
        # Spatial Dependency Modeling (GNN)
        for gnn in self.gnn_layers:
            x = F.relu(gnn(x.view(batch_size * num_nodes, -1), edge_index), inplace=False)
        
        x = x.view(batch_size, num_nodes, -1)  # Reshape back to (batch_size, num_nodes, hidden_dim)
        
        # Generate sinusoidal positional encoding for the current time step across all nodes
        time_encoded_features = self.generate_positional_encoding(time_idx, self.hidden_dim).to(x.device)
        time_encoded_features = time_encoded_features.unsqueeze(1).expand(-1, num_nodes, -1)   # Expand to all nodes in all graphs
        
        # Add positional encoding to node features
        x = x + time_encoded_features

        # Temporal Dependency Modeling (Transformer)
        x = x.permute(1, 0, 2)  # (num_nodes, batch_size, hidden_dim) for transformer input
        x = self.transformer_encoder(x)  # Transformer encoder expects (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)  # Permute back to (batch_size, num_nodes, hidden_dim)

        # Output Projection for each node
        x_res = self.out_proj(x.reshape(-1, self.hidden_dim))  # Apply linear layer to each node's features
        x_res = x_res.view(batch_size, num_nodes, -1)  # Reshape to (batch_size, num_nodes, output_dim)

        
        # Add last time step features and the residual prediction
        x_pred = x_last.unsqueeze(2) + x_res[:, :, -1].unsqueeze(2)  # Predicted next time step is the last time step plus the residual

        return x_pred


def load_data(path, batch_size):
    """
    Loads a dataset from a specified path and returns a DataLoader for batch processing.
    
    Args:
        path (str): File path to load the dataset from.
        batch_size (int): Number of samples per batch.
    
    Returns:
        DataLoader: DataLoader object for iterating over the dataset.
    """
    dataset = torch.load(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def train(model, loader, optimizer, criterion, device):
    """
    Trains the model using the provided data loader and optimizer.

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): DataLoader containing the dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion (callable): Loss function to measure the error.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Average loss and average mean absolute error (MAE) per batch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    total_mae_loss = 0
    mae_criterion = torch.nn.L1Loss()  # Initialize the MAE loss function

    for data in loader:
        data = data.to(device)  # Move data to the specified device
        optimizer.zero_grad()  # Clear gradients before each step
        output = model(data.x, data.edge_index, data.batch, data.time_idx)  # Forward pass
        loss = criterion(output.squeeze(-1).view(-1), data.y)  # Compute loss
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update weights
        mae_loss = mae_criterion(output.squeeze(-1).view(-1), data.y)  # Calculate MAE loss
        total_mae_loss += mae_loss.item()  # Aggregate MAE loss
        total_loss += loss.item()  # Aggregate total loss

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses per batch


def validate(model, loader, criterion, device):
    """
    Validates the model using the provided data loader and loss function.

    Args:
        model (torch.nn.Module): The model to validate.
        loader (DataLoader): DataLoader containing the validation dataset.
        criterion (callable): Loss function to measure the error.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Average loss and average mean absolute error (MAE) per batch during validation.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_mae_loss = 0
    mae_criterion = torch.nn.L1Loss()  # Initialize the MAE loss function

    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            data = data.to(device)  # Move data to the specified device
            output = model(data.x, data.edge_index, data.batch, data.time_idx)  # Compute output
            loss = criterion(output.squeeze(-1).view(-1), data.y)  # Compute loss
            total_loss += loss.item()  # Aggregate total loss
            mae_loss = mae_criterion(output.squeeze(-1).view(-1), data.y)  # Calculate MAE loss
            total_mae_loss += mae_loss.item()  # Aggregate MAE loss

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses per batch


def inference(model, loader, device):
    """
    Performs inference with the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Actual labels and model predictions for each batch in the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    actuals = []  # List to store actual labels
    predictions = []  # List to store predictions

    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            data = data.to(device)  # Move data to the specified device
            output = model(data.x, data.edge_index, data.batch, data.time_idx)  # Compute output
            actuals.append(data.y.numpy())  # Store actual labels
            predictions.append(output.view(-1).numpy())  # Store predictions

    return actuals, predictions


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Calculate the total number of trainable parameters



def parse_args():
    parser = argparse.ArgumentParser(description='Train a Graph Transformer on time-series graph data.')
    parser.add_argument('--dataset', type=str, default='smart_meter_9digit', help='Path to the training data file')
    parser.add_argument('--train-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data\\smart_meter_data_9digit_train_seq4_delaunay_k10.pt', help='Path to the training data file')
    parser.add_argument('--val-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data\\smart_meter_data_9digit_val_seq4_delaunay_k10.pt', help='Path to the validation data file')
    parser.add_argument('--test-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\graph data\\smart_meter_data_9digit_test_seq4_delaunay_k10.pt', help='Path to the validation data file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension of the Transformer')
    parser.add_argument('--output-dim', type=int, default=1, help='Output dimension or number of classes')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in the Graph Transformer')
    parser.add_argument('--num-heads', type=int, default=2, help='Number of attention heads in the Transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the Transformer')
    return parser.parse_args()


def main():
    # Initialize logging
    logger = setup_logging()

    # Parse command-line arguments
    args = parse_args()

    # Set up the computational device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
    train_loader = load_data(args.train_path, args.batch_size)
    val_loader = load_data(args.val_path, args.batch_size)
    
    # Initialize the model with specified parameters
    model = TemporalGraphTransformer(input_dim=6,
                                     hidden_dim=args.hidden_dim,
                                     output_dim=args.output_dim,
                                     num_layers=args.num_layers,
                                     num_heads=args.num_heads,
                                     dropout=args.dropout).to(device)

    # Count the total number of parameters
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()  # Mean Squared Error loss

    # Training and validation loop
    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        train_loss, train_mae_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train MAE Loss: {train_mae_loss:.6f}, Val MAE Loss: {val_mae_loss:.6f}')

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Save and load the model
    model_path = f'models\\{args.dataset}_graph_transformer_model_final.pth'
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Inference
    actuals, predictions = inference(model, test_loader, device)

    # Post-process and plot results
    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)
    print('Test loss MSE:', np.mean((actuals_array - predictions_array) ** 2))
    print('Test loss MAE:', np.mean(np.abs(actuals_array - predictions_array)))
    np.save(f'actuals_graph_transformer_{args.dataset}_test', actuals_array)
    np.save(f'predictions_graph_transformer_{args.dataset}_test', predictions_array)

    # Visualize actual vs predicted values
    columns = np.random.choice(predictions_array.shape[1], 1, replace=False)
    plt.figure(figsize=(10, 5))
    plt.plot(actuals_array[:, columns], label='Actual', alpha=0.7)
    plt.plot(predictions_array[:, columns], label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted of {columns}')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
