import torch
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import argparse
import numpy as np

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

class LSTM(nn.Module):
    """
    LSTM (Long Short-Term Memory) model for time-series prediction.

    Args:
        input_dim (int): Dimensionality of input features per node.
        hidden_dim (int): Dimensionality of hidden states in the LSTM.
        output_dim (int): Dimensionality of the output features per node.
        num_layers (int): Number of layers in the LSTM.
        dropout_rate (float, optional): Dropout rate for the LSTM and output layer. Defaults to 0.5.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequence processing.
        dropout (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer to map LSTM outputs to the desired output dimension.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_nodes = input_dim  # Number of nodes is inferred from input dimension.
        
        # LSTM configuration
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes * batch_size, sequence_length, feature_size)

        Returns:
            torch.Tensor: Output tensor of predicted features, adjusted by the last input feature.
        """
        # Calculate batch size based on total number of nodes and the first dimension of x
        total_size = x.size(0)
        batch_size = total_size // self.num_nodes

        # Reshape input for processing by LSTM
        x = x.view(batch_size, -1, self.num_nodes)  # Rearrange dimensions for batch_first LSTM
        
        # Extract the last features for residual connection
        x_last = x[:, -1, :]

        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        # Apply dropout to the outputs of the last LSTM layer
        lstm_out = self.dropout(lstm_out[:, -1, :])

        # Pass the output through the linear layer to map to the desired output dimension
        x_res = self.linear(lstm_out)
        
        # Add the result to the last input feature for residual learning
        x_pred = x_last + x_res
        
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
        output = model(data.x)  # Forward pass
        loss = criterion(output.view(-1), data.y)  # Compute loss
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update weights
        mae_loss = mae_criterion(output.view(-1), data.y)  # Calculate MAE loss
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
            output = model(data.x)  # Compute output
            loss = criterion(output.view(-1), data.y)  # Compute loss
            mae_loss = mae_criterion(output.view(-1), data.y)  # Calculate MAE loss
            total_loss += loss.item()  # Aggregate total loss
            total_mae_loss += mae_loss.item()  # Aggregate MAE loss

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses per batch


def inference(model, loader, device):
    """
    Performs inference using the provided model and data loader.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Lists containing actual labels and model predictions for each batch in the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    actuals = []  # List to store actual labels
    predictions = []  # List to store predictions

    with torch.no_grad():  # Disable gradient computation for inference
        for data in loader:
            data = data.to(device)  # Move data to the specified device
            output = model(data.x)  # Compute the model output
            actuals.append(data.y.view(-1).numpy())  # Convert and store the actual labels
            predictions.append(output.view(-1).numpy())  # Convert and store the predictions

    return actuals, predictions


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model for which to count parameters.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Calculate the total number of trainable parameters

def parse_args():
    parser = argparse.ArgumentParser(description='Train a LSTM model on time-series graph data.')
    parser.add_argument('--dataset', type=str, default='nrel_full', help='Path to the training data file')
    parser.add_argument('--train-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data\\full_data_test_seq4.pt', help='Path to the training data file')
    parser.add_argument('--val-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data\\full_data_test_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--test-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\lstm data\\full_data_test_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Dimensionality of hidden layers in the LSTM')
    parser.add_argument('--output-dim', type=int, default=5166, help='Dimensionality of the output layer')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of LSTM layers')
    
    return parser.parse_args()

def main():
    # Initialize logging
    logger = setup_logging()
    
    # Parse command-line arguments
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_loader = load_data(args.train_path, args.batch_size)
    val_loader = load_data(args.val_path, args.batch_size)
    test_loader = DataLoader(torch.load(args.test_path), batch_size=1, shuffle=False)

    # Model setup
    model = LSTM(
        input_dim=args.output_dim,  # Dynamically configure based on the dataset: 21672 for smart meter data 9 digit, 5166 for full nrel data, 697 for caltrans speed
        hidden_dim=args.hidden_dim, 
        output_dim=args.output_dim, 
        num_layers=args.num_layers
    ).to(device)
    
    # Display the total number of parameters in the model
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # Lists to track losses
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_mae_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train MAE Loss: {train_mae_loss:.6f}, Val MAE Loss: {val_mae_loss:.6f}')

    # Visualize training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Save and load the trained model
    model_path = f'models/{args.dataset}_lstm_model_final.pth'
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Perform inference
    actuals, predictions = inference(model, test_loader, device)
    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)

    # Output test loss metrics
    print('Test loss MSE:', np.mean((actuals_array - predictions_array)**2))
    print('Test loss MAE:', np.mean(np.abs(actuals_array - predictions_array)))

    # Save predictions and actuals for analysis
    np.save(f'actuals_lstm_{args.dataset}_test', actuals_array)
    np.save(f'predictions_lstm_{args.dataset}_test', predictions_array)

    # Plot actual vs predicted values
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
