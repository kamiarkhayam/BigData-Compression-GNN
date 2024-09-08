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


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for processing sequential or structured data.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        output_dim (int): Dimensionality of the output layer.
        num_layers (int): Number of hidden layers in the MLP.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.

    Attributes:
        layers (nn.Sequential): A sequential container of all layers in the MLP.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0):
        super(MLP, self).__init__()
        
        # Layer list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) with a residual connection from the last input feature.
        """
        x_last = x[:, -1].view(-1, 1)  # Extract and reshape the last input feature for residual connection
        output = self.layers(x)  # Pass input through the MLP layers
        return output + x_last  # Add the last input feature to the output for residual learning


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
    Trains the model for one epoch over the provided dataset.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        loader (DataLoader): DataLoader providing the dataset.
        optimizer (Optimizer): The optimization algorithm (e.g., Adam, SGD).
        criterion (Loss): The loss function used for training (e.g., MSE, CrossEntropy).
        device (torch.device): The device tensors will be transferred to (CPU or GPU).

    Returns:
        tuple: Average training loss and mean absolute error loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    total_mae_loss = 0
    mae_criterion = torch.nn.L1Loss()  # Additional criterion for MAE calculation

    for data in loader:
        data = data.to(device)  # Transfer data to the specified device
        optimizer.zero_grad()  # Clear gradients before the backpropagation
        output = model(data.x)  # Forward pass
        loss = criterion(output.view(-1), data.y)  # Compute loss
        loss.backward()  # Backpropagate the errors
        optimizer.step()  # Update model parameters
        mae_loss = mae_criterion(output.squeeze(-1).view(-1), data.y)  # Calculate MAE
        total_mae_loss += mae_loss.item()  # Sum up MAE loss for the epoch
        total_loss += loss.item()  # Sum up total loss for the epoch

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses

def validate(model, loader, criterion, device):
    """
    Validates the model on the provided dataset.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        loader (DataLoader): DataLoader providing the dataset.
        criterion (Loss): The loss function used for evaluation.
        device (torch.device): The device tensors will be transferred to (CPU or GPU).

    Returns:
        tuple: Average validation loss and mean absolute error loss.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_mae_loss = 0
    mae_criterion = torch.nn.L1Loss()  # Additional criterion for MAE calculation

    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            data = data.to(device)  # Transfer data to the specified device
            output = model(data.x)  # Forward pass
            loss = criterion(output.view(-1), data.y)  # Compute loss
            mae_loss = mae_criterion(output.view(-1), data.y)  # Calculate MAE
            total_loss += loss.item()  # Sum up total loss for the batch
            total_mae_loss += mae_loss.item()  # Sum up MAE loss for the batch

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses


def inference(model, loader, device):
    """
    Performs inference using the trained model and returns the actual and predicted values.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        loader (DataLoader): DataLoader providing the dataset for inference.
        device (torch.device): The device on which the model and data should be processed.

    Returns:
        tuple: Two lists containing the actual labels and the predicted values for all the batches in the loader.
    """
    model.eval()  # Set the model to evaluation mode, which turns off dropout and batch normalization.
    actuals = []  # List to store the actual labels from the dataset.
    predictions = []  # List to store the model's predictions.

    with torch.no_grad():  # Disable gradient calculation to speed up the process and reduce memory usage.
        for data in loader:
            data = data.to(device)  # Move the data to the specified device (GPU or CPU).
            output = model(data.x)  # Compute the model's output.
            actuals.append(data.y.view(-1).numpy())  # Store the actual labels after flattening and converting to numpy.
            predictions.append(output.view(-1).numpy())  # Store the predictions in a similar manner.

    return actuals, predictions  # Return the collected actual labels and predictions.


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
    parser.add_argument('--dataset', type=str, default='caltrans_speed', help='Path to the training data file')
    parser.add_argument('--train-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\lstm data\\caltrans_speed_data_test_seq4.pt', help='Path to the training data file')
    parser.add_argument('--val-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\lstm data\\caltrans_speed_data_test_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--test-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\lstm data\\caltrans_speed_data_test_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension of the MLP')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in the MLP')
    return parser.parse_args()


def main():
    logger = setup_logging()  # Initialize logging
    
    args = parse_args()  # Parse command-line arguments

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_loader = load_data(args.train_path, args.batch_size)
    val_loader = load_data(args.val_path, args.batch_size)
    test_loader = DataLoader(torch.load(args.test_path), batch_size=1, shuffle=False)

    # Model setup
    model = MLP(
        input_dim=4,  # Assuming 4 features per input
        hidden_dim=args.hidden_dim, 
        output_dim=1,  # Assuming a single output value
        num_layers=args.num_layers
    ).to(device)
    
    # Calculate total parameters in the model
    total_params = count_parameters(model)
    print(f"Total number of parameters in the model: {total_params}")

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []  # Lists to track loss

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_mae_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train MAE Loss: {train_mae_loss:.6f}, Val MAE Loss: {val_mae_loss:.6f}")

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Save the trained model
    model_path = f"models\\{args.dataset}_mlp_model_final.pth"
    torch.save(model.state_dict(), model_path)
    
    # Load the model for inference
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Perform inference
    actuals, predictions = inference(model, test_loader, device)
    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)
    
    # Display test loss metrics
    print('Test loss MSE:', np.mean((actuals_array - predictions_array)**2))
    print('Test loss MAE:', np.mean(np.abs(actuals_array - predictions_array)))
    
    # Save the actuals and predictions
    np.save(f'actuals_mlp_{args.dataset}_test', actuals_array)
    np.save(f'predictions_mlp_{args.dataset}_test', predictions_array)

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
