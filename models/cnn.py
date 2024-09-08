import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn as nn
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


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for processing sequential data.
    
    Args:
        num_nodes (int): Number of nodes in each sequence.
        sequence_length (int): Length of each sequence.
        input_channels (int, optional): Number of input channels. Defaults to 1.
        num_filters (list, optional): List of the number of filters for each convolutional layer. Defaults to [16, 32].
        kernel_sizes (list, optional): List of kernel sizes for each convolutional layer. Defaults to [3, 3].
        pool_sizes (list, optional): List of pooling sizes for each convolutional layer. Defaults to [2, 2].
        output_dim (int, optional): Dimension of the output layer. If not provided, defaults to `num_nodes`.
    
    Attributes:
        features (nn.Sequential): Sequence of layers containing convolutions, activations, and pooling.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer producing the output.
    """
    def __init__(self, num_nodes, sequence_length, input_channels=1, num_filters=[16, 32], kernel_sizes=[3, 3], pool_sizes=[2, 2], output_dim=None):
        super(CNN, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.output_dim = output_dim if output_dim is not None else num_nodes
        
        # Initialize convolutional layers
        layers = []
        current_channels = input_channels
        current_dim = (sequence_length, num_nodes)
        
        for num_filter, kernel_size, pool_size in zip(num_filters, kernel_sizes, pool_sizes):
            layers.append(nn.Conv2d(current_channels, num_filter, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            if pool_size > 1:
                layers.append(nn.MaxPool2d(pool_size))
                current_dim = (current_dim[0] // pool_size, current_dim[1] // pool_size)
            current_channels = num_filter
        
        self.features = nn.Sequential(*layers)
        flat_features = current_channels * current_dim[0] * current_dim[1]
        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, self.output_dim)


    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, sequence_length, num_nodes).

        Returns:
            torch.Tensor: Output of the CNN after processing input tensor `x`.
        """
        x_last = x[:, -1, :]  # Extract last time step for residual connection
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.features(x)  # Apply convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the features
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU
        x = self.fc2(x)  # Second fully connected layer to produce the output
        return x + x_last  # Add last time step's features for residual connection


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
            output = model(data.x)  # Compute output
            loss = criterion(output.view(-1), data.y)  # Compute loss
            mae_loss = mae_criterion(output.view(-1), data.y)  # Calculate MAE loss
            total_loss += loss.item()  # Aggregate total loss
            total_mae_loss += mae_loss.item()  # Aggregate MAE loss

    return total_loss / len(loader), total_mae_loss / len(loader)  # Return average losses per batch


def inference(model, loader, device):
    """
    Performs inference using the given model and data loader.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (DataLoader): DataLoader containing the dataset for inference.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Lists of actual and predicted values for each batch in the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    actuals = []  # List to store actual labels
    predictions = []  # List to store predictions

    with torch.no_grad():  # Disable gradient computation for inference
        for data in loader:
            data = data.to(device)  # Move data to the specified device
            output = model(data.x)  # Compute the model output
            actuals.append(data.y.numpy())  # Store the actual labels
            predictions.append(output.view(-1).numpy())  # Store the predictions

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
    parser = argparse.ArgumentParser(description='Train a CNN model on time-series data.')
    parser.add_argument('--dataset', type=str, default='caltrans_speed', help='Path to the training data file')
    parser.add_argument('--train-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cnn data\\caltrans_speed_data_train_seq4.pt', help='Path to the training data file')
    parser.add_argument('--val-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cnn data\\caltrans_speed_data_val_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--test-path', type=str, 
                        default='C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cnn data\\caltrans_speed_data_test_seq4.pt', help='Path to the validation data file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--num-filters', type=int, nargs='+', default=[16, 32], help='List of the number of filters for each convolutional layer')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[3, 3], help='List of kernel sizes for each convolutional layer')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[2, 2], help='List of pooling sizes for each convolutional layer')
    return parser.parse_args()


def main():
    # Setup logging
    logger = setup_logging()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Device configuration
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader = load_data(args.train_path, args.batch_size)
    val_loader = load_data(args.val_path, args.batch_size)
    test_loader = load_data(args.test_path, 1)
    
    # Model setup
    model = CNN(
        num_nodes=697,  #21672 for smart meter data 9 digit, 5166 for full nrel data, 697 for caltrans speed
        sequence_length=4,
        input_channels=1,
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes,
        output_dim=697  # Output one value per node
    ).to(device)
    
    # Log the total number of parameters in the model
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')
    
    # Optimizer and loss function setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    # Lists to store losses
    train_losses, val_losses = [], []
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_mae_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train MAE Loss: {train_mae_loss:.6f}, Val MAE Loss: {val_mae_loss:.6f}')
    
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
    model_path = f'models/{args.dataset}_cnn_model_final.pth'
    torch.save(model.state_dict(), model_path)
    
    # Load the model for inference
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Perform inference
    actuals, predictions = inference(model, test_loader, device)
     
    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)
    
    print('Test loss MSE:', np.mean((actuals_array - predictions_array)**2))
    print('Test loss MAE:', np.mean(np.abs(actuals_array - predictions_array)))
    
    np.save(f'actuals_cnn_{args.dataset}_test', actuals_array)
    np.save(f'predictions_cnn_{args.dataset}_test', predictions_array)
    
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
