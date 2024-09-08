import pandas as pd
import gzip
import os
from datetime import datetime

def process_day(filepath):
    """
    Process a single day's traffic data file, extracting the relevant columns and setting a multi-level index.

    Args:
        filepath (str): Path to the gzipped CSV file containing the traffic data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data with a multi-level index (Station, Lane Type, Station Length, Timestamp).
    """
    with gzip.open(filepath, 'rt') as file:
        # Read the file and extract relevant columns
        df = pd.read_csv(file, header=None, usecols=[0, 1, 5, 6, 9, 11],
                         names=['Timestamp', 'Station', 'Lane Type', 'Station Length', 'Total Flow', 'Avg Speed'])
        # Convert 'Timestamp' to datetime object for easy time-based operations
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M:%S')
        # Set the multi-level index for the dataframe
        df.set_index(['Station', 'Lane Type', 'Station Length', 'Timestamp'], inplace=True)
        return df


def save_dataframe(df, filename):
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The path and name of the file to save the DataFrame to.
    """
    df.to_csv(filename)


def process_files(base_path, start_date, end_date, output_filename_flow, output_filename_speed):
    """
    Process traffic data files for a given date range, aggregating total flow and average speed data, 
    and saving the results to CSV files.

    Args:
        base_path (str): The directory where the daily gzipped data files are located.
        start_date (str): The start date for the data processing period (inclusive).
        end_date (str): The end date for the data processing period (inclusive).
        output_filename_flow (str): The name of the file to save the aggregated flow data.
        output_filename_speed (str): The name of the file to save the aggregated speed data.
    
    Returns:
        None
    """
    # Build timestamp index for the entire period with a frequency of 5 minutes
    timestamps = pd.date_range(start=start_date, end=end_date, freq='5T')
    
    # Initialize master DataFrames to store aggregated flow and speed data
    master_flow = pd.DataFrame()
    master_speed = pd.DataFrame()

    # Iterate through each day in the given date range
    for day in pd.date_range(start=start_date, end=end_date):
        # Construct the filename for the current day
        filename = f"d03_text_station_5min_{day.strftime('%Y_%m_%d')}.txt.gz"
        filepath = os.path.join(base_path, filename)

        # Check if the file exists, otherwise skip
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}, skipping...")
            continue
        
        # Process the data for the current day
        day_data = process_day(filepath)
        
        # Aggregate 'Total Flow' and 'Avg Speed' by station, filling missing values with 0
        flow_data = day_data['Total Flow'].unstack(fill_value=0)
        speed_data = day_data['Avg Speed'].unstack(fill_value=0)

        # Append the current day's data to the master DataFrames
        master_flow = pd.concat([master_flow, flow_data], axis=1)
        master_speed = pd.concat([master_speed, speed_data], axis=1)

        print(f"Completed processing for date: {day.strftime('%Y-%m-%d')}")

    # Save the aggregated flow and speed data to CSV files
    save_dataframe(master_flow, output_filename_flow)
    save_dataframe(master_speed, output_filename_speed)
    
    print("Data processing completed. Files saved.")
    

def load_data(filename):
    """
    Load traffic data from a CSV file, handling different timestamp formats in the columns.

    Args:
        filename (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame with the index set to the first three columns and the columns converted to datetime objects.
                      If the conversion fails, None is returned.
    """
    # Load the CSV file, setting the first three columns as the index
    df = pd.read_csv(filename, index_col=[0, 1, 2])
    try:
        # Convert the column headers (timestamps) to datetime format, allowing mixed formats
        df.columns = pd.to_datetime(df.columns, errors='coerce', infer_datetime_format=True)
        # If any columns could not be converted to datetime, remove them
        if df.columns.isnull().any():
            print("Warning: Some timestamps could not be parsed and will be excluded.")
            df = df.loc[:, df.columns.notnull()]
    except Exception as e:
        # Handle any errors during the datetime conversion process
        print(f"Error converting columns to datetime: {e}")
        return None
    return df


def clean_and_save_data(window_df):
    """
    Clean the DataFrame by removing rows with excessive missing data, interpolating missing values, 
    and saving the cleaned data to a CSV file.

    Args:
        window_df (pd.DataFrame): The DataFrame to clean.

    Returns:
        None
    """
    # Set a threshold to drop rows with more than 10% missing data
    threshold = len(window_df.columns) * 0.1
    # Drop rows that have more missing data than the threshold
    filtered_df = window_df.dropna(thresh=len(window_df.columns) - threshold)

    # Drop rows where the first or last columns contain NaN values
    filtered_df = filtered_df.dropna(subset=[filtered_df.columns[0], filtered_df.columns[-1]])

    # Interpolate missing values along each row (i.e., across time for each station)
    interpolated_df = filtered_df.interpolate(method='linear', limit_direction='both', axis=1)

    # Fill any remaining missing values by forward filling and backward filling
    interpolated_df.fillna(method='ffill', axis=1, inplace=True)
    interpolated_df.fillna(method='bfill', axis=1, inplace=True)

    # Drop rows that still contain NaNs after interpolation and filling
    final_df = interpolated_df.dropna()

    # Check if any NaNs remain after the cleaning process
    if final_df.isna().any().any():
        print("Warning: Some NaNs remain even after interpolation and filling.")
    else:
        print("All missing values have been handled.")

    print(f"Shape of DataFrame after cleaning: {final_df.shape}")
    
    # Save the cleaned DataFrame to a specified file path
    interpolated_df.to_csv('C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\cleaned_speed_data_full.csv')
    print("Cleaned data has been saved to 'cleaned_flow_data.csv'.")

    
# Define parameters for file processing
base_path = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins'
start_date = datetime(2023, 1, 1)  # Start date for data processing
end_date = datetime(2023, 12, 31)  # End date for data processing
output_filename_flow = 'total_flow.csv'  # Output file for total flow data
output_filename_speed = 'average_speed.csv'  # Output file for average speed data

# Process files to extract and save traffic data for the given date range
process_files(base_path, start_date, end_date, output_filename_flow, output_filename_speed)


# Define file paths for the cleaned data
output_filename_flow = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\total_flow.csv'
output_filename_speed = 'C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Caltrans PEMS Station 5mins\\average_speed.csv'

# Load the saved speed data into a DataFrame
df_speed = load_data(output_filename_speed)

# Clean the loaded speed data and save the cleaned version
clean_and_save_data(df_speed)
