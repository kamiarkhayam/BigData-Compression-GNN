import os
import numpy as np
import zipfile
import pandas as pd
from glob import glob
import shutil

def extract_zip(zip_path, extract_to):
    """
    Extract the contents of a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory to extract the contents into.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)  # Extract all contents to the specified directory


def parse_date(date_str):
    """
    Parse a string date into a pandas datetime object using the format '%Y%m%d'.

    Args:
        date_str (str): The date string to be parsed (format: '%Y%m%d').

    Returns:
        pd.Timestamp: A pandas datetime object representing the parsed date.
    """
    # Convert the date string to a pandas datetime object
    return pd.to_datetime(date_str, format='%Y%m%d')


def filter_dates(df):
    """
    Filter a DataFrame to include only rows with dates between January 2021 and December 2022.

    Args:
        df (pd.DataFrame): The DataFrame to filter, containing a column 'INTERVAL_READING_DATE'.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows within the specified date range.
    """
    # Define the start and end date for filtering
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2022-12-31')
    
    # Filter the DataFrame to include only rows where 'INTERVAL_READING_DATE' is within the range
    return df[(df['INTERVAL_READING_DATE'] >= start_date) & (df['INTERVAL_READING_DATE'] <= end_date)]


def process_energy_strings(row):
    """
    Process a list of energy string values, splitting and converting them into floats.

    Args:
        row (list): A list of string values, where each string contains space-separated float numbers.

    Returns:
        list: A list of floats, created by splitting each string and converting the parts to float.
    """
    new_row = []
    # Split each string in the row and convert to float
    for item in row:
        new_row.extend([float(x) for x in item.split()])  # Split the string and convert each part to float
    return new_row


def check_missing_timestamps(df):
    """
    Check for missing timestamps in the DataFrame's column names, assuming they represent timestamps.

    Args:
        df (pd.DataFrame): The DataFrame whose columns are checked for missing timestamps.

    Returns:
        None
    """
    # Identify columns that can be parsed as timestamps
    timestamp_columns = [col for col in df.columns if pd.to_datetime(col, errors='coerce') is not pd.NaT]

    # Convert the identified columns to datetime objects
    actual_timestamps = pd.to_datetime(timestamp_columns)

    # Generate a range of expected timestamps with 30-minute intervals
    start_date = actual_timestamps[0]
    end_date = actual_timestamps[-1]
    expected_timestamps = pd.date_range(start=start_date, end=end_date, freq='30T')

    # Compare the expected timestamps with the actual ones to find any missing timestamps
    missing_timestamps = expected_timestamps.difference(actual_timestamps)

    # Output missing timestamps, if any
    if not missing_timestamps.empty:
        print("Missing timestamps:")
        print(missing_timestamps)
    else:
        print("No missing timestamps.")


def concatenate_and_interpolate_dataframes(filtered_dfs):
    """
    Concatenate a list of filtered DataFrames, ensuring they are indexed by 'ZIP_CODE' and
    interpolating missing timestamps.

    Args:
        filtered_dfs (list of pd.DataFrame): List of DataFrames filtered by ZIP code.

    Returns:
        pd.DataFrame: A concatenated and interpolated DataFrame with timestamps as columns.
    """
    # Ensure each DataFrame is indexed by 'ZIP_CODE' and sorted by index
    for i, df in enumerate(filtered_dfs):
        if df.index.name != 'ZIP_CODE':  # Set 'ZIP_CODE' as the index if it's not already
            df.set_index('ZIP_CODE', inplace=True)
        df.sort_index(inplace=True)  # Sort the DataFrame by 'ZIP_CODE' for consistency

    # Concatenate the DataFrames along the columns (i.e., combine the columns from each DataFrame)
    master_df = pd.concat(filtered_dfs, axis=1)

    # Remove duplicate columns, if any exist (based on column names)
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    
    # Convert column names to datetime objects for easier time-based operations
    master_df.columns = pd.to_datetime(master_df.columns)

    # Determine the range of timestamps in the DataFrame
    start_date = pd.to_datetime(master_df.columns.min())  # Find the earliest timestamp
    end_date = pd.to_datetime(master_df.columns.max())    # Find the latest timestamp
    
    # Generate a complete range of dates from the earliest to the latest timestamp
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate a full range of 30-minute intervals (timestamps) from start to end date
    all_timestamps = pd.date_range(start=all_dates[0], end=all_dates[-1] + pd.Timedelta(days=1), freq='30T')[:-1]
    
    # Filter out specific timestamps (e.g., excluding September 2021 and all of 2023)
    filtered_timestamps = all_timestamps[
        ~((all_timestamps.month == 9) & (all_timestamps.year == 2021)) &  # Exclude September 2021
        ~(all_timestamps.year == 2023)  # Exclude the entire year 2023
    ]
    
    # Find missing timestamps by comparing expected timestamps with existing ones
    missing_timestamps = filtered_timestamps.difference(pd.to_datetime(master_df.columns))
    
    if not missing_timestamps.empty:
        # Reindex the DataFrame to include all expected timestamps, filling missing values with NaN
        master_df = master_df.reindex(columns=all_timestamps, fill_value=np.nan)
        
        # Interpolate missing values linearly along the columns (i.e., time axis)
        master_df.interpolate(method='linear', axis=1, inplace=True)
    
    # Drop the last column (which may contain unwanted data)
    last_column = master_df.columns[-1]
    master_df = master_df.drop(columns=[last_column])  # Remove the last timestamp column

    return master_df  # Return the cleaned, concatenated, and interpolated DataFrame


def load_and_filter_common_zip_codes(csv_paths):
    """
    Load CSV files, find mutual ZIP codes across all DataFrames, filter rows with those ZIP codes, 
    and concatenate and interpolate the resulting DataFrames.

    Args:
        csv_paths (list of str): List of paths to the CSV files to load.

    Returns:
        pd.DataFrame: A concatenated and interpolated DataFrame containing only mutual ZIP codes.
    """
    # Initialize a list to store the DataFrames loaded from the CSV files
    dataframes = []

    # Load each CSV file and append the resulting DataFrame to the list
    for path in csv_paths:
        df = pd.read_csv(path)  # Load the CSV into a DataFrame
        dataframes.append(df)  # Append the DataFrame to the list

    # Find mutual ZIP codes present in all DataFrames
    mutual_zip_codes = set(dataframes[0]['ZIP_CODE'])  # Start with ZIP codes from the first DataFrame
    for df in dataframes[1:]:
        mutual_zip_codes.intersection_update(set(df['ZIP_CODE']))  # Keep only ZIP codes common to all DataFrames

    # Filter each DataFrame to include only rows with mutual ZIP codes
    filtered_dfs = [df[df['ZIP_CODE'].isin(mutual_zip_codes)] for df in dataframes]
    
    # Concatenate and interpolate the filtered DataFrames
    print('going to make master df')
    master_df = concatenate_and_interpolate_dataframes(filtered_dfs)  # Call the concatenation function
    print(f'made master df shape: {master_df.shape}')  # Output the shape of the resulting DataFrame
    
    return master_df  # Return the final concatenated and interpolated DataFrame


def aggregate_zip_codes(df):
    """
    Aggregate data in a DataFrame by ZIP code, grouping and summing the rows based on ZIP code.

    Args:
        df (pd.DataFrame): The DataFrame to aggregate, with ZIP codes as the index.

    Returns:
        pd.DataFrame: An aggregated DataFrame with data grouped by ZIP code.
    """
    # Ensure the index represents ZIP codes
    print('entered function')
    df.index = df.index.astype(str).str[:5]  # Convert the index to a string and use only the first 5 characters
    
    # Group the DataFrame by ZIP code (the new index) and sum the values for each ZIP code
    aggregated_df = df.groupby(df.index).sum()
    print(f'aggregated df shape: {aggregated_df.shape}')  # Output the shape of the aggregated DataFrame
    
    # Return the aggregated DataFrame
    return aggregated_df


    
def aggregate_data(folder_path):
    """
    Extracts, processes, and aggregates energy data from zip files containing CSVs, 
    organizing the data by ZIP code and timestamp.

    Args:
        folder_path (str): Path to the directory containing the zip files.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated energy data indexed by ZIP code and timestamp.
    """
    print(f"Processing main directory: {folder_path}")

    # Define columns representing 30-minute intervals for energy data
    hourly_cols = [f'INTERVAL_HR{h:02d}{m:02d}_ENERGY_QTY' for h in range(0, 24) for m in (30, 0)]
    
    # Remove the column 'INTERVAL_HR0000_ENERGY_QTY' if it exists (no data for this interval)
    hourly_cols = [col for col in hourly_cols if col != 'INTERVAL_HR0000_ENERGY_QTY']
    
    # Add the special column for the end of the day energy reading (24:00)
    hourly_cols.append('INTERVAL_HR2400_ENERGY_QTY')
    
    # Columns to use from the CSV files
    cols_to_use = ['ZIP_CODE', 'ACCOUNT_IDENTIFIER', 'INTERVAL_READING_DATE'] + hourly_cols

    cumulative_data = []  # Initialize list to store data from all CSVs

    # Temporary directory for extracting files
    temp_dir = os.path.join(folder_path, "temp")
    os.makedirs(temp_dir, exist_ok=True)  # Create the temp directory if it doesn't exist

    # Process each group-level zip file in the folder
    for group_zip_path in glob(os.path.join(folder_path, "*.zip")):
        extract_zip(group_zip_path, temp_dir)  # Extract the outer zip file
        print(f"Extracted group zip: {group_zip_path}")
        
        # Extract all inner zip files within the temporary directory
        for inner_zip_path in glob(f"{temp_dir}/**/*.zip", recursive=True):
            extract_zip(inner_zip_path, temp_dir)  # Extract the inner zip files
        
        # Process all CSV files after extracting all zips
        csv_files = glob(f"{temp_dir}/**/*.csv", recursive=True)
        
        for csv_file in csv_files:
            print(f"Processing CSV file: {csv_file}")
            
            # Load the CSV file, using only the relevant columns
            df = pd.read_csv(csv_file, usecols=cols_to_use)
            
            # Convert the 'INTERVAL_READING_DATE' column to datetime
            df['INTERVAL_READING_DATE'] = pd.to_datetime(df['INTERVAL_READING_DATE'], errors='coerce')
            
            # Set a multi-index using ZIP code, account identifier, and the reading date
            df = df.set_index(['ZIP_CODE', 'ACCOUNT_IDENTIFIER', 'INTERVAL_READING_DATE'])
            
            # Stack the interval data into long format and reset the index
            df = df.stack().reset_index()
            df.rename(columns={'level_3': 'Interval', 0: 'Energy'}, inplace=True)  # Rename columns
            
            # Convert the 'Energy' column to float for calculations
            df['Energy'] = df['Energy'].astype(float)

            # Extract hour and minute information from the 'Interval' column
            df['Hour'] = df['Interval'].str.extract(r'(\d{4})')[0].str[:2]
            df['Minute'] = df['Interval'].str.extract(r'(\d{4})')[0].str[2:]
            
            # Create a full datetime column by combining the date, hour, and minute
            df['Date'] = pd.to_datetime(df['INTERVAL_READING_DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['Hour'] + ':' + df['Minute'], format='%Y-%m-%d %H:%M', errors='coerce')
            
            # Drop the intermediate columns used to create the final datetime
            df.drop(['INTERVAL_READING_DATE', 'Interval', 'Hour', 'Minute'], axis=1, inplace=True)

            # Append the processed data to the cumulative list
            cumulative_data.append(df)

        # Clean up the temporary directory after processing the group-level zip
        shutil.rmtree(temp_dir)  # Remove the temp directory to ensure a fresh start for the next zip
        print(f"Cleaned up temporary directory for group: {group_zip_path}")

    # Concatenate all the processed data into a single DataFrame
    cumulative_data = pd.concat(cumulative_data)
    
    # Pivot the data to create a table with ZIP codes as rows and timestamps as columns
    aggregated_data = cumulative_data.pivot_table(index=['ZIP_CODE'], columns='Date', values='Energy', aggfunc='sum').reset_index()
    
    print('returning aggregated data')
    return aggregated_data  # Return the aggregated DataFrame


# Define the base directory
base_dir = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit"

# Generate month folders
month_folders = [f"{year}{month:02d}" for year in [2021, 2022] for month in range(1, 13)]
csv_file_paths = []

# Loop over each month's data
for month_folder in month_folders:
    month_zip_path = os.path.join(base_dir, f"{month_folder}.zip")
    extract_zip(month_zip_path, base_dir)
    month_dir = os.path.join(base_dir, month_folder)
    
    # Aggregate the data
    final_data = aggregate_data(month_dir)
    print('returned aggregated data')
    # Output the cumulative data to a CSV file
    csv_file_path = f"{base_dir}\\cumulative_data_{month_folder}.csv"
    csv_file_paths.append(csv_file_path)
    print('started to save data')
    final_data.to_csv(csv_file_path, index=False)
    print(f'saved data to {csv_file_path}')

    # Clean up the extracted folders
    shutil.rmtree(month_dir)

# Paths of the CSV files saved in the previous step
csv_file_paths = [
    f"{base_dir}\\cumulative_data_{year}{month:02d}.csv"
    for year in [2021, 2022]
    for month in range(1, 13)
    if not (year == 2021 and month == 9)
]

# Convert to NumPy array, filter by mutual ZIP codes, and save
master_df = load_and_filter_common_zip_codes(csv_file_paths)
master_df.to_csv('cumulative_data_9digit_master.csv', index=True)

master_df = aggregate_zip_codes(master_df)
master_df.to_csv('cumulative_data_5digit_master.csv', index=True)