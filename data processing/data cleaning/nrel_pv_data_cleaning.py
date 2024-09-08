import os
import zipfile
import pandas as pd
import shutil

def extract_and_process_zip(base_path, state_abbreviations):
    """
    Extract and process zip files containing PV data for each state, and save the processed data into CSV files.

    Args:
        base_path (str): The directory where the zip files are located.
        state_abbreviations (list): A list of state abbreviations to filter the zip files for processing.

    Returns:
        None
    """
    temp_dir = os.path.join(base_path, "temp")  # Temporary directory for extraction

    # Loop through the list of states to process data for each state
    for state in state_abbreviations:
        state_data_frames = []  # List to store dataframes for each state
        print(f"Starting processing for state: {state.upper()}")
        
        # Walk through the base directory to find zip files
        for root, dirs, files in os.walk(base_path):
            for file in files:
                # Check if the file is a zip and matches the state abbreviation
                if file.endswith(".zip") and file[:2].lower() == state:
                    zip_path = os.path.join(root, file)
                    print(f"Processing zip file: {zip_path}")
                    
                    # Ensure the temp directory is clean before extracting
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)  # Remove existing temp directory if any
                    os.makedirs(temp_dir)  # Create a fresh temp directory
                    
                    # Extract the contents of the zip file into the temp directory
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(path=temp_dir)
                    print(f"Extracted {zip_path} to temporary directory.")
                    
                    # Process each extracted CSV file in the temp directory
                    for extracted_file in os.listdir(temp_dir):
                        if extracted_file.startswith("Actual") and extracted_file.endswith(".csv"):
                            file_path = os.path.join(temp_dir, extracted_file)
                            print(f"Processing file: {file_path}")
                            df = pd.read_csv(file_path)
                            
                            # Extract metadata from the filename
                            metadata = extracted_file.split('_')
                            latitude = metadata[1]
                            longitude = metadata[2]
                            weather_year = metadata[3]
                            pv_type = metadata[4]
                            capacity_mw = metadata[5]
                            
                            # Convert row data to columns for each 5-minute interval
                            series_data = df['Power(MW)'].values.flatten()  # Flatten the power data
                            columns_data = [f"Minute_{i}" for i in range(len(series_data))]  # Create column names
                            data_row = pd.DataFrame([series_data], columns=columns_data)  # Create a new dataframe
                            
                            # Combine metadata with interval data
                            for label, value in zip(['Latitude', 'Longitude', 'Weather Year', 'PV Type', 'CapacityMW'],
                                                    [latitude, longitude, weather_year, pv_type, capacity_mw]):
                                data_row[label] = value  # Add metadata as additional columns
                            
                            state_data_frames.append(data_row)  # Append the processed data row to the state's list

                    # Clean up temporary directory after processing the current zip file
                    shutil.rmtree(temp_dir)  # Remove the temp directory to keep it clean
                    print(f"Temporary directory removed after processing {zip_path}")

        # After processing all zip files for the state, save the combined data to a CSV file
        if state_data_frames:
            state_df = pd.concat(state_data_frames, ignore_index=True)  # Concatenate all the dataframes for the state
            state_df.to_csv(f"{base_path}/data/{state}_data.csv", index=False)  # Save the combined dataframe to a CSV
            print(f"Data for {state.upper()} saved to CSV.")
            del state_data_frames  # Explicitly free up memory by deleting the list


# Base path where the zip files are located
base_path = r"C:\Users\bmb2tn\OneDrive - University of Virginia\Ph.D. Projects\Big Data\NREL PV Data"
state_abbreviations = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 
    'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 
    'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
]

# Call the function with the base path and the list of state abbreviations
extract_and_process_zip(base_path, state_abbreviations)
