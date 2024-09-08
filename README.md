# Graph Neural Networks for Precision-Guaranteed Compression of Big Data

This repository hosts the implementation of the research detailed in "Graph Neural Networks for Precision-Guaranteed Compression of Big Data," presented at IEEE Big Data Conference 2024. The study offers a novel approach utilizing Graph Neural Networks (GNNs) to efficiently manage large-scale data sets in smart infrastructures, ensuring precision in decompression.

## Repository Structure

- **data_processing/**
  - **data_cleaning/**: Contains scripts for data cleaning across three distinct datasets.
  - **data_generation/**: Scripts for generating training, validation, and test datasets in formats suitable for graph, CNN, and LSTM models.
- **models/**: Includes class files and training scripts for the deep learning models used.
- **plotting/**
  - **map_plotting/**: Scripts for generating spatial overview maps of datasets.
  - **results_plotting/**: Scripts for plotting the results of data compression and model performance.
- **compression/**: Contains Python scripts for data compression using DL models and traditional methods for three datasets and utility functions.

## Getting Started

1. **Setup**
   - Ensure Python 3.x is installed on your system.
   - Clone this repository using:
     ```
     git clone <repository-url>
     ```
   - Install required Python packages:
     ```
     pip install -r requirements.txt
     ```

2. **Download the Data**
   - **NREL Dataset**: Available [here](https://www.nrel.gov/grid/solar-power-data.html). Synthetic solar photovoltaic (PV) power plant data points for the United States for the year 2006.
   - **Caltrans Data**: Available through [Caltrans PeMS dashboards](https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=7&submit=Submit).
   - **ComEd Dataset**: Contact ComEd to purchase smart meter data.

3. **Data Preparation**
   - Start by cleaning the data using scripts in the `data_cleaning` directory.
   - Prepare the data in the required formats for different DL model training using scripts in the `data_generation` directory.

4. **Model Training**
   - Proceed to the `models` folder to train various models using the provided scripts.

5. **Data Compression**
   - Utilize the trained models and prepared data to perform compression in the `compression` folder.

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- numpy
- pywt
- zfpy
- sklearn
- scipy
- pandas
- matplotlib
- geopandas
- folium
- shapely
- fitz
- PTL

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Citation
If you use this code or the framework in your research, please cite our paper:
Khayambashi, K., & Alemazkoor, N. (2024). Graph neural networks for precision-guaranteed compression of large-scale spatial data. In Proceedings of the IEEE Big Data Conference. IEEE.
