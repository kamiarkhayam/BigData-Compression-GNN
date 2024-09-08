import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

# Font and style settings
plt.rcParams["font.family"] = "Times New Roman"
label_fontsize = 50
title_fontsize = 50
legend_fontsize = 50
tick_fontsize = 50
tick_length = 10  # Length of the tick marks
tick_width = 2    # Width of the tick marks

# Define datasets, models, and their thresholds
datasets = {
    'smart_meter_full': [0.06, 0.1, 0.4, 1.0],
    'nrel_full_c': [0.001, 0.005, 0.01, 0.02],
    'caltrans_speed': [0.4, 1.0, 2.0, 4.0]
}

models = ['gcn']
base_path = 'C:/Users/bmb2tn/OneDrive - University of Virginia/Ph.D. Projects/Big Data'

# Viridis color map for line coloring
viridis = cm.get_cmap('viridis', 256)  # Get 256 colors from Viridis colormap
actual_color = viridis(230)  # Yellow-like color from Viridis
predicted_color = viridis(30)  # Purple-like color from Viridis
discarded_color = viridis(150)  # Green-like color from Viridis for non-matching points

# Pre-select two random nodes for each dataset
np.random.seed(1)  # For reproducibility
selected_nodes = {}
for dataset in datasets:
    sample_file = os.path.join(base_path, f'{dataset}_gcn_{datasets[dataset][0]}_results.npy')
    sample_data = np.load(sample_file)  # Load data in original shape
    selected_nodes[dataset] = np.random.choice(sample_data.shape[1], 2, replace=False)

# Loop through each dataset, model, and threshold
for dataset, thresholds in datasets.items():
    for model in models:
        for threshold in thresholds:
            results_file = os.path.join(base_path, f'{dataset}_{model}_{threshold}_results.npy')
            actuals_file = os.path.join(base_path, f'{dataset}_{model}_{threshold}_actuals.npy')

            # Load data in original shape (time_steps, num_nodes)
            results = np.load(results_file)
            actuals = np.load(actuals_file)
            time_steps = np.arange(actuals.shape[0])

            # Use pre-selected nodes
            for node_index in selected_nodes[dataset]:
                plt.figure(figsize=(24, 13.5))

                # Plot the actual and predicted data
                actual_series = actuals[:, node_index]
                predicted_series = results[:, node_index]
                line1, = plt.plot(time_steps, actual_series, label='Exact', color=actual_color, linestyle='-', linewidth=3)
                line2, = plt.plot(time_steps, predicted_series, label='Decompressed', color=predicted_color, linestyle='dashed', linewidth=2, dashes=(6, 3))

                # Find time steps where actual and predicted values do not match exactly
                non_matching_times = time_steps[~np.isclose(actual_series, predicted_series, atol=1e-6)]
                ratio = len(non_matching_times) / len(time_steps)

                # Increase ylim to make room for the matching lines at the top
                y_min, y_max = plt.ylim()
                plt.ylim(y_min, y_max * 1.2)  # Increase the max ylim by 30%

                # Highlight non-matching time steps
                for mt in non_matching_times:
                    plt.axvline(x=mt, ymin=0.95, ymax=1, color=discarded_color, linestyle='-', linewidth=0.3)

                # Custom legend entry for 'Discarded' lines
                discarded_line = plt.Line2D([], [], color=discarded_color, linestyle='-', linewidth=2, label='Discarded')

                # Add labels, title, and customize tick labels
                #plt.title(f'{dataset.upper()} - Model: {model.upper()} - Threshold: {threshold} - Node: {node_index}', fontsize=title_fontsize)
                plt.xlabel('Time Step', fontsize=label_fontsize)
                # Setting dataset-specific y-axis labels
                if dataset == 'nrel_full_c':
                    plt.ylabel('Generation/Capacity Ratio', fontsize=label_fontsize, labelpad=20)
                elif dataset == 'caltrans_speed':
                    plt.ylabel('Average Speed (mph)', fontsize=label_fontsize, labelpad=20)
                elif dataset == 'smart_meter_full':
                    plt.ylabel('Energy Consumption (kWh)', fontsize=label_fontsize, labelpad=20)
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)

                
                # Create the legend with all handles
                plt.legend(handles=[line1, line2, discarded_line], fontsize=legend_fontsize, frameon=False, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
                if ratio > 0.55:
                    plt.savefig(os.path.join(base_path, f'{dataset}_{model}_{threshold}_Node{node_index}_actual_vs_decompressed_{ratio}.png'), dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()
                plt.close()
