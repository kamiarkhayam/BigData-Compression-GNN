import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

# Font and style settings
plt.rcParams["font.family"] = "Times New Roman"
label_fontsize = 50
title_fontsize = 50
legend_fontsize = 45
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
viridis = cm.get_cmap('viridis', 256)
error_color_1 = viridis(50)  # Color for first error line
error_color_2 = viridis(200)  # Color for second error line
stored_color_1 = viridis(100)  # Color for stored points for first threshold
stored_color_2 = viridis(250)  # Color for stored points for second threshold

# Pre-select two random nodes for each dataset
np.random.seed(72)
selected_nodes = {}
for dataset in datasets:
    sample_file = os.path.join(base_path, f'{dataset}_gcn_{datasets[dataset][0]}_results.npy')
    sample_data = np.load(sample_file)
    selected_nodes[dataset] = np.random.choice(sample_data.shape[1], 2, replace=False)

# Loop through each dataset, model, and selected nodes
for dataset, thresholds in datasets.items():
    for model in models:
        for node_index in selected_nodes[dataset]:
            pairs = [(1, 2)]  # Define pairs of thresholds
            for pair in pairs:
                plt.figure(figsize=(24, 13.5))
                handles = []  # To store custom handles for the legend
                for i, threshold_index in enumerate(pair):
                    threshold = thresholds[threshold_index]
                    results_file = os.path.join(base_path, f'{dataset}_{model}_{threshold}_results.npy')
                    actuals_file = os.path.join(base_path, f'{dataset}_{model}_{threshold}_actuals.npy')

                    results = np.load(results_file)
                    actuals = np.load(actuals_file)

                    actual_series = actuals[:2000, node_index]
                    predicted_series = results[:2000, node_index]
                    error = np.abs(actual_series - predicted_series) / (np.abs(actual_series) + 0.05)
                    time_steps = np.arange(2000)
                    line_color = error_color_1 if i == 0 else error_color_2
                    stored_color = stored_color_1 if i == 0 else stored_color_2

                    # Plot relative error
                    if dataset == 'smart_meter_full':
                        plt.plot(time_steps, error, label=f'$\epsilon$ = {threshold}kWh', color=line_color, linestyle='-', linewidth=2)
                        line, = plt.plot([], [], color=line_color, label=f'Error for $\epsilon$ = {threshold}kWh')
                    elif dataset == 'nrel_full_c':
                        plt.plot(time_steps, error, label=f'$\epsilon$ = {threshold}', color=line_color, linestyle='-', linewidth=2)
                        line, = plt.plot([], [], color=line_color, label=f'Error for $\epsilon$ = {threshold}')
                    elif dataset == 'caltrans_speed':
                        plt.plot(time_steps, error, label=f'$\epsilon$ = {threshold}mph', color=line_color, linestyle='-', linewidth=2)
                        line, = plt.plot([], [], color=line_color, label=f'Error for $\epsilon$ = {threshold}mph')
                    #line, = plt.plot([], [], color=line_color, label=f'Error for Threshold {threshold}')
                    
                    
                    
                    # Indicate matching time steps
                    non_matching_times = time_steps[~np.isclose(actual_series, predicted_series, atol=1e-6)]
                    ymin_val = 0.95 - 0.05 * i
                    ymax_val = 1 - 0.05 * i
                    for mt in non_matching_times:
                        plt.axvline(x=mt, ymin=ymin_val, ymax=ymax_val, color=stored_color, linestyle='-', linewidth=0.5)
                    # Plot relative error
                    if dataset == 'smart_meter_full':
                        discarded_line = plt.Line2D([], [], color=stored_color, linestyle='-', linewidth=2, label=f'Discarded for $\epsilon$ = {threshold}kWh')
                    elif dataset == 'nrel_full_c':
                        discarded_line = plt.Line2D([], [], color=stored_color, linestyle='-', linewidth=2, label=f'Discarded for $\epsilon$ = {threshold}')
                    elif dataset == 'caltrans_speed':
                        discarded_line = plt.Line2D([], [], color=stored_color, linestyle='-', linewidth=2, label=f'Discarded for $\epsilon$ = {threshold}mph')
                    
                    
                    handles.extend([line, discarded_line])

                # Add labels, title, and customize tick labels
                plt.xlabel('Time Step', fontsize=label_fontsize)
                plt.ylabel('Relative Error', fontsize=label_fontsize, labelpad=20)
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)
                #plt.title(f'{dataset.upper()} - Model: {model.upper()} - Node: {node_index}', fontsize=title_fontsize)
                # Increase ylim to make room for the matching lines at the top
                y_min, y_max = plt.ylim()
                plt.ylim(y_min, y_max * 1.5)  # Increase the max ylim by 30%
                # Create the legend with all handles
                plt.legend(handles=handles, fontsize=legend_fontsize, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncols=2)

                # Save and show plot
                plt.savefig(os.path.join(base_path, f'{dataset}_{model}_Node{node_index}_Combined_Error_Analysis_{pair}.png'), dpi=300)
                plt.show()
                plt.close()
