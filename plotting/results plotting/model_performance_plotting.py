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

# Define datasets and models
datasets = ['nrel_full_c', 'smart_meter_9digit', 'caltrans_speed']
model = 'gcn'
base_path = 'C:/Users/bmb2tn/OneDrive - University of Virginia/Ph.D. Projects/Big Data'

# Color map for line coloring
viridis = cm.get_cmap('viridis', 256)
actual_color = viridis(230)  # Yellow-like color from Viridis
predicted_color = viridis(30)  # Purple-like color from Viridis
error_color = viridis(150)  # Green-like color for errors

# Loop through each dataset
for dataset in datasets:
    plt.figure(figsize=(24, 13.5))
    ax = plt.gca()

    actual_file = os.path.join(base_path, f'actuals_{model}_{dataset}_test.npy')
    prediction_file = os.path.join(base_path, f'predictions_{model}_{dataset}_test.npy')

    if os.path.exists(actual_file) and os.path.exists(prediction_file):
        actuals = np.load(actual_file)
        predictions = np.load(prediction_file)

        mean_actuals = np.mean(actuals, axis=1)
        mean_predictions = np.mean(predictions, axis=1)
        error = np.mean(np.abs(actuals - predictions), axis=1)

        # Plotting the data
        ax.plot(mean_actuals, label='Exact Mean', color=actual_color, linestyle='-', linewidth=3)
        ax.plot(mean_predictions, label='Predicted Mean', color=predicted_color, linestyle='--', linewidth=2)
        ax.plot(error, label='MAE', color=error_color, linestyle='-', linewidth=2)

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.2)  # Extend the upper limit for visibility

        # Set labels and title
        ax.set_xlabel('Time Step', fontsize=label_fontsize)
        ylabel = {
            'nrel_full_c': 'Generation/Capacity Ratio',
            'caltrans_speed': 'Average Speed (mph)',
            'smart_meter_9digit': 'Energy Consumption (kWh)'
        }[dataset]
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=20)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, length=tick_length, width=tick_width)

        # Legend configuration
        ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1), ncols=3)

        #plt.title(f'Model Performance for {dataset.replace("_", " ").title()}', fontsize=title_fontsize)
        plt.savefig(os.path.join(base_path, f'Model_Performance_{model}_{dataset}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
