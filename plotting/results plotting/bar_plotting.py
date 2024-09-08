import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Font and style settings
plt.rcParams["font.family"] = "Times New Roman"
label_fontsize = 19
legend_fontsize = 17
tick_fontsize = 18
ylabel_fontsize = 18  # Font size for y-labels

datasets = ['caltrans_speed', 'nrel_full_c', 'smart_meter_full', ]
models = ['graph_transformer', 'gcn', 'cnn', 'lstm', 'mlp', 'dct', 'wavelet','pca']

viridis = cm.get_cmap('viridis', 8)  # Get 8 colors from the Viridis colormap
color_map = {
    'graph_transformer': viridis(0),
    'gcn': viridis(1),
    'cnn': viridis(2),
    'lstm': viridis(3),
    'mlp': viridis(4),
    'dct': viridis(5),
    'pca': viridis(6),
    'wavelet': viridis(7)
}

model_full_names = {
    'graph_transformer': 'GT',
    'gcn': 'GCN',
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'mlp': 'MLP',
    'dct': 'DCT',
    'pca': 'PCA',
    'wavelet': 'WT'
}

def load_data(filename, model_name, dataset):
    data = {}
    processed_lines = set()  # Set to track processed lines and discard duplicates
    with open(filename, 'r') as file:
        for line in file:
            if line in processed_lines:
                continue
            processed_lines.add(line)  # Mark this line as processed

            threshold, ratio_str = line.strip().strip('[]').split(',')
            try:
                threshold = float(threshold)
                ratio = float(ratio_str)
            except ValueError:
                ratio = 0.0

            if threshold not in data:
                data[threshold] = []

            #if (model_name == 'mlp' or model_name == 'graph_transformer') and dataset == 'smart_meter_full':
            #    ratio *= 1.2
            data[threshold].append(ratio)

    for key in data:
        data[key] = np.mean(data[key])

    return data

def plot_data_for_dataset(dataset):
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds_index = {}
    n_bars = len(models)
    bar_width = 0.1  # Keep bar width reasonable
    group_spacing = 0.2  # Additional spacing between groups of bars

    # Load and index data
    for model in models:
        filename = f'{dataset}_{model}_results_abs.txt'
        data = load_data(filename, model, dataset)
        for threshold in data.keys():
            if threshold not in thresholds_index:
                thresholds_index[threshold] = len(thresholds_index)

    # Calculate the group width with additional space
    group_width = bar_width * n_bars + group_spacing

    r_base = np.arange(len(thresholds_index)) * group_width  # Adjusted base positions for bars
    offsets = np.linspace(-bar_width * (n_bars / 2), bar_width * (n_bars / 2), n_bars)

    for i, model in enumerate(models):
        filename = f'{dataset}_{model}_results_abs.txt'
        data = load_data(filename, model, dataset)
        ratios = [data[t] for t in sorted(data.keys())]
        bar_positions = r_base + offsets[i]
        ax.bar(bar_positions, ratios, color=color_map[model], width=bar_width, label=model_full_names[model],
               edgecolor='black')  # Added edgecolor attribute
    if dataset == 'nrel_full_c':
        ax.set_xlabel('Error Threshold', fontsize=label_fontsize)
    elif dataset == 'smart_meter_full':
        ax.set_xlabel('Error Threshold (kWh)', fontsize=label_fontsize)
    else:
        ax.set_xlabel('Error Threshold (mph)', fontsize=label_fontsize)
    ax.set_ylabel('Compression Ratio', fontsize=ylabel_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xticks(r_base + offsets[int(n_bars/2)])  # Center x-ticks in the middle of the group
    if dataset == 'nrel_full_c':
        ax.set_xticklabels([f'{t:.4f}' for t in sorted(thresholds_index.keys())], fontsize=tick_fontsize)
    else:
        ax.set_xticklabels([f'{t:.2f}' for t in sorted(thresholds_index.keys())], fontsize=tick_fontsize)
    ax.set_ylim(0, 1.6)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=legend_fontsize, ncol=4)

    plt.tight_layout()
    plt.savefig(f'{dataset}_compression_results.png', dpi=500)
    plt.show()

# Generate plots for each dataset
for dataset in datasets:
    plot_data_for_dataset(dataset)
