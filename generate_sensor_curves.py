"""
Generate sensor response curves figure for IJCNN 2026 paper.

Plots mean response curves per class for each sensor over time steps.
Output: figures/sensor_response_curves.pdf
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from enose.data_loader import SENSOR_COLUMNS
from enose.preprocessor import ENosePreprocessor
from run_hop_experiments import load_consolidated_data, CLASS_FILES, CONSOLIDATED_DIR, SPLIT_RATIOS, NORMALIZATION, FLATTEN_METHOD

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    data_summary = load_consolidated_data(CLASS_FILES, CONSOLIDATED_DIR)
    X_raw = data_summary['all_data']
    y = data_summary['all_classes']
    sensors = data_summary['statistics']['Sensores Utilizados']
    sig_size = data_summary['statistics']['Tamanho Final da Assinatura']
    points_per_sensor = sig_size // len(sensors)

    # Preprocess
    preprocessor = ENosePreprocessor()
    processed = preprocessor.process(data_summary, NORMALIZATION, SPLIT_RATIOS, FLATTEN_METHOD)
    X_norm = processed['X_full_norm']
    y_full = processed['y_full']

    unique_classes = sorted(np.unique(y_full))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    time_steps = np.arange(points_per_sensor)

    n_sensors = len(sensors)
    ncols = 2
    nrows = (n_sensors + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.8 * nrows), sharex=True)
    axes = axes.flatten()

    for sensor_idx, sensor_name in enumerate(sensors):
        ax = axes[sensor_idx]
        start = sensor_idx * points_per_sensor
        end = start + points_per_sensor
        sensor_data = X_norm[:, start:end]

        for class_idx, class_label in enumerate(unique_classes):
            class_data = sensor_data[y_full == class_label]
            if class_data.shape[0] == 0:
                continue
            mean_curve = class_data.mean(axis=0)
            std_curve = class_data.std(axis=0)
            ax.plot(time_steps, mean_curve, color=colors[class_idx], linewidth=0.8)
            ax.fill_between(time_steps, mean_curve - std_curve, mean_curve + std_curve,
                            color=colors[class_idx], alpha=0.1)

        ax.set_title(sensor_name, fontsize=8, fontweight='bold')
        ax.set_ylabel('Normalized', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot if odd number of sensors
    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)

    # Shared x-label
    for ax in axes[max(0, nrows * ncols - ncols):nrows * ncols]:
        if ax.get_visible():
            ax.set_xlabel('Time Step', fontsize=7)

    # Legend in the empty subplot area or below
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=1.5,
                              label=c.replace('_', ' ')) for i, c in enumerate(unique_classes)]

    if n_sensors < len(axes):
        axes[n_sensors].set_visible(True)
        axes[n_sensors].axis('off')
        axes[n_sensors].legend(handles=legend_elements, loc='center',
                               fontsize=6, ncol=2, frameon=False)
    else:
        fig.legend(handles=legend_elements, loc='lower center',
                   fontsize=6, ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'sensor_response_curves.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
