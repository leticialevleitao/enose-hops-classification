"""
Hop Variety Classification Experiments for IJCNN 2026 Paper.

Loads consolidated hop e-nose data, trains 14 models (9 classical + 5 time series),
collects metrics, generates visualizations (PCA, t-SNE, confusion matrices).

Usage:
    python run_hop_experiments.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

from enose.data_loader import DataLoadingWorker, SENSOR_COLUMNS, COLUMNS_NAME
from enose.preprocessor import ENosePreprocessor
from enose.models.classical_models import ClassicalModels
from enose.models.time_series_models import TimeSeriesModels

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)

CONSOLIDATED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'consolidated')

# 9 classes: variety + condition (only top-level consolidated files)
CLASS_FILES = {
    'Chinook_Fresh':  'chinook_fresco.csv',
    'Chinook_Aged':   'chinook_passada.csv',
    'Comet_2025':     'come_2025_2.csv',
    'Saaz_2005':      'saaz_2005.csv',
    'Saaz_Fresh':     'saaz_fresco.csv',
    'Saaz_Aged':      'saaz_passado.csv',
    'Saaz_Dry':       'saaz_seco.csv',
    'Vista_Fresh':    'vista_fresco.csv',
    'Zeus_2025':      'zeus_2025_2.csv',
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

SPLIT_RATIOS = (0.70, 0.15, 0.15)
NORMALIZATION = 'minmax'
FLATTEN_METHOD = 'curve'

CLASSICAL_MODELS = [
    'Random Forest', 'SVM', 'k-NN', 'MLP', 'Decision Tree',
    'LDA', 'QDA', 'Logistic Regression', 'Naive Bayes'
]

TS_MODELS = [
    'RocketClassifier', 'TimeSeriesForestClassifier',
    'SignatureClassifier', 'InceptionTimeClassifier',
    'KNeighborsTimeSeriesClassifier'
]


# ============================================================
# Monkey-patch: Fix search spaces for 4 broken models
# ============================================================
def patch_classical_models_search_spaces(cm_instance, n_train):
    """
    Fix known bugs in ClassicalModels search spaces:
    - k-NN: n_neighbors upper bound exceeds n_samples_fit
    - MLP: hidden_layer_sizes as tuple causes conversion error
    - LDA: shrinkage incompatible with solver 'svd'
    - Logistic Regression: 'none' (string) vs None (NoneType)
    """
    from skopt.space import Real, Categorical, Integer

    # k-NN: cap n_neighbors to fit within 3-fold CV (each fold uses 2/3 of data)
    max_k = max(2, min(50, int(n_train * 2 / 3) - 1))
    cm_instance.search_spaces['k-NN'] = {
        'n_neighbors': Integer(1, max_k),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 2)
    }

    # MLP: fix hidden_layer_sizes (tuples break skopt Categorical),
    # so we fix it at (100,) and only optimize activation + alpha
    cm_instance.search_spaces['MLP'] = {
        'activation': Categorical(['relu', 'tanh']),
        'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
    }

    # LDA: use lsqr only (eigen fails on this dataset) with shrinkage
    cm_instance.search_spaces['LDA'] = {
        'solver': Categorical(['lsqr']),
        'shrinkage': Categorical(['auto', 0.1, 0.5, 0.9])
    }

    # Logistic Regression: use None instead of 'none'
    cm_instance.search_spaces['Logistic Regression'] = {
        'C': Real(1e-4, 1e4, prior='log-uniform'),
        'penalty': Categorical(['l2', None]),
        'solver': Categorical(['lbfgs', 'saga'])
    }

    print("  Search spaces patched for k-NN, MLP, LDA, Logistic Regression")


# ============================================================
# Data Loading (reuses data_loader cycle-parsing logic)
# ============================================================
def load_consolidated_data(class_files, data_dir, sensors=None):
    """Load consolidated CSVs, parse cycles, return arrays + labels."""
    if sensors is None:
        sensors = SENSOR_COLUMNS

    all_cycles = []
    all_labels = []

    for class_name, filename in class_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping {class_name}")
            continue

        with open(filepath, 'r') as f:
            current_cycle = []
            in_cycle = False
            for line in f:
                if '_CICLO_INICIADO_' in line:
                    in_cycle = True
                    current_cycle = []
                    continue
                if '_CICLO_FINALIZADO_' in line:
                    in_cycle = False
                    if current_cycle:
                        all_cycles.append(np.array(current_cycle))
                        all_labels.append(class_name)
                    continue
                if in_cycle and len(line.strip()) > 1:
                    try:
                        values = [float(s) for s in line.split()]
                        sensor_values = values[3:3 + len(SENSOR_COLUMNS)]
                        if len(sensor_values) == len(SENSOR_COLUMNS):
                            current_cycle.append(sensor_values)
                    except (ValueError, IndexError):
                        continue

        print(f"  {class_name}: loaded from {filename}")

    if not all_cycles:
        raise ValueError("No valid cycles found!")

    # Standardize length (truncate to min)
    cycle_lengths = [len(c) for c in all_cycles]
    min_len = min(cycle_lengths)
    max_len = max(cycle_lengths)
    print(f"\nCycle lengths: min={min_len}, max={max_len}, mean={np.mean(cycle_lengths):.1f}")
    print(f"Truncating all to {min_len} time steps")

    # Filter sensors
    sensor_indices = [i for i, name in enumerate(SENSOR_COLUMNS) if name in sensors]

    standardized = []
    for cycle in all_cycles:
        truncated = cycle[:min_len, :]
        filtered = truncated[:, sensor_indices]
        signature = filtered.flatten(order='F')  # Column-major: sensor-by-sensor
        standardized.append(signature)

    X = np.vstack(standardized)
    y = np.array(all_labels)

    points_per_sensor = min_len
    selected_sensors = [SENSOR_COLUMNS[i] for i in sensor_indices]

    summary = {
        'all_data': X,
        'all_classes': y,
        'statistics': {
            'Total de Amostras (Ciclos)': len(X),
            'Sensores Utilizados': selected_sensors,
            'Comprimento Mínimo dos Ciclos': int(min_len),
            'Comprimento Máximo dos Ciclos': int(max_len),
            'Tamanho Final da Assinatura': X.shape[1],
        }
    }

    return summary


# ============================================================
# Training
# ============================================================
def train_classical_models(data, data_summary, selected_sensors, classes, use_opt=True):
    """Train all 9 classical models, return results dict with timing."""
    cm = ClassicalModels()

    # Patch search spaces to fix known bugs
    n_train = len(data['y_train'])
    patch_classical_models_search_spaces(cm, n_train)

    results = {}

    for model_name in CLASSICAL_MODELS:
        print(f"\n--- Training {model_name} ---")
        t_start = time.time()
        try:
            result = cm.train_model(
                model_name=model_name,
                data=data,
                data_summary=data_summary,
                selected_sensors=selected_sensors,
                selected_classes=classes,
                hyperparams={},
                use_optimization=use_opt
            )
            t_elapsed = time.time() - t_start
            result['train_time'] = t_elapsed
            results[model_name] = result
            m = result['metrics']
            print(f"  Acc={m.get('test_accuracy', 0):.4f}  F1={m.get('f1', 0):.4f}  G-mean={m.get('gmean', 0):.4f}  Time={t_elapsed:.1f}s")
        except Exception as e:
            t_elapsed = time.time() - t_start
            print(f"  ERROR: {e}")
            results[model_name] = {'metrics': {}, 'error': str(e), 'train_time': t_elapsed}

    return results


def train_ts_models(data, data_summary, selected_sensors, classes):
    """Train all 5 time series models, return results dict with timing."""
    tsm = TimeSeriesModels()
    available = tsm.get_available_models()
    results = {}

    for model_name in TS_MODELS:
        if model_name not in available:
            print(f"\n--- {model_name}: NOT AVAILABLE (missing dependency) ---")
            results[model_name] = {'metrics': {}, 'error': 'not available', 'train_time': 0}
            continue

        print(f"\n--- Training {model_name} ---")
        t_start = time.time()
        try:
            result = tsm.train_model(
                model_name=model_name,
                data=data,
                data_summary=data_summary,
                selected_sensors=selected_sensors,
                selected_classes=classes,
                hyperparams={},
                use_optimization=False
            )
            t_elapsed = time.time() - t_start
            result['train_time'] = t_elapsed
            results[model_name] = result
            m = result['metrics']
            print(f"  Acc={m.get('test_accuracy', 0):.4f}  F1={m.get('f1', 0):.4f}  G-mean={m.get('gmean', 0):.4f}  Time={t_elapsed:.1f}s")
        except Exception as e:
            t_elapsed = time.time() - t_start
            print(f"  ERROR: {e}")
            results[model_name] = {'metrics': {}, 'error': str(e), 'train_time': t_elapsed}

    return results


# ============================================================
# Visualization
# ============================================================
def plot_pca(X, y, output_path):
    """PCA 2D scatter plot with distinct markers per class."""
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']

    for i, (cls, color) in enumerate(zip(classes, colors)):
        mask = y == cls
        marker = markers[i % len(markers)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=cls,
                   marker=marker, alpha=0.7, edgecolors='k', linewidth=0.5, s=60)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.set_title('PCA Projection of E-Nose Signatures')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved PCA plot: {output_path}")
    return pca.explained_variance_ratio_


def plot_tsne(X, y, output_path):
    """t-SNE 2D scatter plot."""
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for cls, color in zip(classes, colors):
        mask = y == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], label=cls,
                   alpha=0.7, edgecolors='k', linewidth=0.5, s=60)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.set_title('t-SNE Projection of E-Nose Signatures')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved t-SNE plot: {output_path}")


def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_path):
    """Normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved confusion matrix: {output_path}")


def plot_r2_sensors(r2_scores, output_path):
    """Bar chart of sensor discriminatory power (R^2)."""
    sensors = list(r2_scores.keys())
    values = list(r2_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sensors, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sensors))))
    ax.set_xlabel('R^2 Score')
    ax.set_title('Sensor Discriminatory Power')
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved R^2 plot: {output_path}")


def plot_model_comparison(all_results, output_path):
    """Bar chart comparing all models by accuracy, F1, G-mean."""
    models = []
    accs = []
    f1s = []
    gmeans = []

    for name, res in all_results.items():
        m = res.get('metrics', {})
        if m.get('test_accuracy') is not None:
            models.append(name)
            accs.append(m['test_accuracy'])
            f1s.append(m.get('f1', 0))
            gmeans.append(m.get('gmean', 0))

    if not models:
        return

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, accs, width, label='Accuracy', color='#2196F3')
    ax.bar(x, f1s, width, label='F1-Score', color='#4CAF50')
    ax.bar(x + width, gmeans, width, label='G-Mean', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved model comparison: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("HOP VARIETY CLASSIFICATION - IJCNN 2026 EXPERIMENTS")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading consolidated hop data...")
    data_summary = load_consolidated_data(CLASS_FILES, CONSOLIDATED_DIR)
    X_raw = data_summary['all_data']
    y = data_summary['all_classes']
    classes = sorted(np.unique(y).tolist())
    sensors = data_summary['statistics']['Sensores Utilizados']

    print(f"\nTotal samples: {len(y)}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Sensors ({len(sensors)}): {sensors}")
    print(f"Signature size: {X_raw.shape[1]}")
    for cls in classes:
        print(f"  {cls}: {np.sum(y == cls)} cycles")

    # 2. Preprocess
    print("\n[2] Preprocessing...")
    preprocessor = ENosePreprocessor()
    processed = preprocessor.process(data_summary, NORMALIZATION, SPLIT_RATIOS, FLATTEN_METHOD)

    data = {
        'X_train': processed['X_train'],
        'y_train': processed['y_train'],
        'X_val': processed['X_val'],
        'y_val': processed['y_val'],
        'X_test': processed['X_test'],
        'y_test': processed['y_test'],
    }

    print(f"  Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")

    r2_scores = processed.get('r2_scores', {})
    print(f"\n  Sensor R^2 scores:")
    for sensor, r2 in r2_scores.items():
        print(f"    {sensor}: {r2:.4f}")

    # 3. Visualizations (on full normalized data)
    print("\n[3] Generating visualizations...")
    X_norm = processed['X_full_norm']
    y_full = processed['y_full']

    pca_var = plot_pca(X_norm, y_full, os.path.join(FIGURES_DIR, 'pca_projection.pdf'))
    plot_tsne(X_norm, y_full, os.path.join(FIGURES_DIR, 'tsne_projection.pdf'))
    if r2_scores:
        plot_r2_sensors(r2_scores, os.path.join(FIGURES_DIR, 'sensor_r2.pdf'))

    # 4. Train classical models
    print("\n[4] Training classical models (with Bayesian optimization)...")
    t0 = time.time()
    classical_results = train_classical_models(data, data_summary, sensors, classes, use_opt=True)
    print(f"\n  Classical models done in {time.time()-t0:.1f}s")

    # 5. Train time series models
    print("\n[5] Training time series models...")
    t0 = time.time()
    ts_results = train_ts_models(data, data_summary, sensors, classes)
    print(f"\n  Time series models done in {time.time()-t0:.1f}s")

    # 6. Compile results
    all_results = {**classical_results, **ts_results}

    print("\n[6] Compiling results table...")
    rows = []
    for name, res in all_results.items():
        m = res.get('metrics', {})
        rows.append({
            'Model': name,
            'Train Acc': m.get('train_accuracy', None),
            'Val Acc': m.get('val_accuracy', None),
            'Test Acc': m.get('test_accuracy', None),
            'Precision': m.get('precision', None),
            'Recall': m.get('recall', None),
            'F1': m.get('f1', None),
            'G-Mean': m.get('gmean', None),
            'Optimized': res.get('used_optimization', False),
            'Best Params': str(res.get('best_params', {})),
            'Train Time': res.get('train_time', None),
            'Error': res.get('error', ''),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Test Acc', ascending=False, na_position='last')

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df[['Model', 'Test Acc', 'F1', 'G-Mean', 'Train Time']].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'results_all_models.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    # 7. Best model confusion matrix
    valid_results = {k: v for k, v in all_results.items()
                     if v.get('metrics', {}).get('test_accuracy') is not None}
    if valid_results:
        best_name = max(valid_results, key=lambda k: valid_results[k]['metrics']['test_accuracy'])
        best_res = valid_results[best_name]
        print(f"\n  Best model: {best_name} (Acc={best_res['metrics']['test_accuracy']:.4f})")

        # Get predictions for confusion matrix
        y_test = best_res.get('y_test', data['y_test'])
        y_pred = best_res.get('y_pred')

        if y_pred is not None:
            plot_confusion_matrix(y_test, y_pred, classes, best_name,
                                  os.path.join(FIGURES_DIR, 'confusion_matrix_best.pdf'))

        # Also plot comparison
        plot_model_comparison(all_results, os.path.join(FIGURES_DIR, 'model_comparison.pdf'))

    # 8. Save detailed results as JSON
    json_results = {}
    for name, res in all_results.items():
        m = res.get('metrics', {})
        json_results[name] = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in m.items()},
            'best_params': str(res.get('best_params', {})),
            'used_optimization': res.get('used_optimization', False),
            'train_time': res.get('train_time', None),
            'error': res.get('error', ''),
        }

    json_path = os.path.join(OUTPUT_DIR, 'results_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    # Save dataset info
    dataset_info = {
        'classes': classes,
        'n_samples': int(len(y)),
        'samples_per_class': {cls: int(np.sum(y == cls)) for cls in classes},
        'n_sensors': len(sensors),
        'sensors': sensors,
        'signature_size': int(X_raw.shape[1]),
        'split_ratios': list(SPLIT_RATIOS),
        'normalization': NORMALIZATION,
        'flatten_method': FLATTEN_METHOD,
        'r2_scores': {k: float(v) for k, v in r2_scores.items()},
        'pca_variance_explained': [float(v) for v in pca_var[:2]] if pca_var is not None else [],
        'train_size': int(len(data['y_train'])),
        'val_size': int(len(data['y_val'])),
        'test_size': int(len(data['y_test'])),
    }
    info_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n  Detailed results: {json_path}")
    print(f"  Dataset info: {info_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
