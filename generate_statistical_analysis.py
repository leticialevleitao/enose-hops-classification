"""
Statistical analysis for IJCNN 2026 paper.

Since we have a single train/test split (not multiple datasets), we use:
1. Bootstrap confidence intervals on the test set predictions
2. McNemar's test for pairwise model comparisons
3. A summary ranking diagram

Output: figures/bootstrap_ci.pdf, statistical results printed to console
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

from enose.data_loader import SENSOR_COLUMNS
from enose.preprocessor import ENosePreprocessor
from enose.models.classical_models import ClassicalModels
from enose.models.time_series_models import TimeSeriesModels
from run_hop_experiments import (
    load_consolidated_data, CLASS_FILES, CONSOLIDATED_DIR,
    SPLIT_RATIOS, NORMALIZATION, FLATTEN_METHOD,
    CLASSICAL_MODELS, TS_MODELS, patch_classical_models_search_spaces,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
N_BOOTSTRAP = 1000


def bootstrap_accuracy(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        acc = np.mean(y_true[idx] == y_pred[idx])
        accs.append(acc)
    accs = np.array(accs)
    return np.mean(accs), np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test comparing two classifiers. Returns chi2 statistic and p-value."""
    from scipy.stats import chi2
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # b: A correct, B wrong; c: A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0
    # McNemar with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    data_summary = load_consolidated_data(CLASS_FILES, CONSOLIDATED_DIR)
    X_raw = data_summary['all_data']
    y = data_summary['all_classes']
    classes = sorted(np.unique(y).tolist())
    sensors = data_summary['statistics']['Sensores Utilizados']

    preprocessor = ENosePreprocessor()
    processed = preprocessor.process(data_summary, NORMALIZATION, SPLIT_RATIOS, FLATTEN_METHOD)

    data = {
        'X_train': processed['X_train'], 'y_train': processed['y_train'],
        'X_val': processed['X_val'], 'y_val': processed['y_val'],
        'X_test': processed['X_test'], 'y_test': processed['y_test'],
    }

    # Train all models and collect predictions
    print("\nTraining classical models...")
    cm = ClassicalModels()
    n_train = len(data['y_train'])
    patch_classical_models_search_spaces(cm, n_train)

    predictions = {}
    for model_name in CLASSICAL_MODELS:
        print(f"  {model_name}...", end=" ")
        try:
            result = cm.train_model(
                model_name=model_name, data=data, data_summary=data_summary,
                selected_sensors=sensors, selected_classes=classes,
                hyperparams={}, use_optimization=True
            )
            if result.get('y_pred') is not None:
                predictions[model_name] = result['y_pred']
                acc = result['metrics'].get('test_accuracy', 0)
                print(f"OK (acc={acc:.2f})")
            else:
                print("no predictions")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nTraining time series models...")
    tsm = TimeSeriesModels()
    available = tsm.get_available_models()
    for model_name in TS_MODELS:
        if model_name not in available:
            print(f"  {model_name}: not available")
            continue
        print(f"  {model_name}...", end=" ")
        try:
            result = tsm.train_model(
                model_name=model_name, data=data, data_summary=data_summary,
                selected_sensors=sensors, selected_classes=classes,
                hyperparams={}, use_optimization=False
            )
            if result.get('y_pred') is not None:
                predictions[model_name] = result['y_pred']
                acc = result['metrics'].get('test_accuracy', 0)
                print(f"OK (acc={acc:.2f})")
            else:
                print("no predictions")
        except Exception as e:
            print(f"FAILED: {e}")

    y_test = data['y_test']

    # 1. Bootstrap confidence intervals
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    ci_results = {}
    for name, y_pred in sorted(predictions.items(), key=lambda x: np.mean(x[1] == y_test), reverse=True):
        mean_acc, ci_low, ci_high = bootstrap_accuracy(y_test, y_pred)
        ci_results[name] = (mean_acc, ci_low, ci_high)
        print(f"  {name:30s}: {mean_acc:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # 2. Plot bootstrap CI
    if ci_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(ci_results.keys())
        means = [ci_results[n][0] for n in names]
        lows = [ci_results[n][1] for n in names]
        highs = [ci_results[n][2] for n in names]
        errors = [[m - l for m, l in zip(means, lows)],
                  [h - m for m, h in zip(means, highs)]]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=errors, capsize=3, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Accuracy (95% Bootstrap CI)')
        ax.set_title('Model Accuracy with Bootstrap Confidence Intervals')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        fig.tight_layout()
        ci_path = os.path.join(FIGURES_DIR, 'bootstrap_ci.pdf')
        fig.savefig(ci_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved: {ci_path}")

    # 3. McNemar's tests (pairwise)
    print("\n" + "=" * 60)
    print("McNEMAR'S PAIRWISE TESTS")
    print("=" * 60)
    model_names = list(predictions.keys())
    mcnemar_results = []
    for a, b in combinations(model_names, 2):
        chi2_stat, p_val = mcnemar_test(y_test, predictions[a], predictions[b])
        sig = "*" if p_val < 0.05 else ""
        mcnemar_results.append({'Model A': a, 'Model B': b, 'Chi2': chi2_stat, 'p-value': p_val, 'Sig': sig})
        if p_val < 0.05:
            print(f"  {a} vs {b}: chi2={chi2_stat:.3f}, p={p_val:.4f} *")

    if not any(r['p-value'] < 0.05 for r in mcnemar_results):
        print("  No statistically significant pairwise differences found (p < 0.05).")
        print("  (This is expected with small test sets where many models perform similarly.)")

    # Save statistical results
    stats_path = os.path.join(OUTPUT_DIR, 'statistical_results.json')
    stats_output = {
        'bootstrap_ci': {name: {'mean': float(m), 'ci_low': float(l), 'ci_high': float(h)}
                         for name, (m, l, h) in ci_results.items()},
        'mcnemar_tests': mcnemar_results,
        'n_bootstrap': N_BOOTSTRAP,
        'test_size': int(len(y_test)),
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2, default=str)
    print(f"\n  Statistical results saved: {stats_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
