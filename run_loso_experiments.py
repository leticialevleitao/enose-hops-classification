"""
LOSO (Leave-One-Session-Out) experiment runner for hop variety classification.

Compares the 9 classical ML models under a session-aware evaluation protocol.
Each fold holds out one complete measurement session (5 cycles from the same
physical sample on the same day) as the test set.

Why LOSO instead of random 70/15/15?
    The existing pipeline splits cycles randomly, putting cycles from the SAME
    measurement session in both train and test. Because these cycles are highly
    correlated (same sample, same day), this inflates accuracy. LOSO prevents
    this leakage and answers the scientifically meaningful question:
    "Can the model classify a hop sample it has never measured before?"

Usage:
    python run_loso_experiments.py
"""

import os
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from enose.session_loader import load_with_sessions
from enose.loso_cv import run_loso

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)

CONSOLIDATED_DIR = os.path.join(os.path.dirname(__file__), "data", "consolidated")
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR      = os.path.join(os.path.dirname(__file__), "figures")

# Same 9-class cone dataset as the IJCNN paper
CLASS_FILES = {
    "Chinook_Fresh": "chinook_fresco.csv",
    "Chinook_Aged":  "chinook_passada.csv",
    "Comet_2025":    "come_2025_2.csv",
    "Saaz_2005":     "saaz_2005.csv",
    "Saaz_Fresh":    "saaz_fresco.csv",
    "Saaz_Aged":     "saaz_passado.csv",
    "Saaz_Dry":      "saaz_seco.csv",
    "Vista_Fresh":   "vista_fresco.csv",
    "Zeus_2025":     "zeus_2025_2.csv",
}

NORMALIZATION  = "minmax"
FLATTEN_METHOD = "curve"

# ============================================================
# Model factories
# Each entry is (display_name, factory_function).
# A factory is called once per LOSO fold to produce a fresh model instance.
# ============================================================
def make_models():
    # Use moderate, well-generalizing defaults suitable for LOSO evaluation.
    # The highly-tuned params (e.g. C=1e6) from the random-split paper are
    # deliberately NOT reused here: they were optimised against a leaky split
    # and tend to overfit, causing extremely long convergence in some folds.
    return {
        "SVM":                lambda: SVC(kernel="rbf", C=10, gamma="scale",
                                          max_iter=5000, random_state=SEED),
        "Random Forest":      lambda: RandomForestClassifier(n_estimators=200,
                                                              random_state=SEED),
        "MLP":                lambda: MLPClassifier(hidden_layer_sizes=(100,),
                                                    activation="relu", alpha=1e-3,
                                                    max_iter=500, random_state=SEED),
        "Decision Tree":      lambda: DecisionTreeClassifier(max_depth=10,
                                                              random_state=SEED),
        "LDA":                lambda: LinearDiscriminantAnalysis(solver="lsqr",
                                                                  shrinkage="auto"),
        "QDA":                lambda: QuadraticDiscriminantAnalysis(),
        "Logistic Regression":lambda: LogisticRegression(C=1.0, penalty="l2",
                                                          solver="lbfgs", max_iter=2000,
                                                          random_state=SEED),
        "Naive Bayes":        lambda: GaussianNB(),
        "k-NN":               lambda: KNeighborsClassifier(n_neighbors=3,
                                                            weights="distance"),
    }


# ============================================================
# Visualization helpers
# ============================================================
def plot_loso_summary(summary_df, output_path):
    """Horizontal bar chart: mean accuracy ± std per model, sorted by accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = summary_df.dropna(subset=["mean_accuracy"]).sort_values("mean_accuracy")

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df["mean_accuracy"], xerr=df["std_accuracy"],
            capsize=4, color="steelblue", alpha=0.8, error_kw={"elinewidth": 1.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_xlabel("Mean Accuracy (LOSO, ±1 std)")
    ax.set_title("Leave-One-Session-Out — Model Comparison\n(9-class hop cone dataset)")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=df["mean_accuracy"].max(), color="red", linestyle="--",
               linewidth=1, alpha=0.6, label=f"Best: {df['mean_accuracy'].max():.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrix_loso(cm, classes, model_name, output_path):
    """Normalized confusion matrix from aggregated LOSO predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (LOSO) — {model_name}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_session_accuracy(all_results, output_path):
    """
    Heatmap: sessions (rows) × models (columns), coloured by per-fold accuracy.
    Reveals which sessions are consistently hard to classify.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Collect per-fold accuracy for each model
    model_names = []
    session_accs = {}  # {session_id: {model: acc}}

    for model_name, result in all_results.items():
        if "error" in result or not result.get("fold_results"):
            continue
        model_names.append(model_name)
        for fold in result["fold_results"]:
            sid = fold["session"]
            if sid not in session_accs:
                session_accs[sid] = {}
            session_accs[sid][model_name] = fold["accuracy"]

    if not model_names or not session_accs:
        return

    sessions_sorted = sorted(session_accs.keys())
    matrix = np.full((len(sessions_sorted), len(model_names)), np.nan)
    for r, sid in enumerate(sessions_sorted):
        for c, mname in enumerate(model_names):
            matrix[r, c] = session_accs[sid].get(mname, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.2), max(6, len(sessions_sorted) * 0.4)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=model_names, yticklabels=sessions_sorted,
                ax=ax, vmin=0, vmax=1, linewidths=0.3)
    ax.set_title("Per-Session Accuracy Heatmap (LOSO)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Held-out Session")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 65)
    print("HOP VARIETY CLASSIFICATION — LEAVE-ONE-SESSION-OUT EVALUATION")
    print("=" * 65)

    # 1. Load data with session tracking
    print("\n[1/3] Loading data with session annotations...")
    data = load_with_sessions(CLASS_FILES, CONSOLIDATED_DIR)

    print(f"\nSession structure:")
    for cls, sessions in data["sessions_per_class"].items():
        print(f"  {cls:20s}: {len(sessions)} sessions — {sessions}")

    # 2. Run LOSO for each model
    print("\n[2/3] Running LOSO cross-validation...")
    models = make_models()
    all_results = {}

    rows = []
    for model_name, factory in models.items():
        print(f"\n--- {model_name} ---")
        t_start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_loso(
                    data=data,
                    model_factory=factory,
                    normalization=NORMALIZATION,
                    flatten_method=FLATTEN_METHOD,
                    verbose=True,
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"error": str(e)}

        elapsed = time.time() - t_start
        result["train_time"] = elapsed
        all_results[model_name] = result

        if "error" not in result:
            rows.append({
                "model":            model_name,
                "mean_accuracy":    result["mean_accuracy"],
                "std_accuracy":     result["std_accuracy"],
                "overall_accuracy": result["overall_accuracy"],
                "mean_f1":          result["mean_f1"],
                "std_f1":           result["std_f1"],
                "overall_f1":       result["overall_f1"],
                "overall_gmean":    result["overall_gmean"],
                "n_folds":          result["n_folds"],
                "skipped_folds":    len(result.get("skipped_folds", [])),
                "train_time_s":     round(elapsed, 2),
            })
        else:
            rows.append({
                "model": model_name,
                "mean_accuracy": None,
                "error": result["error"],
            })

    # 3. Save results and generate figures
    print("\n[3/3] Saving results and generating figures...")

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, "loso_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    detailed_path = os.path.join(OUTPUT_DIR, "loso_results_detailed.json")
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {detailed_path}")

    # Summary bar chart
    plot_loso_summary(summary_df, os.path.join(FIGURES_DIR, "loso_model_comparison.pdf"))

    # Per-session heatmap
    plot_per_session_accuracy(all_results, os.path.join(FIGURES_DIR, "loso_session_heatmap.pdf"))

    # Confusion matrix for best model
    valid = summary_df.dropna(subset=["mean_accuracy"])
    if not valid.empty:
        best_model = valid.loc[valid["overall_accuracy"].idxmax(), "model"]
        best_result = all_results[best_model]
        if "confusion_matrix" in best_result:
            plot_confusion_matrix_loso(
                best_result["confusion_matrix"],
                best_result["classes"],
                best_model,
                os.path.join(FIGURES_DIR, f"loso_cm_{best_model.replace(' ', '_')}.pdf"),
            )

    # Print final ranking
    print("\n" + "=" * 65)
    print("FINAL RANKING (LOSO — overall accuracy on all concatenated folds)")
    print("=" * 65)
    valid_sorted = valid.sort_values("overall_accuracy", ascending=False)
    print(f"\n{'Model':<22} {'Mean±Std':>12} {'Overall Acc':>12} {'F1':>8} {'G-Mean':>8}")
    print("-" * 65)
    for _, row in valid_sorted.iterrows():
        print(
            f"  {row['model']:<20} "
            f"{row['mean_accuracy']:.3f}±{row['std_accuracy']:.3f}  "
            f"{row['overall_accuracy']:>11.3f}  "
            f"{row['overall_f1']:>7.3f}  "
            f"{row['overall_gmean']:>7.3f}"
        )

    print("\nDone. Results in results/ and figures/")


if __name__ == "__main__":
    main()
