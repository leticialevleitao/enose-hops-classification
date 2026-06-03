"""
Modern-models LOSO experiment for the hop e-nose TCC.

Extends the baseline LOSO pipeline (run_loso_experiments.py) with:
  Tabular boosting / modern tabular:
      XGBoost, LightGBM, CatBoost, TabPFN, RealTabPFN, AutoGluon
  Time-series:
      RocketClassifier, Arsenal, RISE  (sktime)

Results are stored alongside the classical LOSO results so the final ranking
table covers ALL 14 models (9 classical + 5 modern).

Usage:
    cd "<repo root>"
    python run_loso_modern.py > results/loso_modern_log.txt 2>&1
"""

import os
import sys
import json
import time
import tempfile
import shutil
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from enose.session_loader import load_with_sessions
from enose.loso_cv import run_loso, flatten_cycles, _compute_gmean

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data", "consolidated")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
RES_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

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

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Label-encoding wrapper (XGBoost requires integer labels, not strings)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class _LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper that encodes string labels → int before fit/predict."""
    def __init__(self, clf):
        self.clf = clf
        self.le_ = None

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.clf.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.clf.predict(X)
        return self.le_.inverse_transform(y_enc.astype(int))

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# ─────────────────────────────────────────────────────────────────────────────
# AutoGluon sklearn-like wrapper
# ─────────────────────────────────────────────────────────────────────────────
class _AutoGluonTabularClassifier(BaseEstimator, ClassifierMixin):
    """
    Lightweight sklearn-compatible wrapper around AutoGluon TabularPredictor.
    """
    def __init__(self, presets="medium_quality", time_limit=120, verbosity=0):
        self.presets = presets
        self.time_limit = time_limit
        self.verbosity = verbosity
        self.predictor_ = None
        self.feature_cols_ = None
        self.path_ = None

    def fit(self, X, y):
        from autogluon.tabular import TabularPredictor

        self.feature_cols_ = [f"f{i}" for i in range(X.shape[1])]
        train_df = pd.DataFrame(X, columns=self.feature_cols_)
        train_df["target"] = np.asarray(y).astype(str)

        self.path_ = tempfile.mkdtemp(prefix="ag_loso_")
        self.predictor_ = TabularPredictor(
            label="target",
            problem_type="multiclass",
            eval_metric="accuracy",
            path=self.path_,
            verbosity=self.verbosity,
        )
        self.predictor_.fit(
            train_data=train_df,
            presets=self.presets,
            time_limit=self.time_limit,
        )
        return self

    def predict(self, X):
        test_df = pd.DataFrame(X, columns=self.feature_cols_)
        pred = self.predictor_.predict(test_df)
        return pred.to_numpy() if hasattr(pred, "to_numpy") else np.array(pred)

    def predict_proba(self, X):
        test_df = pd.DataFrame(X, columns=self.feature_cols_)
        proba = self.predictor_.predict_proba(test_df)
        return proba.to_numpy() if hasattr(proba, "to_numpy") else np.array(proba)

    def __del__(self):
        # Best-effort cleanup of temp model directory.
        try:
            if self.path_ and os.path.exists(self.path_):
                shutil.rmtree(self.path_, ignore_errors=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Tabular model factories  (sklearn-compatible API → use existing run_loso())
# ─────────────────────────────────────────────────────────────────────────────
def make_tabular_models():
    models = {}

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda: _LabelEncodedClassifier(XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=SEED,
            verbosity=0,
            n_jobs=-1,
        ))
    except Exception as e:
        print(f"WARNING: XGBoost unavailable: {e}")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
    except Exception as e:
        print(f"WARNING: LightGBM unavailable: {e}")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = lambda: CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=SEED,
            verbose=0,
        )
    except Exception as e:
        print(f"WARNING: CatBoost unavailable: {e}")

    # TabPFN (state-of-the-art tabular foundation model)
    try:
        from tabpfn import TabPFNClassifier
        models["TabPFN"] = lambda: TabPFNClassifier()
    except Exception as e:
        print(f"WARNING: TabPFN unavailable: {e}")

    # NOTE: tabpfn_extensions HPO variants (AutoTabPFNClassifier / TunedTabPFNClassifier)
    # only support TabPFN v2.5 weights and are deprecated. TabPFN v3 (above) is the
    # current state-of-the-art model and already covers the "RealTabPFN" use case
    # mentioned by the advisor.

    # AutoGluon (AutoML ensemble for tabular data)
    try:
        from autogluon.tabular import TabularPredictor  # noqa: F401
        models["AutoGluon"] = lambda: _AutoGluonTabularClassifier(
            presets="medium_quality_faster_train",
            time_limit=20,
            verbosity=0,
        )
    except Exception as e:
        print(f"WARNING: AutoGluon unavailable: {e}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Time-series LOSO runner (sktime expects (n, n_channels, n_timepoints))
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_ts(data, model_factory, normalization="minmax", verbose=True, model_name="model"):
    """
    LOSO for sktime classifiers.

    sktime format: X shape = (n_samples, n_channels, n_timepoints)
    Our data:      X_raw_3d shape = (n_cycles, time_steps, n_sensors)
    So we transpose to (n_cycles, n_sensors, time_steps) before passing to sktime.

    Normalization is applied per-sensor using training data statistics.
    """
    X_3d        = data["X_raw_3d"]      # (n_cycles, T, S)
    y           = data["y"]
    session_ids = data["session_ids"]
    classes     = data["classes"]

    loso = LeaveOneGroupOut()

    fold_results = []
    skipped_folds = []
    all_y_true = []
    all_y_pred = []

    n_folds = loso.get_n_splits(X_3d, y, groups=session_ids)
    if verbose:
        print(f"  Total sessions (folds): {n_folds}")
        print(f"  {'Fold':>4}  {'Session':<32}  {'n_test':>6}  {'Acc':>6}  {'F1':>6}")
        print("  " + "-" * 62)

    for fold_idx, (train_idx, test_idx) in enumerate(
        loso.split(X_3d, y, groups=session_ids)
    ):
        held_out = session_ids[test_idx[0]]

        X_train_3d = X_3d[train_idx]   # (n_train, T, S)
        X_test_3d  = X_3d[test_idx]    # (n_test,  T, S)
        y_train    = y[train_idx]
        y_test     = y[test_idx]

        # Skip folds where test class is absent from training set
        missing = set(np.unique(y_test)) - set(np.unique(y_train))
        if missing:
            warnings.warn(f"Fold {fold_idx} ({held_out}): {missing} absent from train — skipped.")
            skipped_folds.append(held_out)
            continue

        # Normalize per-sensor using training statistics
        N_tr, T, S = X_train_3d.shape
        if normalization == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Fit on all (time_step × sensor) pairs from training cycles → (N_tr*T, S)
        scaler.fit(X_train_3d.reshape(-1, S))
        X_train_norm = scaler.transform(X_train_3d.reshape(-1, S)).reshape(N_tr, T, S)
        X_test_norm  = scaler.transform(X_test_3d.reshape(-1, S)).reshape(len(test_idx), T, S)

        # Transpose to sktime format: (n, S, T)
        X_tr_sk = X_train_norm.transpose(0, 2, 1).astype(np.float32)
        X_te_sk = X_test_norm.transpose(0, 2, 1).astype(np.float32)

        clf = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr_sk, y_train)
            y_pred = clf.predict(X_te_sk)

        fold_acc = float(accuracy_score(y_test, y_pred))
        fold_f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        fold_results.append({
            "fold":     fold_idx,
            "session":  held_out,
            "n_test":   int(len(y_test)),
            "accuracy": fold_acc,
            "f1":       fold_f1,
            "y_true":   y_test.tolist(),
            "y_pred":   y_pred.tolist(),
        })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        if verbose:
            print(f"  {fold_idx+1:4d}  {held_out:<32}  {len(y_test):6d}  {fold_acc:6.3f}  {fold_f1:6.3f}")

    if not fold_results:
        return {"error": "No valid folds.", "skipped_folds": skipped_folds}

    accs = np.array([r["accuracy"] for r in fold_results])
    f1s  = np.array([r["f1"]       for r in fold_results])
    all_yt = np.array(all_y_true)
    all_yp = np.array(all_y_pred)

    overall_acc   = float(accuracy_score(all_yt, all_yp))
    overall_f1    = float(f1_score(all_yt, all_yp, average="weighted", zero_division=0))
    overall_gmean = _compute_gmean(all_yt, all_yp, classes)
    cm_agg        = confusion_matrix(all_yt, all_yp, labels=classes)

    if verbose:
        print("  " + "-" * 62)
        print(f"  Mean  acc: {accs.mean():.3f} +/- {accs.std():.3f}")
        print(f"  Mean  F1 : {f1s.mean():.3f} +/- {f1s.std():.3f}")
        print(f"  Overall acc (concatenated): {overall_acc:.3f}")
        print(f"  Overall F1  (concatenated): {overall_f1:.3f}")
        print(f"  G-Mean (concatenated):      {overall_gmean:.3f}")

    return {
        "fold_results":     fold_results,
        "n_folds":          len(fold_results),
        "skipped_folds":    skipped_folds,
        "mean_accuracy":    float(accs.mean()),
        "std_accuracy":     float(accs.std()),
        "mean_f1":          float(f1s.mean()),
        "std_f1":           float(f1s.std()),
        "overall_accuracy": overall_acc,
        "overall_f1":       overall_f1,
        "overall_gmean":    overall_gmean,
        "confusion_matrix": cm_agg.tolist(),
        "classes":          classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Time-series model factories
# ─────────────────────────────────────────────────────────────────────────────
def make_ts_models():
    from sktime.classification.kernel_based import RocketClassifier, Arsenal
    from sktime.classification.interval_based import RandomIntervalSpectralEnsemble

    return {
        # ROCKET: convolutional kernel transform + ridge. Handles multivariate.
        "ROCKET": lambda: RocketClassifier(
            num_kernels=10000,
            random_state=SEED,
        ),
        # Arsenal: ensemble of ROCKET + ridge classifiers. More robust than single ROCKET.
        "Arsenal": lambda: Arsenal(
            num_kernels=2000,
            n_estimators=15,
            random_state=SEED,
        ),
        # RISE: random interval spectral features — multivariate-capable, faster than TSForest
        "RISE": lambda: RandomIntervalSpectralEnsemble(
            n_estimators=200,
            random_state=SEED,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined_ranking(all_results, classical_csv_path):
    """Bar chart ranking all models by LOSO overall accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Load classical results
    rows = []
    if os.path.exists(classical_csv_path):
        df_old = pd.read_csv(classical_csv_path)
        for _, row in df_old.iterrows():
            rows.append({
                "Model": row["model"],
                "Overall Acc": row["overall_accuracy"],
                "F1": row["overall_f1"],
                "G-Mean": row["overall_gmean"],
                "Type": "Classical",
            })

    # Add modern results
    type_map = {
        "XGBoost": "Boosting", "LightGBM": "Boosting", "CatBoost": "Boosting",
        "TabPFN": "Modern-Tabular", "AutoGluon": "Modern-Tabular",
        "ROCKET": "Time-Series", "Arsenal": "Time-Series", "RISE": "Time-Series",
    }
    for name, res in all_results.items():
        if "error" in res:
            continue
        rows.append({
            "Model": name,
            "Overall Acc": res["overall_accuracy"],
            "F1": res["overall_f1"],
            "G-Mean": res["overall_gmean"],
            "Type": type_map.get(name, "Modern"),
        })

    df = pd.DataFrame(rows).sort_values("Overall Acc", ascending=True)

    palette = {
        "Classical": "#4C72B0",
        "Boosting": "#DD8452",
        "Modern-Tabular": "#8172B2",
        "Time-Series": "#55A868",
    }
    colors = [palette.get(t, "#888") for t in df["Type"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["Model"], df["Overall Acc"], color=colors)

    for bar, val in zip(bars, df["Overall Acc"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("LOSO Overall Accuracy (concatenated folds)")
    ax.set_title("LOSO Model Ranking — All Models\n(9 classical + boosting + modern-tabular + time-series)")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="50% baseline")

    # Legend for type colours
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k) for k, v in palette.items()]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, "loso_all_models_ranking.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(res, model_name):
    """Confusion matrix for a single model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "confusion_matrix" not in res or "classes" not in res:
        return
    cm     = np.array(res["confusion_matrix"])
    labels = res["classes"]

    # Normalise to per-class recall
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, cmap="Blues", ax=ax, vmin=0, vmax=1,
                linewidths=0.4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"LOSO Confusion Matrix — {model_name}\n(normalised per true class)")
    plt.xticks(rotation=35, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("/", "-")
    path = os.path.join(FIG_DIR, f"loso_cm_{safe_name}.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  LOSO — Modern Models Experiment")
    print("=" * 65)

    # Load data
    print("\nLoading data...")
    data = load_with_sessions(CLASS_FILES, DATA_DIR)
    print(f"  {len(data['y'])} cycles | {data['n_sessions']} sessions | "
          f"{len(data['classes'])} classes")

    tabular_models = make_tabular_models()
    ts_models      = make_ts_models()

    all_results = {}

    # ── Tabular boosting models ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TABULAR / MODERN MODELS  (XGBoost, LightGBM, CatBoost, TabPFN, TabPFN-Auto, AutoGluon)")
    print("=" * 65)

    for name, factory in tabular_models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            res = run_loso(data, factory, normalization="minmax",
                           flatten_method="curve", verbose=True)
            res["train_time"] = time.time() - t0
            all_results[name] = res
            print(f"  [{name}] done in {res['train_time']:.1f}s")
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            all_results[name] = {"error": str(e)}

    # ── Time-series models ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TIME-SERIES MODELS  (ROCKET, TSForest)")
    print("=" * 65)

    for name, factory in ts_models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            res = run_loso_ts(data, factory, normalization="minmax",
                              verbose=True, model_name=name)
            res["train_time"] = time.time() - t0
            all_results[name] = res
            print(f"  [{name}] done in {res['train_time']:.1f}s")
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            all_results[name] = {"error": str(e)}

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SAVING RESULTS")
    print("=" * 65)

    # JSON (detailed per-fold)
    json_path = os.path.join(RES_DIR, "loso_modern_results_detailed.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # CSV summary
    rows = []
    for name, res in all_results.items():
        if "error" in res:
            rows.append({"model": name, "error": res["error"]})
            continue
        rows.append({
            "model":            name,
            "mean_accuracy":    round(res["mean_accuracy"], 4),
            "std_accuracy":     round(res["std_accuracy"], 4),
            "overall_accuracy": round(res["overall_accuracy"], 4),
            "overall_f1":       round(res["overall_f1"], 4),
            "overall_gmean":    round(res["overall_gmean"], 4),
            "n_folds":          res["n_folds"],
            "train_time_s":     round(res.get("train_time", 0), 1),
        })
    df_modern = pd.DataFrame(rows)
    csv_path = os.path.join(RES_DIR, "loso_modern_results_summary.csv")
    df_modern.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    classical_csv = os.path.join(RES_DIR, "loso_results_summary.csv")
    plot_combined_ranking(all_results, classical_csv)
    for name, res in all_results.items():
        if "error" not in res:
            plot_confusion_matrix(res, name)

    # ── Final ranking table (modern models only) ──────────────────────────────
    print("\n" + "=" * 65)
    print("  MODERN MODELS RANKING (LOSO overall accuracy)")
    print("=" * 65)
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True)
    print(f"\n  {'Model':<20}  {'Mean+/-Std':>14}  {'Overall Acc':>11}  {'F1':>8}  {'G-Mean':>8}")
    print("  " + "-" * 70)
    for name, res in ranked:
        print(f"  {name:<20}  "
              f"{res['mean_accuracy']:.3f}+/-{res['std_accuracy']:.3f}  "
              f"{res['overall_accuracy']:11.3f}  "
              f"{res['overall_f1']:8.3f}  "
              f"{res['overall_gmean']:8.3f}")

    # ── Combined ranking (classical + modern) ─────────────────────────────────
    if os.path.exists(classical_csv):
        df_cl = pd.read_csv(classical_csv)
        df_combined = pd.concat([
            df_cl[["model", "mean_accuracy", "std_accuracy",
                   "overall_accuracy", "overall_f1", "overall_gmean"]],
            df_modern[df_modern["error"].isna()][
                ["model", "mean_accuracy", "std_accuracy",
                 "overall_accuracy", "overall_f1", "overall_gmean"]
            ] if "error" in df_modern.columns else
            df_modern[["model", "mean_accuracy", "std_accuracy",
                        "overall_accuracy", "overall_f1", "overall_gmean"]],
        ], ignore_index=True).sort_values("overall_accuracy", ascending=False)

        print("\n" + "=" * 65)
        print("  COMBINED RANKING (all 14 models)")
        print("=" * 65)
        print(f"\n  {'Model':<22}  {'Mean+/-Std':>14}  {'Overall Acc':>11}  {'F1':>8}  {'G-Mean':>8}")
        print("  " + "-" * 72)
        for _, row in df_combined.iterrows():
            try:
                print(f"  {row['model']:<22}  "
                      f"{row['mean_accuracy']:.3f}+/-{row['std_accuracy']:.3f}  "
                      f"{row['overall_accuracy']:11.3f}  "
                      f"{row['overall_f1']:8.3f}  "
                      f"{row['overall_gmean']:8.3f}")
            except Exception:
                pass

        combined_csv = os.path.join(RES_DIR, "loso_combined_ranking.csv")
        df_combined.to_csv(combined_csv, index=False)
        print(f"\n  Saved combined ranking: {combined_csv}")

    print("\nDone. Results in results/ and figures/")


if __name__ == "__main__":
    main()
