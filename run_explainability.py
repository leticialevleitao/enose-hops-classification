"""
SHAP + Feature Importance Analysis for hop e-nose TCC.

The 378 features come from curve-flattening 7 sensors × 54 time steps:
  Features   0– 53  →  TGS826   (time steps 0-53)
  Features  54–107  →  TGS2611
  Features 108–161  →  TGS2603
  Features 162–215  →  TGS813
  Features 216–269  →  TGS822
  Features 270–323  →  TGS2602
  Features 324–377  →  TGS823

SHAP is computed on models trained on ALL data (no LOSO) — standard for
global interpretability analysis in ML papers.

Figures:
  explainability_sensor_importance.pdf   — per-sensor mean |SHAP| (3 models)
  explainability_temporal_profile.pdf    — mean |SHAP| per time step per sensor
  explainability_shap_heatmap.pdf        — 2D sensor × time heatmap of mean |SHAP|
  explainability_permutation_lda.pdf     — permutation importance for LDA
  explainability_class_shap.pdf          — per-class top sensor contributions (RF)

Usage:
    cd "<repo root>"
    python run_explainability.py > results/explainability_log.txt 2>&1
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from enose.session_loader import load_with_sessions
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

SENSOR_COLUMNS = ["TGS826", "TGS2611", "TGS2603", "TGS813", "TGS822", "TGS2602", "TGS823"]
SEED = 42
np.random.seed(SEED)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)
    return plt, sns


def save(fig, name, plt):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature name builder  (curve layout: sensor0_t0..tN, sensor1_t0..tN, ...)
# ─────────────────────────────────────────────────────────────────────────────
def make_feature_names(sensors, n_time):
    names = []
    for s in sensors:
        names.extend([f"{s}_t{t}" for t in range(n_time)])
    return names


def sensor_of(feature_idx, n_time):
    return feature_idx // n_time


def time_of(feature_idx, n_time):
    return feature_idx % n_time


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    data = load_with_sessions(CLASS_FILES, DATA_DIR)
    X3d  = data["X_raw_3d"]           # (130, 54, 7)
    y    = data["y"]
    N, T, S = X3d.shape

    # Global MinMax normalisation (all data — for explainability, not LOSO)
    scaler = MinMaxScaler()
    scaler.fit(X3d.reshape(-1, S))
    X_norm = scaler.transform(X3d.reshape(-1, S)).reshape(N, T, S)

    # Curve flatten: (N, S, T) → (N, S*T)
    X_flat = X_norm.transpose(0, 2, 1).reshape(N, S * T)

    feat_names = make_feature_names(SENSOR_COLUMNS, T)
    feat_df    = pd.DataFrame(X_flat, columns=feat_names)

    print(f"  X: {X_flat.shape}  |  classes: {np.unique(y).tolist()}")
    return X_flat, y, feat_names, T, S, data["classes"]


# ─────────────────────────────────────────────────────────────────────────────
# Train models on all data
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X, y):
    print("\nTraining models on full dataset for SHAP computation...")

    # Random Forest (best classical under LOSO)
    rf = RandomForestClassifier(n_estimators=500, max_depth=11,
                                 random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    print(f"  RF train acc: {rf.score(X, y):.4f}")

    # XGBoost
    from xgboost import XGBClassifier
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                         random_state=SEED, verbosity=0, n_jobs=-1,
                         eval_metric="mlogloss")
    xgb.fit(X, y_enc)
    print(f"  XGB train acc: {xgb.score(X, y_enc):.4f}")

    # LightGBM (best G-Mean under LOSO)
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    lgbm.fit(X, y)
    print(f"  LGBM train acc: {lgbm.score(X, y):.4f}")

    # LDA (best LOSO overall)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X, y)
    print(f"  LDA train acc: {lda.score(X, y):.4f}")

    return rf, xgb, lgbm, lda, le


def _normalize_shap(sv, n_samples, n_features, n_classes):
    """
    Normalize SHAP output to (n_samples, n_features, n_classes)
    regardless of the shape returned by the specific SHAP version.
    """
    if isinstance(sv, list):
        # Older SHAP: list of n_classes arrays each (n_samples, n_features)
        return np.stack(sv, axis=2)
    # Numpy array — find the right transposition by matching known dimension sizes
    shape = sv.shape
    if len(shape) == 2:
        # Binary case or single class: (n_samples, n_features) — expand
        return sv[:, :, np.newaxis]
    # Find which axis corresponds to which dimension
    # (assumes all three dimensions are different sizes, which holds here)
    target = [n_samples, n_features, n_classes]
    perm = []
    remaining = list(range(3))
    for t in target:
        for ax in remaining:
            if shape[ax] == t:
                perm.append(ax)
                remaining.remove(ax)
                break
    if len(perm) == 3:
        return sv.transpose(perm)
    # Fallback: return as-is (shouldn't happen)
    return sv


# ─────────────────────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(rf, xgb, lgbm, X, feat_names, le, y):
    import shap
    print("\nComputing SHAP values (TreeExplainer)...")

    # RF — TreeExplainer
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    exp_rf   = shap.TreeExplainer(rf)
    sv_rf_raw = exp_rf.shap_values(X)
    sv_rf = _normalize_shap(sv_rf_raw, n_samples, n_features, n_classes)
    print(f"  RF SHAP shape: {sv_rf.shape}")

    # XGBoost — uses integer labels
    exp_xgb   = shap.TreeExplainer(xgb)
    sv_xgb_raw = exp_xgb.shap_values(X)
    sv_xgb = _normalize_shap(sv_xgb_raw, n_samples, n_features, n_classes)
    print(f"  XGB SHAP shape: {sv_xgb.shape}")

    # LightGBM
    exp_lgbm   = shap.TreeExplainer(lgbm)
    sv_lgbm_raw = exp_lgbm.shap_values(X)
    sv_lgbm = _normalize_shap(sv_lgbm_raw, n_samples, n_features, n_classes)
    print(f"  LGBM SHAP shape: {sv_lgbm.shape}")

    return sv_rf, sv_xgb, sv_lgbm


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate SHAP per sensor and per time step
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_shap(sv_3d, n_time, n_sensors, sensor_names):
    """
    sv_3d: (n_samples, n_features, n_classes)
    Returns:
        sensor_imp  : (n_sensors,)   mean |SHAP| per sensor
        time_imp    : (n_time,)      mean |SHAP| per time step (across sensors)
        sensor_time : (n_sensors, n_time) 2D heatmap
    """
    abs_shap = np.abs(sv_3d).mean(axis=(0, 2))   # mean over samples + classes → (n_features,)

    sensor_imp   = np.zeros(n_sensors)
    time_imp     = np.zeros(n_time)
    sensor_time  = np.zeros((n_sensors, n_time))

    for feat_idx in range(n_sensors * n_time):
        s = sensor_of(feat_idx, n_time)
        t = time_of(feat_idx, n_time)
        v = abs_shap[feat_idx]
        sensor_imp[s]     += v
        time_imp[t]       += v
        sensor_time[s, t] += v

    # Normalise to mean per feature (not sum)
    sensor_imp  /= n_time
    time_imp    /= n_sensors

    return sensor_imp, time_imp, sensor_time


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Sensor importance (RF, XGB, LGBM side by side)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sensor_importance(sv_rf, sv_xgb, sv_lgbm, n_time, n_sensors):
    print("\n[1/5] Sensor importance bar chart...")
    plt, sns = _plt()

    models  = {"Random Forest": sv_rf, "XGBoost": sv_xgb, "LightGBM": sv_lgbm}
    colours = {"Random Forest": "#4C72B0", "XGBoost": "#DD8452", "LightGBM": "#55A868"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (name, sv) in zip(axes, models.items()):
        imp, _, _ = aggregate_shap(sv, n_time, n_sensors, SENSOR_COLUMNS)
        # Sort by importance
        order = np.argsort(imp)[::-1]
        bars = ax.barh([SENSOR_COLUMNS[i] for i in order[::-1]],
                       imp[order[::-1]],
                       color=colours[name])
        ax.set_xlabel("Mean |SHAP| per time step")
        ax.set_title(name, fontweight="bold")
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, imp[order[::-1]]):
            ax.text(bar.get_width() + max(imp)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=7)

    fig.suptitle("Sensor importance — mean |SHAP| value per sensor\n"
                 "(higher = sensor contributes more to classification)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "explainability_sensor_importance.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Temporal importance profile per sensor
# ─────────────────────────────────────────────────────────────────────────────
def fig_temporal_profile(sv_rf, sv_lgbm, n_time, n_sensors):
    print("\n[2/5] Temporal SHAP profile...")
    plt, sns = _plt()

    _, _, st_rf   = aggregate_shap(sv_rf,   n_time, n_sensors, SENSOR_COLUMNS)
    _, _, st_lgbm = aggregate_shap(sv_lgbm, n_time, n_sensors, SENSOR_COLUMNS)

    t = np.arange(n_time)
    palette = sns.color_palette("tab10", n_sensors)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for ax, (name, st) in zip(axes, [("Random Forest", st_rf), ("LightGBM", st_lgbm)]):
        for s_idx, sensor in enumerate(SENSOR_COLUMNS):
            ax.plot(t, st[s_idx], color=palette[s_idx], linewidth=1.5,
                    label=sensor, alpha=0.85)
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(f"{name} — per-sensor temporal importance profile")
        ax.legend(fontsize=8, loc="upper right", ncol=4, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step within measurement cycle")
    fig.suptitle("Which time region of the sensor response drives classification?",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "explainability_temporal_profile.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — 2D SHAP heatmap: sensor × time
# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_heatmap(sv_rf, sv_lgbm, n_time, n_sensors):
    print("\n[3/5] 2D sensor x time SHAP heatmap...")
    plt, sns = _plt()

    _, _, st_rf   = aggregate_shap(sv_rf,   n_time, n_sensors, SENSOR_COLUMNS)
    _, _, st_lgbm = aggregate_shap(sv_lgbm, n_time, n_sensors, SENSOR_COLUMNS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    for ax, (name, st) in zip(axes, [("Random Forest", st_rf), ("LightGBM", st_lgbm)]):
        df_heat = pd.DataFrame(st, index=SENSOR_COLUMNS,
                               columns=[f"t{i}" for i in range(n_time)])
        sns.heatmap(df_heat, ax=ax, cmap="YlOrRd",
                    xticklabels=10, yticklabels=True,
                    cbar_kws={"label": "Mean |SHAP|"})
        ax.set_title(f"{name}\nSensor × time-step importance", fontweight="bold")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Sensor")
        # Only show every 10th time tick
        xticks = list(range(0, n_time, 10))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=8)

    fig.suptitle("SHAP importance heatmap — sensor x time (bright = most informative)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "explainability_shap_heatmap.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Permutation importance for LDA
# ─────────────────────────────────────────────────────────────────────────────
def fig_lda_permutation(lda, X, y, n_time, n_sensors):
    print("\n[4/5] Permutation importance for LDA...")
    plt, sns = _plt()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(lda, X, y, n_repeats=10,
                                      random_state=SEED, n_jobs=-1)

    imp_mean = perm.importances_mean     # (n_features,)

    # Aggregate per sensor
    sensor_imp = np.zeros(n_sensors)
    for feat_idx in range(n_sensors * n_time):
        s = sensor_of(feat_idx, n_time)
        sensor_imp[s] += imp_mean[feat_idx]
    sensor_imp /= n_time

    order = np.argsort(sensor_imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([SENSOR_COLUMNS[i] for i in order[::-1]],
                   sensor_imp[order[::-1]], color="#9467BD")
    ax.set_xlabel("Mean decrease in accuracy when sensor is permuted")
    ax.set_title("LDA — Permutation importance per sensor\n"
                 "(higher = sensor is more critical to LDA decisions)")
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, sensor_imp[order[::-1]]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "explainability_permutation_lda.pdf", plt)

    print("  LDA permutation importance per sensor:")
    for i in order:
        print(f"    {SENSOR_COLUMNS[i]:10s}: {sensor_imp[i]:.4f}")

    return sensor_imp


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-class sensor SHAP (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
def fig_class_shap(sv_rf, y, classes, n_time, n_sensors):
    """
    For each class: compute mean |SHAP| per sensor using only samples of that class.
    Shows which sensors the model relies on to identify each hop variety.
    """
    print("\n[5/5] Per-class sensor SHAP (RF)...")
    plt, sns = _plt()

    # sv_rf: (n_samples, n_features, n_classes)
    # For class c: take sv_rf[:, :, c_idx] = contribution to predicting class c
    class_sensor_imp = np.zeros((len(classes), n_sensors))

    for c_idx, cls in enumerate(classes):
        sv_class = sv_rf[:, :, c_idx]     # (130, 378) — contribution toward class c
        # Use samples OF this class (most informative for "what makes Saaz?")
        mask = y == cls
        sv_cls_samples = np.abs(sv_class[mask])    # (n_cls, 378)
        # Aggregate per sensor
        for feat_idx in range(n_sensors * n_time):
            s = sensor_of(feat_idx, n_time)
            class_sensor_imp[c_idx, s] += sv_cls_samples[:, feat_idx].mean()
        class_sensor_imp[c_idx] /= n_time

    # Normalise each class row to [0,1] for comparability
    row_max = class_sensor_imp.max(axis=1, keepdims=True)
    class_sensor_norm = np.where(row_max > 0,
                                  class_sensor_imp / row_max, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw importance heatmap
    df_raw = pd.DataFrame(class_sensor_imp,
                           index=classes, columns=SENSOR_COLUMNS)
    sns.heatmap(df_raw, ax=axes[0], cmap="YlOrRd", annot=True,
                fmt=".4f", linewidths=0.4,
                cbar_kws={"label": "Mean |SHAP|"}, annot_kws={"size": 7})
    axes[0].set_title("Mean |SHAP| per sensor per class (RF)\nraw values",
                      fontweight="bold")
    axes[0].set_xlabel("Sensor")
    axes[0].set_ylabel("Class")

    # Row-normalised heatmap
    df_norm = pd.DataFrame(class_sensor_norm,
                            index=classes, columns=SENSOR_COLUMNS)
    sns.heatmap(df_norm, ax=axes[1], cmap="YlOrRd", annot=True,
                fmt=".2f", linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={"label": "Relative importance (0-1)"},
                annot_kws={"size": 7})
    axes[1].set_title("Row-normalised (each class max=1)\n— which sensor matters most per class",
                      fontweight="bold")
    axes[1].set_xlabel("Sensor")
    axes[1].set_ylabel("Class")

    fig.suptitle("Per-class sensor SHAP importance (Random Forest)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "explainability_class_shap.pdf", plt)

    # Print top sensor per class
    print("\n  Top sensor per class (RF SHAP):")
    for c_idx, cls in enumerate(classes):
        top_s = int(np.argmax(class_sensor_imp[c_idx]))
        print(f"    {cls:<20s}: {SENSOR_COLUMNS[top_s]}")

    return class_sensor_imp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SHAP + Feature Importance Analysis")
    print("=" * 65)

    X, y, feat_names, n_time, n_sensors, classes = load_data()

    # Train
    rf, xgb, lgbm, lda, le = train_models(X, y)

    # SHAP
    sv_rf, sv_xgb, sv_lgbm = compute_shap(rf, xgb, lgbm, X, feat_names, le, y)

    # Figures
    fig_sensor_importance(sv_rf, sv_xgb, sv_lgbm, n_time, n_sensors)
    fig_temporal_profile(sv_rf, sv_lgbm, n_time, n_sensors)
    fig_shap_heatmap(sv_rf, sv_lgbm, n_time, n_sensors)
    lda_imp = fig_lda_permutation(lda, X, y, n_time, n_sensors)
    class_shap = fig_class_shap(sv_rf, y, classes, n_time, n_sensors)

    # ── Global sensor ranking across all methods ───────────────────────────
    print("\n" + "=" * 65)
    print("  GLOBAL SENSOR RANKING (mean |SHAP| per sensor)")
    print("=" * 65)

    rf_imp, _, _   = aggregate_shap(sv_rf,   n_time, n_sensors, SENSOR_COLUMNS)
    xgb_imp, _, _  = aggregate_shap(sv_xgb,  n_time, n_sensors, SENSOR_COLUMNS)
    lgbm_imp, _, _ = aggregate_shap(sv_lgbm, n_time, n_sensors, SENSOR_COLUMNS)

    # Average rank across all methods (SHAP + permutation)
    methods_imp = np.stack([rf_imp, xgb_imp, lgbm_imp, lda_imp], axis=0)  # (4, 7)
    # Rank each method (0=most important)
    ranks = np.argsort(np.argsort(-methods_imp, axis=1), axis=1)
    avg_rank = ranks.mean(axis=0)

    print(f"\n  {'Sensor':<12}  {'RF-SHAP':>10}  {'XGB-SHAP':>10}  {'LGBM-SHAP':>10}  {'LDA-Perm':>10}  {'Avg Rank':>10}")
    print("  " + "-" * 68)
    for i in np.argsort(avg_rank):
        print(f"  {SENSOR_COLUMNS[i]:<12}  {rf_imp[i]:>10.4f}  {xgb_imp[i]:>10.4f}  "
              f"{lgbm_imp[i]:>10.4f}  {lda_imp[i]:>10.4f}  {avg_rank[i]:>10.2f}")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "sensor_importance": {
            "Random Forest SHAP":  {s: float(rf_imp[i])   for i, s in enumerate(SENSOR_COLUMNS)},
            "XGBoost SHAP":        {s: float(xgb_imp[i])  for i, s in enumerate(SENSOR_COLUMNS)},
            "LightGBM SHAP":       {s: float(lgbm_imp[i]) for i, s in enumerate(SENSOR_COLUMNS)},
            "LDA Permutation":     {s: float(lda_imp[i])  for i, s in enumerate(SENSOR_COLUMNS)},
        },
        "average_rank":   {s: float(avg_rank[i]) for i, s in enumerate(SENSOR_COLUMNS)},
        "class_top_sensor": {
            cls: SENSOR_COLUMNS[int(np.argmax(class_shap[c_idx]))]
            for c_idx, cls in enumerate(classes)
        },
    }
    out_path = os.path.join(RES_DIR, "explainability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 65)
    print("  Explainability analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
