"""
run_expanded.py
===============
Multi-form experiments extending the cone LOSO results to pellets and oil extracts.

Four experiments:
  Exp 1 – Cross-form generalization: train on cones, test on pellets
           (5 shared varieties: Chinook, Comet, Saaz, Vista, Zeus)
  Exp 2 – Form classification: can the E-nose detect cone / pellet / oil?
  Exp 3 – Pellet variety classification (stratified 5-fold, 9 varieties)
  Exp 4 – Oil variety classification (stratified 5-fold, 12 varieties)

All figures are saved to figures/ as PDF.
Results are saved to results/expanded_results.json.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: catboost not installed – skipping CatBoost in Exps 3&4")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(REPO_DIR, "data", "consolidated")
FIG_DIR    = os.path.join(REPO_DIR, "figures")
RES_DIR    = os.path.join(REPO_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

sys.path.insert(0, REPO_DIR)
from enose.session_loader import _parse_cycles_from_csv
from enose.data_loader import SENSOR_COLUMNS

SENSORS = SENSOR_COLUMNS  # 7 sensors

# ---------------------------------------------------------------------------
# Cone class files (already used in LOSO)
# ---------------------------------------------------------------------------
CONE_CLASS_FILES = {
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

# Variety-level grouping (collapse conditions for cross-form)
CONE_TO_VARIETY = {
    "Chinook_Fresh": "Chinook",
    "Chinook_Aged":  "Chinook",
    "Comet_2025":    "Comet",
    "Saaz_2005":     "Saaz",
    "Saaz_Fresh":    "Saaz",
    "Saaz_Aged":     "Saaz",
    "Saaz_Dry":      "Saaz",
    "Vista_Fresh":   "Vista",
    "Zeus_2025":     "Zeus",
}

# Pellet files (9 varieties, each has 1 session of 5 cycles)
PELLET_CLASS_FILES = {
    "Cascade":      "cascade_pellete.csv",
    "Chinook":      "chinook_pellete.csv",
    "Comet":        "comet_pellete.csv",
    "Nugget":       "nugget_pellete.csv",
    "Saaz":         "saaz_pellete.csv",
    "Triple_Pearl": "triple_pearl_pellete.csv",
    "Triumph":      "triumph_pellete.csv",
    "Vista":        "vista_pellete.csv",
    "Zeus":         "zeus_pellete.csv",
}

# Oil files (12 classes)
OIL_CLASS_FILES = {
    "Cascade_OE":        "cascade_pelletizado_oe.csv",
    "Chinook_OE":        "chinook_pelletizado_oe.csv",
    "Comet_Fresh_OE":    "comet_fresco_oe.csv",
    "Comet_OE":          "comet_pelletizado_oe.csv",
    "Nugget_OE":         "nugget_pelletizado_oe.csv",
    "Saaz_OE":           "saaz_pelletizado_oe.csv",
    "Saaz_Dry_OE":       "saaz_seco_oe.csv",
    "Triple_Pearl_OE":   "triple_pearl_pelletizado_oe.csv",
    "Triumph_OE":        "triumph_pelletizado_oe.csv",
    "Vista_OE":          "vista_pelletizado_oe.csv",
    "Zeus_Fresh_OE":     "zeus_fresco_oe.csv",
    "Zeus_OE":           "zeus_pelletizado_oe.csv",
}

# ---------------------------------------------------------------------------
# Helper: load a dict of {label: filename} → X_3d (N,T,S), y, groups
# ---------------------------------------------------------------------------
def load_form(class_files, truncate_to=None):
    """
    Load all cycles from a dict of {label: filename}.

    Returns:
        X_3d  – (N, T, 7) float32
        y     – (N,) str labels
        groups – (N,) str session IDs (one per file = one session per file)
        min_len – min cycle length found
    """
    all_cycles, all_labels, all_groups = [], [], []
    for label, fname in class_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found – skipping {label}")
            continue
        cycles = _parse_cycles_from_csv(fpath)
        all_cycles.extend(cycles)
        all_labels.extend([label] * len(cycles))
        all_groups.extend([label] * len(cycles))   # file = session
        print(f"  {label:25s}: {len(cycles)} cycles from {fname}")

    lens = [len(c) for c in all_cycles]
    min_len = min(lens)
    if truncate_to is not None:
        min_len = min(min_len, truncate_to)

    X_3d = np.stack([c[:min_len, :] for c in all_cycles]).astype(np.float32)
    y     = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"  -> {len(y)} cycles, truncated to {min_len} time steps, shape {X_3d.shape}")
    return X_3d, y, groups, min_len


def flatten(X_3d):
    """(N, T, S) -> (N, S*T) curve layout (sensor-major order)."""
    N, T, S = X_3d.shape
    return X_3d.transpose(0, 2, 1).reshape(N, S * T)


def normalize_fit_transform(X_train_flat, X_test_flat=None):
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_train_flat)
    X_te = scaler.transform(X_test_flat) if X_test_flat is not None else None
    return X_tr, X_te, scaler


def make_models():
    models = {
        "LDA":          LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            verbose=0, random_seed=42
        )
    return models


def gmean(y_true, y_pred):
    """Geometric mean of per-class recalls."""
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    return float(np.exp(np.mean(np.log(recalls + 1e-9))))


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# EXP 1: Cross-form generalization (cones -> pellets)
# ---------------------------------------------------------------------------
def exp1_cross_form():
    print("\n" + "="*70)
    print("EXP 1: Cross-form generalization (cones -> pellets)")
    print("="*70)

    # Shared varieties: Chinook, Comet, Saaz, Vista, Zeus
    SHARED = {"Chinook", "Comet", "Saaz", "Vista", "Zeus"}

    # Load cones with variety labels
    print("\nLoading cones (variety-level labels):")
    cone_cycles, cone_labels, cone_groups = [], [], []
    for class_name, fname in CONE_CLASS_FILES.items():
        variety = CONE_TO_VARIETY[class_name]
        if variety not in SHARED:
            continue
        fpath = os.path.join(DATA_DIR, fname)
        cycles = _parse_cycles_from_csv(fpath)
        cone_cycles.extend(cycles)
        cone_labels.extend([variety] * len(cycles))
        cone_groups.extend([class_name] * len(cycles))
        print(f"  {class_name:25s} ({variety}): {len(cycles)} cycles")

    # Truncate cones to 54 (established LOSO target)
    TRUNC = 54
    cone_lens = [len(c) for c in cone_cycles]
    cone_X3d  = np.stack([c[:TRUNC, :] for c in cone_cycles]).astype(np.float32)
    cone_y    = np.array(cone_labels)

    print(f"\n  Cone training set: {cone_X3d.shape}, {len(np.unique(cone_y))} varieties")

    # Load pellets for shared varieties
    print("\nLoading pellets (shared varieties only):")
    pell_cycles, pell_labels = [], []
    for variety in sorted(SHARED):
        fname = PELLET_CLASS_FILES.get(variety)
        if fname is None:
            continue
        fpath = os.path.join(DATA_DIR, fname)
        cycles = _parse_cycles_from_csv(fpath)
        pell_cycles.extend(cycles)
        pell_labels.extend([variety] * len(cycles))
        print(f"  {variety:20s}: {len(cycles)} cycles")

    pell_X3d = np.stack([c[:TRUNC, :] for c in pell_cycles]).astype(np.float32)
    pell_y   = np.array(pell_labels)
    print(f"\n  Pellet test set:  {pell_X3d.shape}, {len(np.unique(pell_y))} varieties")

    # Flatten
    cone_flat = flatten(cone_X3d)
    pell_flat = flatten(pell_X3d)

    # Normalize: fit on cones, apply to pellets
    cone_norm, pell_norm, _ = normalize_fit_transform(cone_flat, pell_flat)

    models = make_models()
    results = {}

    print("\n  Model             Acc    F1     G-Mean")
    print("  " + "-"*42)
    for mname, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(cone_norm, cone_y)
            preds = model.predict(pell_norm)

        acc  = accuracy_score(pell_y, preds)
        f1   = f1_score(pell_y, preds, average="weighted", zero_division=0)
        gm   = gmean(pell_y, preds)
        results[mname] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "gmean": round(gm, 4)}
        print(f"  {mname:16s}  {acc:.3f}  {f1:.3f}  {gm:.3f}")

    # --- Figure: Cross-form confusion matrix (best model) ---
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_model_name]
    preds_best = best_model.predict(pell_norm)
    varieties = sorted(SHARED)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (mname, model) in zip(axes, models.items()):
        p = model.predict(pell_norm)
        cm = confusion_matrix(pell_y, p, labels=varieties)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(varieties)))
        ax.set_yticks(range(len(varieties)))
        ax.set_xticklabels(varieties, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(varieties, fontsize=8)
        ax.set_title(f"{mname}\nAcc={results[mname]['accuracy']:.1%}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (pellets)")
        for i in range(len(varieties)):
            for j in range(len(varieties)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)

    fig.suptitle("Exp 1: Cross-form Generalization (Trained on Cones, Tested on Pellets)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig("exp1_cross_form_confusion.pdf")

    # --- Figure: Per-variety accuracy bar chart ---
    variety_accs = {}
    for variety in varieties:
        mask = pell_y == variety
        if mask.sum() == 0:
            continue
        p = best_model.predict(pell_norm[mask])
        variety_accs[variety] = accuracy_score(pell_y[mask], p)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71" if v >= 0.8 else "#e74c3c" if v < 0.4 else "#f39c12"
              for v in variety_accs.values()]
    bars = ax.bar(variety_accs.keys(), variety_accs.values(), color=colors, edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Per-variety accuracy (pellets)")
    ax.set_title(f"Exp 1: Per-variety accuracy ({best_model_name}, trained on cones)")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, label="Random (5-class)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, variety_accs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.0%}",
                ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("exp1_cross_form_per_variety.pdf")

    return results


# ---------------------------------------------------------------------------
# EXP 2: Form classification (cone vs pellet vs oil)
# ---------------------------------------------------------------------------
def exp2_form_classification():
    print("\n" + "="*70)
    print("EXP 2: Form classification (cone / pellet / oil)")
    print("="*70)

    # Truncate to oil min (52) to include all forms
    TRUNC = 52
    all_cycles, all_labels, all_groups = [], [], []

    print("\nLoading cones:")
    for class_name, fname in CONE_CLASS_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        cycles = _parse_cycles_from_csv(fpath)
        for i, c in enumerate(cycles):
            all_cycles.append(c[:TRUNC, :])
        all_labels.extend(["Cone"] * len(cycles))
        all_groups.extend([f"cone_{class_name}"] * len(cycles))
    print(f"  Cone cycles: {all_labels.count('Cone')}")

    print("Loading pellets:")
    for class_name, fname in PELLET_CLASS_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        cycles = _parse_cycles_from_csv(fpath)
        for i, c in enumerate(cycles):
            all_cycles.append(c[:TRUNC, :])
        all_labels.extend(["Pellet"] * len(cycles))
        all_groups.extend([f"pellet_{class_name}"] * len(cycles))
    print(f"  Pellet cycles: {all_labels.count('Pellet')}")

    print("Loading oils:")
    for class_name, fname in OIL_CLASS_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        cycles = _parse_cycles_from_csv(fpath)
        for i, c in enumerate(cycles):
            all_cycles.append(c[:TRUNC, :])
        all_labels.extend(["Oil"] * len(cycles))
        all_groups.extend([f"oil_{class_name}"] * len(cycles))
    print(f"  Oil cycles:    {all_labels.count('Oil')}")

    X_3d   = np.stack([np.array(c, dtype=np.float32) for c in all_cycles])
    y      = np.array(all_labels)
    groups = np.array(all_groups)
    forms  = ["Cone", "Pellet", "Oil"]
    print(f"\n  Total: {len(y)} cycles, truncated to {TRUNC} steps")

    X_flat = flatten(X_3d)
    logo   = LeaveOneGroupOut()
    models = make_models()

    results = {}
    print("\n  Model             Acc    F1     G-Mean")
    print("  " + "-"*42)

    for mname, model in models.items():
        all_true, all_pred = [], []
        for train_idx, test_idx in logo.split(X_flat, y, groups):
            X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scaler = MinMaxScaler()
            X_tr_n = scaler.fit_transform(X_tr)
            X_te_n = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_n, y_tr)
                all_pred.extend(model.predict(X_te_n))
            all_true.extend(y_te)

        acc = accuracy_score(all_true, all_pred)
        f1  = f1_score(all_true, all_pred, average="weighted", zero_division=0)
        gm  = gmean(all_true, all_pred)
        results[mname] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "gmean": round(gm, 4)}
        print(f"  {mname:16s}  {acc:.3f}  {f1:.3f}  {gm:.3f}")

    # --- Figure: Form confusion matrix (best model LOGO) ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    model = models[best_name]
    all_true_plot, all_pred_plot = [], []
    for train_idx, test_idx in logo.split(X_flat, y, groups):
        X_tr_n = MinMaxScaler().fit_transform(X_flat[train_idx])
        scaler = MinMaxScaler()
        X_tr_n = scaler.fit_transform(X_flat[train_idx])
        X_te_n = scaler.transform(X_flat[test_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_n, y[train_idx])
            all_pred_plot.extend(model.predict(X_te_n))
        all_true_plot.extend(y[test_idx])

    cm = confusion_matrix(all_true_plot, all_pred_plot, labels=forms)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(forms); ax.set_yticklabels(forms)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Exp 2: Form Classification ({best_name})\nAcc={results[best_name]['accuracy']:.1%}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    save_fig("exp2_form_classification_cm.pdf")

    # --- Figure: 3-class bar results ---
    fig, ax = plt.subplots(figsize=(6, 4))
    mnames = list(results.keys())
    accs   = [results[m]["accuracy"] for m in mnames]
    f1s    = [results[m]["f1"] for m in mnames]
    x = np.arange(len(mnames))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", edgecolor="black")
    ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color="#2ecc71", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(mnames)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Exp 2: Form Classification (LOGO-CV)")
    ax.legend(); ax.axhline(1/3, color="gray", linestyle="--", linewidth=0.8, label="Random")
    plt.tight_layout()
    save_fig("exp2_form_classification_bars.pdf")

    return results


# ---------------------------------------------------------------------------
# EXP 3: Pellet variety classification (stratified 5-fold)
# ---------------------------------------------------------------------------
def exp3_pellet_variety():
    print("\n" + "="*70)
    print("EXP 3: Pellet variety classification (stratified 5-fold)")
    print("="*70)

    print("\nLoading pellets:")
    X_3d, y, groups, min_len = load_form(PELLET_CLASS_FILES)
    X_flat = flatten(X_3d)
    classes = sorted(np.unique(y))
    print(f"  Classes: {classes}")

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    results = {}

    print("\n  Model             Acc    F1     G-Mean")
    print("  " + "-"*42)
    for mname, model in models.items():
        all_true, all_pred = [], []
        for train_idx, test_idx in skf.split(X_flat, y):
            X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scaler = MinMaxScaler()
            X_tr_n = scaler.fit_transform(X_tr)
            X_te_n = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_n, y_tr)
                all_pred.extend(model.predict(X_te_n))
            all_true.extend(y_te)

        acc = accuracy_score(all_true, all_pred)
        f1  = f1_score(all_true, all_pred, average="weighted", zero_division=0)
        gm  = gmean(all_true, all_pred)
        results[mname] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "gmean": round(gm, 4)}
        print(f"  {mname:16s}  {acc:.3f}  {f1:.3f}  {gm:.3f}")

    # --- Figure: Best-model confusion matrix ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    model     = models[best_name]
    all_true_plot, all_pred_plot = [], []
    for train_idx, test_idx in skf.split(X_flat, y):
        scaler = MinMaxScaler()
        X_tr_n = scaler.fit_transform(X_flat[train_idx])
        X_te_n = scaler.transform(X_flat[test_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_n, y[train_idx])
            all_pred_plot.extend(model.predict(X_te_n))
        all_true_plot.extend(y[test_idx])

    cm = confusion_matrix(all_true_plot, all_pred_plot, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Exp 3: Pellet Variety Classification ({best_name})\nAcc={results[best_name]['accuracy']:.1%}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.tight_layout()
    save_fig("exp3_pellet_variety_cm.pdf")

    return results


# ---------------------------------------------------------------------------
# EXP 4: Oil variety classification (stratified 5-fold)
# ---------------------------------------------------------------------------
def exp4_oil_variety():
    print("\n" + "="*70)
    print("EXP 4: Oil variety classification (stratified 5-fold)")
    print("="*70)

    print("\nLoading oils:")
    X_3d, y, groups, min_len = load_form(OIL_CLASS_FILES)
    X_flat = flatten(X_3d)
    classes = sorted(np.unique(y))
    print(f"  Classes: {classes}")

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    results = {}

    print("\n  Model             Acc    F1     G-Mean")
    print("  " + "-"*42)
    for mname, model in models.items():
        all_true, all_pred = [], []
        for train_idx, test_idx in skf.split(X_flat, y):
            X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scaler = MinMaxScaler()
            X_tr_n = scaler.fit_transform(X_tr)
            X_te_n = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_n, y_tr)
                all_pred.extend(model.predict(X_te_n))
            all_true.extend(y_te)

        acc = accuracy_score(all_true, all_pred)
        f1  = f1_score(all_true, all_pred, average="weighted", zero_division=0)
        gm  = gmean(all_true, all_pred)
        results[mname] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "gmean": round(gm, 4)}
        print(f"  {mname:16s}  {acc:.3f}  {f1:.3f}  {gm:.3f}")

    # --- Figure: Best-model confusion matrix ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    model     = models[best_name]
    all_true_plot, all_pred_plot = [], []
    for train_idx, test_idx in skf.split(X_flat, y):
        scaler = MinMaxScaler()
        X_tr_n = scaler.fit_transform(X_flat[train_idx])
        X_te_n = scaler.transform(X_flat[test_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_n, y[train_idx])
            all_pred_plot.extend(model.predict(X_te_n))
        all_true_plot.extend(y[test_idx])

    cm = confusion_matrix(all_true_plot, all_pred_plot, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Exp 4: Oil Extract Variety Classification ({best_name})\nAcc={results[best_name]['accuracy']:.1%}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.tight_layout()
    save_fig("exp4_oil_variety_cm.pdf")

    return results


# ---------------------------------------------------------------------------
# Summary figure: all 4 experiments side-by-side
# ---------------------------------------------------------------------------
def fig_summary(exp1, exp2, exp3, exp4):
    fig, ax = plt.subplots(figsize=(12, 5))

    all_models = sorted(set(
        list(exp1.keys()) + list(exp2.keys()) + list(exp3.keys()) + list(exp4.keys())
    ))
    exp_labels = [
        "Exp 1\nCross-form\n(cones->pellets)",
        "Exp 2\nForm class.\n(cone/pell/oil)",
        "Exp 3\nPellet variety\n(5-fold)",
        "Exp 4\nOil variety\n(5-fold)",
    ]
    exps = [exp1, exp2, exp3, exp4]

    colors_map = {"LDA": "#2980b9", "Random Forest": "#27ae60", "CatBoost": "#e67e22"}
    width = 0.25
    n_models = len(all_models)
    x = np.arange(len(exps))

    for i, mname in enumerate(all_models):
        accs = [exps[e].get(mname, {}).get("accuracy", np.nan) for e in range(len(exps))]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width,
                      label=mname, color=colors_map.get(mname, "#95a5a6"),
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.01, f"{val:.0%}",
                        ha="center", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-form E-nose Experiments – All Results")
    ax.legend(loc="upper right")
    ax.axhline(1/5, color="gray", linestyle=":", linewidth=0.8)
    ax.text(3.5, 1/5 + 0.01, "5-class chance (20%)", fontsize=7, color="gray")
    plt.tight_layout()
    save_fig("expanded_summary.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running multi-form expanded experiments...")
    print(f"Data dir:    {DATA_DIR}")
    print(f"Figures dir: {FIG_DIR}")
    print(f"Results dir: {RES_DIR}")

    results = {}
    results["exp1_cross_form"]      = exp1_cross_form()
    results["exp2_form_class"]      = exp2_form_classification()
    results["exp3_pellet_variety"]  = exp3_pellet_variety()
    results["exp4_oil_variety"]     = exp4_oil_variety()

    fig_summary(
        results["exp1_cross_form"],
        results["exp2_form_class"],
        results["exp3_pellet_variety"],
        results["exp4_oil_variety"],
    )

    # Save results
    out_path = os.path.join(RES_DIR, "expanded_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for exp_name, exp_res in results.items():
        print(f"\n{exp_name}:")
        for mname, metrics in exp_res.items():
            print(f"  {mname:18s} acc={metrics['accuracy']:.1%}  f1={metrics['f1']:.3f}")

    print("\nDone.")
