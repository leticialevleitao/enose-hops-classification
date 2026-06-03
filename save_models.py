"""
save_models.py
==============
Train best models on the FULL dataset (no held-out split) and serialize them
as joblib artifacts in models/.

Each artifact is a dict:
    {
        "model":    fitted sklearn-compatible classifier,
        "scaler":   fitted MinMaxScaler,
        "classes":  list of class labels,
        "n_samples": int,
        "n_features": int,
        "loso_accuracy": float,   # from LOSO/kfold experiments (reference)
        "description": str,
        "trained_on": "full_dataset",
        "timestamp": ISO-8601 string,
    }

Models saved
------------
  1. form_detector_lda        — LDA classifies cone / pellet / oil  (91.7% LOSO)
  2. cone_variety_tabpfn      — TabPFN classifies 9 cone varieties  (82.4% LOSO)
  3. cone_variety_lda         — LDA fallback for cone varieties      (70.0% LOSO)
  4. pellet_variety_lda       — LDA classifies 9 pellet varieties    (93.3% 5-fold)
  5. oil_variety_lda          — LDA classifies 12 oil varieties      (95.0% 5-fold)

Usage
-----
    cd "<repo root>"
    python save_models.py
"""

import os
import sys
import json
import warnings
import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

from enose.session_loader import load_with_sessions
from enose.loso_cv import flatten_cycles
from run_expanded import load_form, flatten, CONE_CLASS_FILES, PELLET_CLASS_FILES, OIL_CLASS_FILES

from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import joblib

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(BASE_DIR, "data", "consolidated")

# LOSO / kfold accuracy from previous experiments (for metadata only)
KNOWN_ACCURACY = {
    "form_detector_lda":    0.917,
    "cone_variety_tabpfn":  0.824,
    "cone_variety_lda":     0.700,
    "pellet_variety_lda":   0.933,
    "oil_variety_lda":      0.950,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_artifact(name, model, scaler, classes, X, description):
    artifact = {
        "model":          model,
        "scaler":         scaler,
        "classes":        list(classes),
        "n_samples":      int(X.shape[0]),
        "n_features":     int(X.shape[1]),
        "loso_accuracy":  KNOWN_ACCURACY.get(name, None),
        "description":    description,
        "trained_on":     "full_dataset",
        "timestamp":      datetime.datetime.now().isoformat(),
    }
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(artifact, path, compress=3)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [OK] Saved {name}.joblib  ({size_kb:.0f} KB)")
    return path


def fit_and_save_lda(name, X_flat, y, description):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_flat)
    clf = LinearDiscriminantAnalysis(solver="svd", tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_norm, y)
    classes = clf.classes_
    train_acc = accuracy_score(y, clf.predict(X_norm))
    print(f"  Train acc (full data, sanity check): {train_acc:.3f}")
    save_artifact(name, clf, scaler, classes, X_norm, description)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Form detector (LDA: cone / pellet / oil)
# ─────────────────────────────────────────────────────────────────────────────
def train_form_detector():
    print("\n" + "=" * 60)
    print("  1. Form Detector — LDA (cone / pellet / oil)")
    print("=" * 60)

    segments = []

    print("  Loading cones...")
    X_cone, y_cone, _, _ = load_form(CONE_CLASS_FILES)
    segments.append((X_cone, np.array(["cone"] * len(y_cone))))

    print("  Loading pellets...")
    X_pellet, y_pellet, _, _ = load_form(PELLET_CLASS_FILES)
    segments.append((X_pellet, np.array(["pellet"] * len(y_pellet))))

    print("  Loading oils...")
    X_oil, y_oil, _, _ = load_form(OIL_CLASS_FILES)
    segments.append((X_oil, np.array(["oil"] * len(y_oil))))

    # Truncate all to the shortest time length
    min_T = min(X.shape[1] for X, _ in segments)
    X_all = np.vstack([X[:, :min_T, :] for X, _ in segments])
    y_all = np.concatenate([y for _, y in segments])

    X_flat = flatten(X_all)
    print(f"  Combined: {X_flat.shape[0]} cycles, {X_flat.shape[1]} features")

    fit_and_save_lda(
        "form_detector_lda", X_flat, y_all,
        "LDA trained on all forms (cone/pellet/oil) to detect processing form. "
        "First stage of the production routing pipeline. "
        f"LOSO accuracy: {KNOWN_ACCURACY['form_detector_lda']:.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cone variety — TabPFN (best model, 82.4% LOSO)
# ─────────────────────────────────────────────────────────────────────────────
def train_cone_tabpfn():
    print("\n" + "=" * 60)
    print("  2. Cone Variety — TabPFN v3  (82.4% LOSO)")
    print("=" * 60)

    cone_files = {
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

    data = load_with_sessions(cone_files, DATA_DIR)
    X_3d = data["X_raw_3d"]   # (N, T, S)
    y    = data["y"]
    classes = data["classes"]

    X_flat = flatten_cycles(X_3d, method="curve")
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_flat)

    print(f"  Shape: {X_norm.shape} | Classes: {list(classes)}")

    try:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_norm, y)
        train_acc = accuracy_score(y, clf.predict(X_norm))
        print(f"  Train acc (sanity check): {train_acc:.3f}")
        save_artifact(
            "cone_variety_tabpfn", clf, scaler, classes, X_norm,
            "TabPFN v3 trained on all 125 cone cycles (9 classes). "
            "Best model in LOSO benchmark. Second stage of routing pipeline for cone samples. "
            f"LOSO accuracy: {KNOWN_ACCURACY['cone_variety_tabpfn']:.1%}"
        )
    except Exception as e:
        print(f"  WARNING: TabPFN failed ({e}). Skipping cone_variety_tabpfn.")
        print("  Tip: set TABPFN_TOKEN env var before running.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cone variety — LDA (fallback, 70.0% LOSO)
# ─────────────────────────────────────────────────────────────────────────────
def train_cone_lda():
    print("\n" + "=" * 60)
    print("  3. Cone Variety — LDA  (70.0% LOSO, classical best)")
    print("=" * 60)

    cone_files = {
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

    data = load_with_sessions(cone_files, DATA_DIR)
    X_3d = data["X_raw_3d"]
    y    = data["y"]
    classes = data["classes"]

    X_flat = flatten_cycles(X_3d, method="curve")
    print(f"  Shape: {X_flat.shape} | Classes: {list(classes)}")

    fit_and_save_lda(
        "cone_variety_lda", X_flat, y,
        "LDA trained on all 125 cone cycles (9 classes). "
        "Classical best model; lighter and faster than TabPFN. "
        f"LOSO accuracy: {KNOWN_ACCURACY['cone_variety_lda']:.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pellet variety — LDA (93.3% 5-fold)
# ─────────────────────────────────────────────────────────────────────────────
def train_pellet_lda():
    print("\n" + "=" * 60)
    print("  4. Pellet Variety — LDA  (93.3% 5-fold)")
    print("=" * 60)

    print("  Loading pellets...")
    X_3d, y, _, _ = load_form(PELLET_CLASS_FILES)
    X_flat = flatten(X_3d)
    print(f"  Shape: {X_flat.shape} | Classes: {np.unique(y).tolist()}")

    fit_and_save_lda(
        "pellet_variety_lda", X_flat, y,
        "LDA trained on all 45 pellet cycles (9 varieties). "
        "Second stage of routing pipeline for pellet samples. "
        f"5-fold accuracy: {KNOWN_ACCURACY['pellet_variety_lda']:.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Oil variety — LDA (95.0% 5-fold)
# ─────────────────────────────────────────────────────────────────────────────
def train_oil_lda():
    print("\n" + "=" * 60)
    print("  5. Oil Variety — LDA  (95.0% 5-fold)")
    print("=" * 60)

    print("  Loading oils...")
    X_3d, y, _, _ = load_form(OIL_CLASS_FILES)
    X_flat = flatten(X_3d)
    print(f"  Shape: {X_flat.shape} | Classes: {np.unique(y).tolist()}")

    fit_and_save_lda(
        "oil_variety_lda", X_flat, y,
        "LDA trained on all 60 oil extract cycles (12 varieties). "
        "Second stage of routing pipeline for oil samples. "
        f"5-fold accuracy: {KNOWN_ACCURACY['oil_variety_lda']:.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 60)
    print("  SAVED MODELS SUMMARY")
    print("=" * 60)
    metadata = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".joblib"):
            continue
        path = os.path.join(MODELS_DIR, fname)
        artifact = joblib.load(path)
        row = {
            "file":           fname,
            "model_type":     type(artifact["model"]).__name__,
            "n_samples":      artifact["n_samples"],
            "n_features":     artifact["n_features"],
            "n_classes":      len(artifact["classes"]),
            "loso_accuracy":  artifact["loso_accuracy"],
            "timestamp":      artifact["timestamp"],
        }
        metadata.append(row)
        print(f"\n  {fname}")
        print(f"    model:       {row['model_type']}")
        print(f"    samples:     {row['n_samples']}")
        print(f"    features:    {row['n_features']}")
        print(f"    classes:     {row['n_classes']}  {artifact['classes']}")
        print(f"    accuracy:    {row['loso_accuracy']:.1%}" if row['loso_accuracy'] else "")

    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved: {meta_path}")
    print(f"\n  All models in: {MODELS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SAVING TRAINED MODELS")
    print("  Full dataset — no held-out split")
    print("=" * 60)

    train_form_detector()
    train_cone_lda()        # LDA first (no auth needed)
    train_cone_tabpfn()     # TabPFN (requires TABPFN_TOKEN)
    train_pellet_lda()
    train_oil_lda()
    print_summary()

    print("\nDone. Models saved to models/")
