# enose/loso_cv.py
"""
Leave-One-Session-Out (LOSO) cross-validation engine.

Scientific rationale
--------------------
Each measurement session = the e-nose connected to one physical hop sample on
one day, producing 5 consecutive cycles. Cycles within a session are correlated
(same sample, same environment, same day). A random split that places some of
these cycles in train and others in test creates data leakage — the model has
already "seen" that sample — inflating accuracy.

LOSO evaluates the more realistic question:
    "Can the model correctly classify a measurement session it has NEVER seen?"

This is the strongest possible test for generalization to new hop samples.
"""

import warnings
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_cycles(X_3d, method="curve"):
    """
    Flatten (n_cycles, time_steps, n_sensors) → (n_cycles, features).

    method='curve'      [s1_t0..tN, s2_t0..tN, ...]  sensor-by-sensor
    method='interleave' [t0_s1..sN, t1_s1..sN, ...]  time-step-by-time-step
    """
    n, T, S = X_3d.shape
    if method == "curve":
        # (n, T, S) → (n, S, T) → (n, S*T)
        return X_3d.transpose(0, 2, 1).reshape(n, S * T)
    elif method == "interleave":
        # (n, T, S) → (n, T*S)
        return X_3d.reshape(n, T * S)
    else:
        raise ValueError(f"Unknown flatten method: '{method}'. Use 'curve' or 'interleave'.")


def _compute_gmean(y_true, y_pred, classes):
    """Geometric mean of per-class recall × specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    class_counts = cm.sum(axis=1)
    recalls = np.divide(
        cm.diagonal(), class_counts,
        out=np.zeros_like(cm.diagonal(), dtype=float),
        where=class_counts != 0,
    )
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    gmeans = np.sqrt(np.array(recalls) * np.array(specificities))
    return float(np.mean(np.nan_to_num(gmeans)))


# ---------------------------------------------------------------------------
# Main LOSO runner
# ---------------------------------------------------------------------------

def run_loso(data, model_factory, normalization="minmax", flatten_method="curve",
             verbose=True):
    """
    Leave-One-Session-Out cross-validation.

    Each fold holds out ALL cycles of one session as the test set and trains
    on the remaining sessions. The scaler is fit only on training cycles.

    Args:
        data          : dict returned by session_loader.load_with_sessions()
        model_factory : callable() → a fresh, unfitted sklearn-compatible classifier
                        Called once per fold to guarantee independence.
        normalization : 'minmax' or 'zscore'
        flatten_method: 'curve' or 'interleave'
        verbose       : print per-fold results

    Returns:
        dict with keys:
            fold_results     — list of per-fold dicts
            n_folds          — number of valid folds
            mean_accuracy    — mean per-fold accuracy (weighted by n_test)
            std_accuracy     — std of per-fold accuracy
            mean_f1          — mean per-fold F1
            std_f1           — std of per-fold F1
            overall_accuracy — accuracy over all concatenated test predictions
            overall_f1       — F1 over all concatenated test predictions
            overall_gmean    — G-Mean over all concatenated test predictions
            confusion_matrix — aggregated confusion matrix (list of lists)
            classes          — class label order for confusion matrix
            skipped_folds    — list of sessions that were skipped (class not in train)
    """
    X_3d = data["X_raw_3d"]        # (n_cycles, time_steps, n_sensors)
    y = data["y"]
    session_ids = data["session_ids"]
    classes = data["classes"]

    loso = LeaveOneGroupOut()

    fold_results = []
    skipped_folds = []
    all_y_true = []
    all_y_pred = []

    n_folds_total = loso.get_n_splits(X_3d, y, groups=session_ids)
    if verbose:
        print(f"  Total sessions (folds): {n_folds_total}")
        print(f"  {'Fold':>4}  {'Session':<32}  {'n_test':>6}  {'Acc':>6}  {'F1':>6}")
        print("  " + "-" * 62)

    for fold_idx, (train_idx, test_idx) in enumerate(
        loso.split(X_3d, y, groups=session_ids)
    ):
        held_out_session = session_ids[test_idx[0]]

        X_train_3d = X_3d[train_idx]
        X_test_3d  = X_3d[test_idx]
        y_train    = y[train_idx]
        y_test     = y[test_idx]

        # Check all test classes appear in training set
        train_classes = set(np.unique(y_train))
        test_classes  = set(np.unique(y_test))
        missing = test_classes - train_classes
        if missing:
            warnings.warn(
                f"Fold {fold_idx} ({held_out_session}): class(es) {missing} absent "
                f"from training set — fold skipped."
            )
            skipped_folds.append(held_out_session)
            continue

        # Flatten
        X_train_flat = flatten_cycles(X_train_3d, flatten_method)
        X_test_flat  = flatten_cycles(X_test_3d,  flatten_method)

        # Normalize — scaler fit on training cycles ONLY
        if normalization == "minmax":
            scaler = MinMaxScaler()
        elif normalization == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization: '{normalization}'")

        X_train_norm = scaler.fit_transform(X_train_flat)
        X_test_norm  = scaler.transform(X_test_flat)

        # Train a fresh model instance for this fold
        clf = model_factory()
        clf.fit(X_train_norm, y_train)
        y_pred = clf.predict(X_test_norm)

        fold_acc = float(accuracy_score(y_test, y_pred))
        fold_f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        fold_results.append({
            "fold":     fold_idx,
            "session":  held_out_session,
            "n_test":   int(len(y_test)),
            "accuracy": fold_acc,
            "f1":       fold_f1,
            "y_true":   y_test.tolist(),
            "y_pred":   y_pred.tolist(),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        if verbose:
            print(
                f"  {fold_idx+1:4d}  {held_out_session:<32}  "
                f"{len(y_test):6d}  {fold_acc:6.3f}  {fold_f1:6.3f}"
            )

    if not fold_results:
        return {"error": "No valid folds completed.", "skipped_folds": skipped_folds}

    if skipped_folds and verbose:
        print(f"\n  WARNING: {len(skipped_folds)} fold(s) skipped: {skipped_folds}")

    # Aggregate metrics
    accs = np.array([r["accuracy"] for r in fold_results])
    f1s  = np.array([r["f1"]       for r in fold_results])

    all_y_true_arr = np.array(all_y_true)
    all_y_pred_arr = np.array(all_y_pred)

    overall_acc   = float(accuracy_score(all_y_true_arr, all_y_pred_arr))
    overall_f1    = float(f1_score(all_y_true_arr, all_y_pred_arr, average="weighted", zero_division=0))
    overall_gmean = _compute_gmean(all_y_true_arr, all_y_pred_arr, classes)
    cm_agg        = confusion_matrix(all_y_true_arr, all_y_pred_arr, labels=classes)

    if verbose:
        print("  " + "-" * 62)
        print(f"  Mean  acc: {accs.mean():.3f} ± {accs.std():.3f}")
        print(f"  Mean  F1 : {f1s.mean():.3f} ± {f1s.std():.3f}")
        print(f"  Overall acc (concatenated): {overall_acc:.3f}")
        print(f"  Overall F1  (concatenated): {overall_f1:.3f}")
        print(f"  G-Mean (concatenated):      {overall_gmean:.3f}")

    return {
        "fold_results":      fold_results,
        "n_folds":           len(fold_results),
        "skipped_folds":     skipped_folds,
        "mean_accuracy":     float(accs.mean()),
        "std_accuracy":      float(accs.std()),
        "mean_f1":           float(f1s.mean()),
        "std_f1":            float(f1s.std()),
        "overall_accuracy":  overall_acc,
        "overall_f1":        overall_f1,
        "overall_gmean":     overall_gmean,
        "confusion_matrix":  cm_agg.tolist(),
        "classes":           classes,
    }
