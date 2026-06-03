"""
run_production_pipeline.py
==========================
End-to-end inference pipeline for blind hop sample classification.

Architecture (2-stage routing):
    Stage 1 — Form classifier (LDA, 91.7% LOGO)
              Detects: cone | pellet | oil
    Stage 2 — Variety specialist (routed by Stage 1)
              cone   → TabPFN v3  (82.4% LOSO)  or LDA fallback (70.0%)
              pellet → LDA        (93.3% 5-fold)
              oil    → LDA        (95.0% 5-fold)

Usage
-----
    # Demo with synthetic random sample:
    python run_production_pipeline.py --demo

    # Classify a real CSV file (one measurement session):
    python run_production_pipeline.py --csv path/to/sample.csv

    # Batch evaluate the full dataset (with known labels):
    python run_production_pipeline.py --evaluate
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

warnings.filterwarnings("ignore")

import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────
class HopClassificationPipeline:
    """
    Two-stage routing pipeline for hop variety classification.

    Stage 1: detect processing form  (cone / pellet / oil)
    Stage 2: identify variety using the specialist model for that form
    """

    FORM_TO_MODEL = {
        "cone":   "cone_variety_tabpfn",    # primary: TabPFN v3
        "pellet": "pellet_variety_lda",
        "oil":    "oil_variety_lda",
    }
    CONE_FALLBACK = "cone_variety_lda"      # if TabPFN unavailable

    def __init__(self, models_dir=MODELS_DIR, verbose=True):
        self.models_dir = models_dir
        self.verbose    = verbose
        self._cache     = {}
        self._load_all()

    # ── Loading ──────────────────────────────────────────────────────────────
    def _load(self, name):
        if name in self._cache:
            return self._cache[name]
        path = os.path.join(self.models_dir, f"{name}.joblib")
        if not os.path.exists(path):
            return None
        artifact = joblib.load(path)
        self._cache[name] = artifact
        if self.verbose:
            acc = artifact.get("loso_accuracy")
            acc_str = f"{acc:.1%}" if acc else "N/A"
            print(f"  [loaded] {name}  ({type(artifact['model']).__name__}, acc={acc_str})")
        return artifact

    def _load_all(self):
        if self.verbose:
            print("Loading models...")
        self._load("form_detector_lda")
        for model_name in self.FORM_TO_MODEL.values():
            art = self._load(model_name)
            if art is None and model_name == self.FORM_TO_MODEL["cone"]:
                if self.verbose:
                    print(f"  [warn] {model_name} not found, loading fallback: {self.CONE_FALLBACK}")
                self._load(self.CONE_FALLBACK)

    # ── Feature extraction ───────────────────────────────────────────────────
    @staticmethod
    def _flatten(X_3d, n_features_expected=None):
        """
        (N, T, S) → (N, S*T) curve layout.
        Truncates T to match model's expected feature count if needed.
        """
        N, T, S = X_3d.shape
        if n_features_expected is not None:
            T_expected = n_features_expected // S
            if T > T_expected:
                X_3d = X_3d[:, :T_expected, :]
                T = T_expected
        return X_3d.transpose(0, 2, 1).reshape(N, S * T)

    # ── Single-sample inference ───────────────────────────────────────────────
    def predict_one(self, X_3d):
        """
        Classify one sample (or batch of samples from the same session).

        Parameters
        ----------
        X_3d : np.ndarray, shape (N_cycles, T, 7)
            Raw sensor readings for one measurement session.

        Returns
        -------
        dict with keys:
            form          – detected form ('cone', 'pellet', 'oil')
            form_proba    – confidence scores for each form
            variety       – predicted variety label
            variety_proba – confidence scores for each variety (if available)
            pipeline_accuracy_reference – known LOSO/kfold accuracy of the specialist
        """
        # ── Stage 1: form detection ──────────────────────────────────────────
        form_art = self._cache.get("form_detector_lda")
        if form_art is None:
            raise RuntimeError("form_detector_lda not loaded.")

        n_feat_form = form_art["n_features"]
        X_form_flat = self._flatten(X_3d, n_features_expected=n_feat_form)
        X_form_norm = form_art["scaler"].transform(X_form_flat)

        form_pred  = form_art["model"].predict(X_form_norm)
        form_proba = None
        if hasattr(form_art["model"], "predict_proba"):
            proba = form_art["model"].predict_proba(X_form_norm)
            form_proba = dict(zip(form_art["classes"], proba.mean(axis=0)))

        # Majority vote over cycles
        unique, counts = np.unique(form_pred, return_counts=True)
        form = unique[counts.argmax()]

        if self.verbose:
            print(f"\n  Stage 1 — Form detected: {form.upper()}", end="")
            if form_proba:
                top = sorted(form_proba.items(), key=lambda x: -x[1])
                conf_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
                print(f"  (confidence: {conf_str})", end="")
            print()

        # ── Stage 2: variety classification ─────────────────────────────────
        specialist_name = self.FORM_TO_MODEL.get(form)
        specialist_art  = self._cache.get(specialist_name)

        # Fallback to LDA for cones if TabPFN not available
        if specialist_art is None and form == "cone":
            specialist_name = self.CONE_FALLBACK
            specialist_art  = self._cache.get(specialist_name)

        if specialist_art is None:
            raise RuntimeError(f"No specialist model loaded for form='{form}'.")

        n_feat_var = specialist_art["n_features"]
        X_var_flat = self._flatten(X_3d, n_features_expected=n_feat_var)
        X_var_norm = specialist_art["scaler"].transform(X_var_flat)

        variety_pred  = specialist_art["model"].predict(X_var_norm)
        variety_proba = None
        if hasattr(specialist_art["model"], "predict_proba"):
            proba = specialist_art["model"].predict_proba(X_var_norm)
            variety_proba = dict(zip(specialist_art["classes"], proba.mean(axis=0)))

        unique, counts = np.unique(variety_pred, return_counts=True)
        variety = unique[counts.argmax()]

        if self.verbose:
            print(f"  Stage 2 — Variety predicted: {variety}", end="")
            if variety_proba:
                top3 = sorted(variety_proba.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.2f}" for k, v in top3)
                print(f"  (top-3: {top_str})", end="")
            print()

        ref_acc = specialist_art.get("loso_accuracy")
        return {
            "form":           form,
            "form_proba":     form_proba,
            "variety":        variety,
            "variety_proba":  variety_proba,
            "specialist_model": specialist_name,
            "reference_accuracy": ref_acc,
        }

    # ── Batch evaluation ─────────────────────────────────────────────────────
    def evaluate(self, samples):
        """
        Evaluate pipeline on a list of (X_3d, true_form, true_variety) tuples.
        Batches samples by form for efficiency (avoids 125 individual TabPFN calls).
        """
        from sklearn.metrics import accuracy_score
        from collections import defaultdict

        # Group by true form for batch processing
        groups = defaultdict(list)
        for i, (X_3d, true_form, true_variety) in enumerate(samples):
            groups[true_form].append((i, X_3d, true_variety))

        rows = [None] * len(samples)

        for true_form, items in groups.items():
            idxs     = [it[0] for it in items]
            X_list   = [it[1] for it in items]
            varieties = [it[2] for it in items]

            # Stage 1: batch form prediction
            form_art = self._cache["form_detector_lda"]
            # All cycles from this group stacked
            min_T_form = min(x.shape[1] for x in X_list)
            X_all = np.vstack([x[:, :min_T_form, :] for x in X_list])
            X_form_flat = self._flatten(X_all, n_features_expected=form_art["n_features"])
            X_form_norm = form_art["scaler"].transform(X_form_flat)
            form_preds  = form_art["model"].predict(X_form_norm)

            # Stage 2: batch variety prediction for each form
            specialist_name = self.FORM_TO_MODEL.get(true_form)
            spec_art = self._cache.get(specialist_name)
            if spec_art is None and true_form == "cone":
                specialist_name = self.CONE_FALLBACK
                spec_art = self._cache.get(specialist_name)

            min_T_var = min(x.shape[1] for x in X_list)
            X_var_all  = np.vstack([x[:, :min_T_var, :] for x in X_list])
            X_var_flat = self._flatten(X_var_all, n_features_expected=spec_art["n_features"])
            X_var_norm = spec_art["scaler"].transform(X_var_flat)
            var_preds  = spec_art["model"].predict(X_var_norm)

            print(f"  [{true_form:6s}] {len(items):3d} samples — "
                  f"form_acc={accuracy_score(np.repeat([true_form], len(form_preds)), form_preds):.3f}  "
                  f"var_acc={accuracy_score(varieties, var_preds):.3f}")

            for local_i, (global_i, var_true, form_pred, var_pred) in enumerate(
                zip(idxs, varieties, form_preds, var_preds)
            ):
                rows[global_i] = {
                    "true_form":    true_form,
                    "pred_form":    form_pred,
                    "form_correct": form_pred == true_form,
                    "true_variety": var_true,
                    "pred_variety": var_pred,
                    "var_correct":  var_pred == var_true,
                }

        df = pd.DataFrame(rows)
        form_acc    = df["form_correct"].mean()
        var_acc     = df["var_correct"].mean()
        cascade_acc = (df["form_correct"] & df["var_correct"]).mean()

        print(f"\n  {'Metric':<38} {'Value':>7}")
        print("  " + "-" * 47)
        print(f"  {'Stage 1 — Form accuracy':<38} {form_acc:7.3f}")
        print(f"  {'Stage 2 — Variety accuracy (all)':<38} {var_acc:7.3f}")
        print(f"  {'Cascade accuracy (both correct)':<38} {cascade_acc:7.3f}")
        print(f"  {'N samples':<38} {len(df):7d}")
        return df, {"form_acc": form_acc, "variety_acc": var_acc, "cascade_acc": cascade_acc}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_sample(csv_path):
    """Load a single e-nose CSV file and return X_3d (N_cycles, T, 7)."""
    sys.path.insert(0, BASE_DIR)
    from enose.session_loader import _parse_cycles_from_csv
    cycles = _parse_cycles_from_csv(csv_path)
    if not cycles:
        raise ValueError(f"No cycles found in {csv_path}")
    min_T = min(len(c) for c in cycles)
    X_3d  = np.stack([c[:min_T, :] for c in cycles]).astype(np.float32)
    print(f"  Loaded {len(cycles)} cycles, {min_T} time steps, 7 sensors → shape {X_3d.shape}")
    return X_3d


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on full dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_evaluation_set():
    """Build list of (X_3d, true_form, true_variety) from all known CSVs."""
    from run_expanded import CONE_CLASS_FILES, PELLET_CLASS_FILES, OIL_CLASS_FILES, load_form

    DATA_DIR = os.path.join(BASE_DIR, "data", "consolidated")
    samples = []

    for form_label, class_files in [
        ("cone",   CONE_CLASS_FILES),
        ("pellet", PELLET_CLASS_FILES),
        ("oil",    OIL_CLASS_FILES),
    ]:
        X_3d, y, _, _ = load_form(class_files)
        for i, label in enumerate(y):
            samples.append((X_3d[i:i+1], form_label, label))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Demo with synthetic data
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(pipeline):
    print("\n" + "=" * 55)
    print("  DEMO — Synthetic blind sample (random data)")
    print("=" * 55)
    print("  (Real use: replace with actual e-nose CSV readings)")
    X_demo = np.random.rand(5, 54, 7).astype(np.float32)  # 5 cycles, 54 steps, 7 sensors
    result = pipeline.predict_one(X_demo)
    print(f"\n  >> Final prediction: {result['form'].upper()} — {result['variety']}")
    print(f"  >> Specialist model: {result['specialist_model']}")
    if result["reference_accuracy"]:
        print(f"  >> Reference accuracy (validation): {result['reference_accuracy']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hop classification pipeline — 2-stage routing"
    )
    parser.add_argument("--demo",     action="store_true",
                        help="Run demo with synthetic random data")
    parser.add_argument("--csv",      type=str, default=None,
                        help="Path to a single e-nose CSV file for classification")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate pipeline on the full known dataset")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-sample verbose output")
    args = parser.parse_args()

    print("=" * 55)
    print("  HOP CLASSIFICATION PIPELINE")
    print("  Two-stage routing: form → variety specialist")
    print("=" * 55)

    pipeline = HopClassificationPipeline(verbose=not args.quiet)

    if args.csv:
        print(f"\n  Classifying: {args.csv}")
        X_3d  = load_csv_sample(args.csv)
        result = pipeline.predict_one(X_3d)
        print(f"\n  >> RESULT: {result['form'].upper()} — {result['variety']}")

    elif args.evaluate:
        print("\n  Building evaluation set from all known CSVs...")
        samples = build_evaluation_set()
        print(f"  {len(samples)} samples loaded")
        pipeline.verbose = False   # suppress per-sample output
        df, metrics = pipeline.evaluate(samples)
        out_path = os.path.join(BASE_DIR, "results", "pipeline_evaluation.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Detailed results saved: {out_path}")

    else:
        run_demo(pipeline)

    print("\nDone.")


if __name__ == "__main__":
    main()
