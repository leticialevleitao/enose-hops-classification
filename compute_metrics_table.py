"""
compute_metrics_table.py
========================
Produce a comprehensive 4-metric comparison table (Accuracy, Precision,
F1-score, G-mean) for ALL experimental scenarios:

  Scenario A — Cone variety classification   (LOSO, 9 classes)
  Scenario B — Form detection                (LOGO, 3 classes)
  Scenario C — Pellet variety classification (5-fold, 9 classes)
  Scenario D — Oil variety classification    (5-fold, 12 classes)

Outputs
-------
  results/metrics_table_full.csv   — machine-readable
  results/metrics_table.tex        — LaTeX booktabs table for TCC
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")

from enose.session_loader import load_with_sessions
from enose.loso_cv import flatten_cycles, _compute_gmean
from run_expanded import (
    load_form, flatten,
    CONE_CLASS_FILES, PELLET_CLASS_FILES, OIL_CLASS_FILES
)

DATA_DIR = os.path.join(BASE_DIR, "data", "consolidated")
SEED = 42
np.random.seed(SEED)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, classes):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    gm   = _compute_gmean(np.array(y_true), np.array(y_pred), list(classes))
    return dict(accuracy=round(acc,4), precision=round(prec,4),
                f1=round(f1,4), gmean=round(gm,4))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A — Cone LOSO  (read from saved detailed JSON)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_a_from_json():
    """Extract all-model metrics from the two saved detailed JSON files."""
    rows = []
    for fname in ["loso_results_detailed.json", "loso_modern_results_detailed.json"]:
        path = os.path.join(RES_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for model_name, res in data.items():
            if "error" in res or "fold_results" not in res:
                continue
            all_yt, all_yp = [], []
            for fold in res["fold_results"]:
                all_yt.extend(fold["y_true"])
                all_yp.extend(fold["y_pred"])
            classes = res.get("classes", sorted(set(all_yt)))
            m = compute_metrics(all_yt, all_yp, classes)
            rows.append({"scenario": "A_cone_loso", "model": model_name, **m})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B — Form detection  (LOGO: each file = one session)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_b_form_detection():
    print("  [B] Form detection (LOGO)...")
    segments = []
    for form_label, class_files in [("cone", CONE_CLASS_FILES),
                                     ("pellet", PELLET_CLASS_FILES),
                                     ("oil", OIL_CLASS_FILES)]:
        X_3d, y_var, groups, _ = load_form(class_files)
        y_form = np.array([form_label] * len(y_var))
        segments.append((X_3d, y_form, groups))

    min_T = min(X.shape[1] for X, _, _ in segments)
    X_all  = np.vstack([X[:, :min_T, :] for X, _, _ in segments])
    y_all  = np.concatenate([y for _, y, _ in segments])
    grp_all = np.concatenate([g for _, _, g in segments])

    X_flat = flatten(X_all)
    classes = np.unique(y_all)

    models = {
        "LDA":          LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(iterations=200, depth=6,
                                                 learning_rate=0.1, verbose=0,
                                                 random_seed=SEED)
    except ImportError:
        pass

    logo = LeaveOneGroupOut()
    rows = []
    for mname, clf_proto in models.items():
        all_yt, all_yp = [], []
        for train_idx, test_idx in logo.split(X_flat, y_all, groups=grp_all):
            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(X_flat[train_idx])
            Xte = scaler.transform(X_flat[test_idx])
            clf = clf_proto.__class__(**clf_proto.get_params())
            clf.fit(Xtr, y_all[train_idx])
            all_yt.extend(y_all[test_idx].tolist())
            all_yp.extend(clf.predict(Xte).tolist())
        m = compute_metrics(all_yt, all_yp, classes)
        rows.append({"scenario": "B_form_detection", "model": mname, **m})
        print(f"    {mname:20s}  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  f1={m['f1']:.3f}  gm={m['gmean']:.3f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C / D — Variety within form  (stratified 5-fold)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_variety_kfold(label, class_files, scenario_key):
    print(f"  [{label}] {scenario_key} (5-fold StratifiedKFold)...")
    X_3d, y, _, _ = load_form(class_files)
    X_flat = flatten(X_3d)
    classes = np.unique(y)

    models = {
        "LDA":          LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(iterations=200, depth=6,
                                                 learning_rate=0.1, verbose=0,
                                                 random_seed=SEED)
    except ImportError:
        pass

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for mname, clf_proto in models.items():
        all_yt, all_yp = [], []
        for train_idx, test_idx in skf.split(X_flat, y):
            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(X_flat[train_idx])
            Xte = scaler.transform(X_flat[test_idx])
            clf = clf_proto.__class__(**clf_proto.get_params())
            clf.fit(Xtr, y[train_idx])
            all_yt.extend(y[test_idx].tolist())
            all_yp.extend(clf.predict(Xte).tolist())
        m = compute_metrics(all_yt, all_yp, classes)
        rows.append({"scenario": scenario_key, "model": mname, **m})
        print(f"    {mname:20s}  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  f1={m['f1']:.3f}  gm={m['gmean']:.3f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generator
# ─────────────────────────────────────────────────────────────────────────────
SCENARIO_LABELS = {
    "A_cone_loso":       "A — Variedade de Cones (LOSO, 9 classes)",
    "B_form_detection":  "B — Detecção de Forma (LOGO, 3 classes)",
    "C_pellet_variety":  "C — Variedade de Pellets (5-fold, 9 classes)",
    "D_oil_variety":     "D — Variedade de Óleos (5-fold, 12 classes)",
}

# Models to highlight in scenario A (top models only for readability)
TOP_MODELS_A = ["TabPFN", "LDA", "ROCKET", "Arsenal",
                "XGBoost", "LightGBM", "Random Forest",
                "CatBoost", "AutoGluon", "MLP", "SVM"]

def fmt(val):
    """Format float as comma-decimal (Portuguese style)."""
    return f"{val:.3f}".replace(".", ",")

def to_latex(df):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Comparativo de métricas por cenário experimental. "
                 r"Acurácia, Precisão (média ponderada), F1-score (média ponderada) "
                 r"e G-mean calculados sobre os dados concatenados de todas as dobras.}")
    lines.append(r"  \label{tab:metrics_full}")
    lines.append(r"  \begin{tabular}{llrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Cenário} & \textbf{Modelo} & \textbf{Acurácia} "
                 r"& \textbf{Precisão} & \textbf{F1} & \textbf{G-mean} \\")
    lines.append(r"    \midrule")

    for sc_key, sc_label in SCENARIO_LABELS.items():
        sc_df = df[df["scenario"] == sc_key].copy()
        if sc_key == "A_cone_loso":
            # Keep only top models and sort by accuracy
            sc_df = sc_df[sc_df["model"].isin(TOP_MODELS_A)]
        sc_df = sc_df.sort_values("accuracy", ascending=False)
        if sc_df.empty:
            continue

        # Scenario header row
        lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{6}}{{l}}{{\textit{{{sc_label}}}}} \\")
        lines.append(r"    \midrule")

        for i, (_, row) in enumerate(sc_df.iterrows()):
            bold = i == 0  # best model in bold
            mname = row["model"].replace("_", r"\_")
            if bold:
                mname = rf"\textbf{{{mname}}}"
                acc  = rf"\textbf{{{fmt(row['accuracy'])}}}"
                prec = rf"\textbf{{{fmt(row['precision'])}}}"
                f1   = rf"\textbf{{{fmt(row['f1'])}}}"
                gm   = rf"\textbf{{{fmt(row['gmean'])}}}"
            else:
                acc, prec, f1, gm = fmt(row['accuracy']), fmt(row['precision']), fmt(row['f1']), fmt(row['gmean'])
            lines.append(rf"    & {mname} & {acc} & {prec} & {f1} & {gm} \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  COMPUTING COMPREHENSIVE METRICS TABLE")
    print("=" * 60)

    all_rows = []

    print("\n[A] Cone variety LOSO — reading from saved JSON...")
    all_rows.extend(scenario_a_from_json())
    a_models = [r["model"] for r in all_rows if r["scenario"] == "A_cone_loso"]
    print(f"    {len(a_models)} models loaded: {a_models}")

    all_rows.extend(scenario_b_form_detection())
    all_rows.extend(scenario_variety_kfold("C", PELLET_CLASS_FILES, "C_pellet_variety"))
    all_rows.extend(scenario_variety_kfold("D", OIL_CLASS_FILES,    "D_oil_variety"))

    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = os.path.join(RES_DIR, "metrics_table_full.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV saved: {csv_path}")

    # Save LaTeX
    latex_str = to_latex(df)
    tex_path = os.path.join(RES_DIR, "metrics_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[OK] LaTeX saved: {tex_path}")

    # Print preview
    print("\n" + "=" * 60)
    print("  RESULTS PREVIEW")
    print("=" * 60)
    for sc in SCENARIO_LABELS:
        sc_df = df[df["scenario"] == sc].sort_values("accuracy", ascending=False)
        if sc_df.empty:
            continue
        print(f"\n  {SCENARIO_LABELS[sc]}")
        print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'F1':>6} {'G-mean':>7}")
        print("  " + "-" * 50)
        for _, r in sc_df.iterrows():
            if sc == "A_cone_loso" and r["model"] not in TOP_MODELS_A:
                continue
            print(f"  {r['model']:<22} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['f1']:6.3f} {r['gmean']:7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
