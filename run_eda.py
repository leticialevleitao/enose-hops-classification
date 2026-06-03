"""
Exploratory Data Analysis (EDA) for the hop e-nose dataset.

Generates publication-quality figures:
  1.  eda_class_distribution.pdf  — cycles per class and sessions per class
  2.  eda_sensor_curves.pdf        — mean ± std sensor response curve per class
  3.  eda_pca_class.pdf            — PCA 2D scatter coloured by class
  4.  eda_pca_session.pdf          — PCA 2D scatter coloured by session within class
  5.  eda_tsne_class.pdf           — t-SNE 2D scatter coloured by class
  6.  eda_sensor_r2.pdf            — R² discriminatory power per sensor per class pair
  7.  eda_inter_session_dist.pdf   — pairwise L2 distance between sessions within same class
  8.  eda_intra_vs_inter.pdf       — intra-class vs inter-class session distance (box plot)

Usage:
    cd "C:/Users/t-leticiad/OneDrive - Microsoft/Desktop/TCC/portable-enose-hops-master (github repo)"
    python run_eda.py > results/eda_run_log.txt 2>&1
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "consolidated")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
RES_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── class → file map (cone dataset only) ─────────────────────────────────────
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

# ── matplotlib / seaborn (lazy import to avoid buffering issues) ──────────────
def _import_plot_libs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    return plt, mpatches, sns


def save(fig, name, plt):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    sys.path.insert(0, BASE_DIR)
    from enose.session_loader import load_with_sessions

    print("Loading data with session labels...")
    data = load_with_sessions(CLASS_FILES, DATA_DIR)

    X3d          = data["X_raw_3d"]          # (130, 54, 7)
    y            = data["y"]                 # (130,)
    session_ids  = data["session_ids"]       # (130,)
    sensors      = data["selected_sensors"]  # 7 sensor names
    min_len      = data["min_len"]
    classes      = data["classes"]
    sessions_per_class = data["sessions_per_class"]

    print(f"\nDataset summary:")
    print(f"  Cycles  : {len(y)}")
    print(f"  Classes : {len(classes)}  -> {classes}")
    print(f"  Sessions: {data['n_sessions']}")
    print(f"  Shape   : {X3d.shape}  (n_cycles × time_steps × n_sensors)")
    print(f"  Sensors : {sensors}")

    return X3d, y, session_ids, sensors, classes, sessions_per_class


# ─────────────────────────────────────────────────────────────────────────────
# 2. Flatten helper (MinMax per-cycle, then concatenate = 'curve' method)
# ─────────────────────────────────────────────────────────────────────────────
def flatten_minmax(X3d):
    """Per-sensor MinMax normalisation then curve-flatten → (N, T*S)."""
    from sklearn.preprocessing import MinMaxScaler
    N, T, S = X3d.shape
    X_flat = np.zeros((N, T * S), dtype=np.float32)
    for i in range(N):
        cycle = X3d[i]          # (T, S)
        scaled = MinMaxScaler().fit_transform(cycle)   # per-sensor min-max
        X_flat[i] = scaled.T.flatten(order='C')        # curve: sensor by sensor
    return X_flat


def flatten_global(X3d):
    """Global MinMax across the entire training set then curve-flatten."""
    from sklearn.preprocessing import MinMaxScaler
    N, T, S = X3d.shape
    # Fit scaler on all data stacked as (N*T, S)
    scaler = MinMaxScaler()
    scaler.fit(X3d.reshape(-1, S))
    X_flat = np.zeros((N, T * S), dtype=np.float32)
    for i in range(N):
        scaled = scaler.transform(X3d[i])    # (T, S)
        X_flat[i] = scaled.T.flatten(order='C')
    return X_flat, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Class & session distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_distribution(y, session_ids, classes, sessions_per_class):
    print("\n[1/8] Class & session distribution...")
    plt, mpatches, sns = _import_plot_libs()

    # Count cycles and sessions per class
    cycles_count   = {c: int(np.sum(y == c)) for c in classes}
    sessions_count = {c: len(sessions_per_class[c]) for c in classes}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # cycles
    ax = axes[0]
    bars = ax.barh(classes, [cycles_count[c] for c in classes],
                   color=sns.color_palette("tab10", len(classes)))
    ax.set_xlabel("Number of cycles")
    ax.set_title("Cycles per class")
    for bar, val in zip(bars, [cycles_count[c] for c in classes]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    # sessions
    ax = axes[1]
    bars = ax.barh(classes, [sessions_count[c] for c in classes],
                   color=sns.color_palette("tab10", len(classes)))
    ax.set_xlabel("Number of sessions")
    ax.set_title("Sessions per class (each = 5 cycles, 1 physical measurement)")
    ax.set_xlim(0, max(sessions_count.values()) + 1)
    for bar, val in zip(bars, [sessions_count[c] for c in classes]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    fig.suptitle("Dataset Overview — Cone Hops E-Nose (9 classes)", fontweight="bold")
    fig.tight_layout()
    save(fig, "eda_class_distribution.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Sensor response curves per class
# ─────────────────────────────────────────────────────────────────────────────
def fig_sensor_curves(X3d, y, classes, sensors):
    print("\n[2/8] Sensor response curves...")
    plt, mpatches, sns = _import_plot_libs()

    n_sensors = len(sensors)
    palette   = sns.color_palette("tab10", len(classes))

    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2.5 * n_sensors), sharex=True)

    T = X3d.shape[1]
    t = np.arange(T)

    for s_idx, (ax, sensor_name) in enumerate(zip(axes, sensors)):
        for c_idx, cls in enumerate(classes):
            mask = y == cls
            curves = X3d[mask, :, s_idx]          # (n_cycles_in_class, T)
            mean   = curves.mean(axis=0)
            std    = curves.std(axis=0)
            color  = palette[c_idx]
            ax.plot(t, mean, color=color, linewidth=1.4, label=cls)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_ylabel(sensor_name, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
    axes[0].set_title("Mean ± std sensor response per class (all sessions pooled)")
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    save(fig, "eda_sensor_curves.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — PCA coloured by class
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_class(X_flat, y, classes):
    print("\n[3/8] PCA coloured by class...")
    from sklearn.decomposition import PCA
    plt, mpatches, sns = _import_plot_libs()

    pca   = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_flat)
    ev    = pca.explained_variance_ratio_

    palette = sns.color_palette("tab10", len(classes))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette[i], marker=markers[i % len(markers)],
                   label=cls, alpha=0.75, edgecolors="k", linewidth=0.4, s=55)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(f"PCA — 9-class hop e-nose signatures  (PC1+PC2 = {sum(ev)*100:.1f}% variance)")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "eda_pca_class.pdf", plt)

    # Print explained variance for all components
    pca_full = PCA(random_state=SEED).fit(X_flat)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"  PCA: {n90} components -> 90%, {n95} -> 95%, {n99} -> 99% variance")
    print(f"  PC1={ev[0]*100:.1f}%  PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    return X_pca, pca


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — PCA coloured by session (within each class)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_session(X_pca, y, session_ids, classes):
    print("\n[4/8] PCA coloured by session...")
    plt, mpatches, sns = _import_plot_libs()

    # Use a consistent session colour per class (each class gets its own subplot)
    n_cols = 3
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for ax_idx, cls in enumerate(classes):
        ax   = axes[ax_idx]
        mask = y == cls
        sids = np.unique(session_ids[mask])
        pal  = sns.color_palette("husl", len(sids))
        for s_idx, sid in enumerate(sids):
            smask = (y == cls) & (session_ids == sid)
            short = sid.split("_S")[-1]          # just the session number
            ax.scatter(X_pca[smask, 0], X_pca[smask, 1],
                       color=pal[s_idx], label=f"S{short}",
                       alpha=0.8, edgecolors="k", linewidth=0.4, s=60)
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)

    # Hide unused axes
    for ax_idx in range(len(classes), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("PCA per class — coloured by measurement session\n"
                 "(spread = inter-session variance within same variety)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "eda_pca_session.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — t-SNE coloured by class
# ─────────────────────────────────────────────────────────────────────────────
def fig_tsne_class(X_flat, y, classes):
    print("\n[5/8] t-SNE coloured by class...")
    from sklearn.manifold import TSNE
    plt, mpatches, sns = _import_plot_libs()

    perplexity = min(30, len(X_flat) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                max_iter=1000, init="pca")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_tsne = tsne.fit_transform(X_flat)

    palette = sns.color_palette("tab10", len(classes))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   color=palette[i], marker=markers[i % len(markers)],
                   label=cls, alpha=0.75, edgecolors="k", linewidth=0.4, s=55)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE (perplexity={perplexity}) — 9-class hop e-nose signatures")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "eda_tsne_class.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Sensor R² discriminatory power
# ─────────────────────────────────────────────────────────────────────────────
def fig_sensor_r2(X3d, y, sensors, classes):
    print("\n[6/8] Sensor R² discriminatory power...")
    plt, mpatches, sns = _import_plot_libs()

    n_sensors = len(sensors)
    T         = X3d.shape[1]

    # Compute R² per sensor (class-level ANOVA-style: between-class / total variance)
    r2_scores = {}
    for s_idx, sensor_name in enumerate(sensors):
        # Mean response per cycle for this sensor
        sensor_means = X3d[:, :, s_idx].mean(axis=1)   # (N,)
        grand_mean   = sensor_means.mean()
        ss_total     = np.sum((sensor_means - grand_mean) ** 2)
        ss_within    = sum(
            np.sum((sensor_means[y == cls] - sensor_means[y == cls].mean()) ** 2)
            for cls in classes
        )
        r2 = (ss_total - ss_within) / ss_total if ss_total > 0 else 0.0
        r2_scores[sensor_name] = float(r2)

    sorted_sensors = sorted(r2_scores, key=r2_scores.get, reverse=True)
    sorted_r2      = [r2_scores[s] for s in sorted_sensors]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sorted_sensors, sorted_r2,
                   color=sns.color_palette("RdYlGn", n_sensors))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R² (between-class variance / total variance)")
    ax.set_title("Sensor discriminatory power — higher R² = more useful for classification")
    for bar, val in zip(bars, sorted_r2):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "eda_sensor_r2.pdf", plt)

    print("  Sensor R² scores:")
    for s in sorted_sensors:
        print(f"    {s:10s}: {r2_scores[s]:.4f}")

    return r2_scores


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Pairwise session distance within same class
# ─────────────────────────────────────────────────────────────────────────────
def fig_inter_session_dist(X_flat, y, session_ids, classes, sessions_per_class):
    """
    For each class: compute centroid of each session in feature space,
    then show pairwise L2 distances between session centroids as a heatmap.
    """
    print("\n[7/8] Inter-session distance heatmap...")
    plt, mpatches, sns = _import_plot_libs()

    # How many subplots?  Only classes with >= 2 sessions
    eligible = [c for c in classes if len(sessions_per_class[c]) >= 2]
    n_cols = 3
    n_rows = int(np.ceil(len(eligible) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for ax_idx, cls in enumerate(eligible):
        ax   = axes[ax_idx]
        sids = sessions_per_class[cls]

        # Compute session centroids in flat feature space
        centroids = []
        for sid in sids:
            mask     = (y == cls) & (session_ids == sid)
            centroid = X_flat[mask].mean(axis=0)
            centroids.append(centroid)

        # Pairwise L2 distances
        n = len(sids)
        dist_matrix = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            d = np.linalg.norm(centroids[i] - centroids[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

        short_labels = [sid.split(f"{cls}_")[-1] for sid in sids]
        sns.heatmap(dist_matrix, annot=True, fmt=".2f",
                    xticklabels=short_labels, yticklabels=short_labels,
                    cmap="YlOrRd", ax=ax, cbar=False,
                    annot_kws={"size": 8})
        ax.set_title(cls, fontsize=9, fontweight="bold")

    for ax_idx in range(len(eligible), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Pairwise L2 distance between session centroids (within same class)\n"
        "High distance = sessions look different in sensor space → harder to classify",
        fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "eda_inter_session_dist.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Intra-class vs inter-class session distance (box plot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_intra_vs_inter(X_flat, y, session_ids, classes, sessions_per_class):
    """
    Compare within-class session distances to between-class session distances.
    Good classification requires: inter-class >> intra-class.
    """
    print("\n[8/8] Intra vs inter-class session distance...")
    plt, mpatches, sns = _import_plot_libs()

    # Compute one centroid per session
    all_sessions = np.unique(session_ids)
    centroids = {}
    session_to_class = {}
    for sid in all_sessions:
        mask = session_ids == sid
        centroids[sid] = X_flat[mask].mean(axis=0)
        session_to_class[sid] = y[mask][0]

    intra_distances = []
    inter_distances = []

    for si, sj in combinations(all_sessions, 2):
        d = float(np.linalg.norm(centroids[si] - centroids[sj]))
        if session_to_class[si] == session_to_class[sj]:
            intra_distances.append(d)
        else:
            inter_distances.append(d)

    df = pd.DataFrame({
        "Distance": intra_distances + inter_distances,
        "Type": (["Intra-class\n(same variety)"] * len(intra_distances) +
                 ["Inter-class\n(different varieties)"] * len(inter_distances))
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Type", y="Distance", palette=["#2196F3", "#F44336"],
                width=0.4, ax=ax, order=["Intra-class\n(same variety)",
                                          "Inter-class\n(different varieties)"])
    sns.stripplot(data=df, x="Type", y="Distance", color="black", alpha=0.4,
                  size=3, jitter=True, ax=ax,
                  order=["Intra-class\n(same variety)",
                         "Inter-class\n(different varieties)"])

    ax.set_ylabel("L2 distance between session centroids")
    ax.set_xlabel("")
    ax.set_title("Intra-class vs inter-class session distances\n"
                 "(Good separability: inter >> intra)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "eda_intra_vs_inter.pdf", plt)

    print(f"  Intra-class distances : mean={np.mean(intra_distances):.3f}  "
          f"std={np.std(intra_distances):.3f}  n={len(intra_distances)}")
    print(f"  Inter-class distances : mean={np.mean(inter_distances):.3f}  "
          f"std={np.std(inter_distances):.3f}  n={len(inter_distances)}")

    ratio = np.mean(inter_distances) / np.mean(intra_distances)
    print(f"  Inter/Intra ratio     : {ratio:.2f}x  "
          f"({'good separability' if ratio > 2 else 'poor separability — classes overlap in session space'})")

    return intra_distances, inter_distances


# ─────────────────────────────────────────────────────────────────────────────
# Save numeric EDA summary
# ─────────────────────────────────────────────────────────────────────────────
def save_eda_summary(y, session_ids, classes, sessions_per_class, r2_scores,
                     intra_distances, inter_distances):
    summary = {
        "n_cycles":   int(len(y)),
        "n_classes":  len(classes),
        "n_sessions": int(len(np.unique(session_ids))),
        "cycles_per_class": {c: int(np.sum(y == c)) for c in classes},
        "sessions_per_class": {c: len(sessions_per_class[c]) for c in classes},
        "sensor_r2": r2_scores,
        "session_distances": {
            "intra_mean": float(np.mean(intra_distances)),
            "intra_std":  float(np.std(intra_distances)),
            "inter_mean": float(np.mean(inter_distances)),
            "inter_std":  float(np.std(inter_distances)),
            "inter_over_intra_ratio": float(np.mean(inter_distances) / np.mean(intra_distances))
        }
    }

    out_path = os.path.join(RES_DIR, "eda_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  E-Nose Hop EDA")
    print("=" * 65)

    # 1. Load
    X3d, y, session_ids, sensors, classes, sessions_per_class = load_data()

    # 2. Flatten (global MinMax, curve layout — consistent with LOSO pipeline)
    print("\nFlattening with global MinMax normalisation...")
    X_flat, _ = flatten_global(X3d)
    print(f"  X_flat shape: {X_flat.shape}")

    # 3. Figures
    fig_distribution(y, session_ids, classes, sessions_per_class)
    fig_sensor_curves(X3d, y, classes, sensors)
    X_pca, pca = fig_pca_class(X_flat, y, classes)
    fig_pca_session(X_pca, y, session_ids, classes)
    fig_tsne_class(X_flat, y, classes)
    r2_scores = fig_sensor_r2(X3d, y, sensors, classes)
    fig_inter_session_dist(X_flat, y, session_ids, classes, sessions_per_class)
    intra, inter = fig_intra_vs_inter(X_flat, y, session_ids, classes, sessions_per_class)

    # 4. Save JSON summary
    save_eda_summary(y, session_ids, classes, sessions_per_class,
                     r2_scores, intra, inter)

    print("\n" + "=" * 65)
    print("  EDA complete. All figures in figures/  results in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
