"""
Unsupervised clustering analysis for the hop e-nose dataset.

Research questions:
  1. Without labels, does E-nose data naturally group into 9 classes?
  2. Do Saaz varieties (Fresh/Aged/Dry/2005) form a super-cluster?
  3. Does K-Means find k=9 as optimal (elbow / silhouette)?
  4. Does session drift create within-class sub-clusters?

Methods:  K-Means  |  Agglomerative (Ward)  |  DBSCAN

Metrics:  ARI  |  NMI  |  Silhouette  |  Davies-Bouldin  |  Calinski-Harabasz

Figures generated:
  clustering_elbow.pdf           — inertia + silhouette sweep over k
  clustering_kmeans_pca.pdf      — K-Means (k=9) vs true labels on PCA
  clustering_dendrogram.pdf      — hierarchical dendrogram coloured by class
  clustering_heatmap.pdf         — cross-tabulation cluster x class heatmap
  clustering_metrics.pdf         — ARI/NMI/Silhouette bar chart all methods
  clustering_variety_pca.pdf     — PCA coloured by hop VARIETY (not condition)

Usage:
    cd "<repo root>"
    python run_clustering.py > results/clustering_log.txt 2>&1
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from itertools import product

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from enose.session_loader import load_with_sessions
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

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

# Map class → hop variety (ignoring condition/year)
VARIETY_MAP = {
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
# Data loading + global MinMax normalisation
# ─────────────────────────────────────────────────────────────────────────────
def load_and_flatten():
    data = load_with_sessions(CLASS_FILES, DATA_DIR)
    X3d = data["X_raw_3d"]          # (130, 54, 7)
    y   = data["y"]
    session_ids = data["session_ids"]
    classes     = data["classes"]

    N, T, S = X3d.shape
    scaler  = MinMaxScaler()
    scaler.fit(X3d.reshape(-1, S))
    X_norm  = scaler.transform(X3d.reshape(-1, S)).reshape(N, T, S)
    # curve-flatten: (N, S, T) → (N, S*T)
    X_flat  = X_norm.transpose(0, 2, 1).reshape(N, S * T)

    varieties = np.array([VARIETY_MAP[c] for c in y])
    return X_flat, y, session_ids, classes, varieties, data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def cluster_metrics(X, y_true, y_pred, method_name):
    valid = y_pred != -1          # exclude DBSCAN noise
    n_clusters = len(set(y_pred[valid])) if valid.any() else 0
    n_noise    = int((y_pred == -1).sum())

    ari = adjusted_rand_score(y_true[valid], y_pred[valid]) if valid.sum() > 1 else 0.0
    nmi = normalized_mutual_info_score(y_true[valid], y_pred[valid]) if valid.sum() > 1 else 0.0

    if n_clusters >= 2 and valid.sum() > n_clusters:
        sil = silhouette_score(X[valid], y_pred[valid])
        db  = davies_bouldin_score(X[valid], y_pred[valid])
        ch  = calinski_harabasz_score(X[valid], y_pred[valid])
    else:
        sil, db, ch = float("nan"), float("nan"), float("nan")

    return {
        "method":     method_name,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "ARI":        round(ari, 4),
        "NMI":        round(nmi, 4),
        "Silhouette": round(sil, 4) if not np.isnan(sil) else None,
        "Davies-Bouldin": round(db, 4) if not np.isnan(db) else None,
        "Calinski-Harabasz": round(ch, 1) if not np.isnan(ch) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — K-Means elbow + silhouette sweep
# ─────────────────────────────────────────────────────────────────────────────
def fig_elbow(X, y, classes):
    print("\n[1/6] K-Means elbow + silhouette sweep...")
    plt, sns = _plt()

    ks         = range(2, 16)
    inertias   = []
    silhouettes = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(ks), inertias, "o-", color="#2196F3", linewidth=2)
    ax1.axvline(9, color="red", linestyle="--", linewidth=1, label="k=9 (true classes)")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.set_title("K-Means Elbow Method")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(ks), silhouettes, "s-", color="#4CAF50", linewidth=2)
    ax2.axvline(9, color="red", linestyle="--", linewidth=1, label="k=9 (true classes)")
    best_k = list(ks)[int(np.argmax(silhouettes))]
    ax2.axvline(best_k, color="orange", linestyle=":", linewidth=1.5,
                label=f"best silhouette k={best_k}")
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Score vs k")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimal k selection for hop e-nose data", fontweight="bold")
    fig.tight_layout()
    save(fig, "clustering_elbow.pdf", plt)

    print(f"  Best silhouette at k={best_k}  (silhouette={max(silhouettes):.4f})")
    print(f"  Silhouette at k=9: {silhouettes[9-2]:.4f}")
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — K-Means (k=9) vs true labels on PCA
# ─────────────────────────────────────────────────────────────────────────────
def fig_kmeans_pca(X, y, classes, best_k):
    print("\n[2/6] K-Means (k=9) vs true labels on PCA...")
    plt, sns = _plt()

    pca   = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    ev    = pca.explained_variance_ratio_

    km9     = KMeans(n_clusters=9, random_state=SEED, n_init=10)
    km_best = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    labels9    = km9.fit_predict(X)
    labels_best = km_best.fit_predict(X)

    palette9    = sns.color_palette("tab10", 9)
    palette_cls = sns.color_palette("tab10", len(classes))
    markers     = ["o","s","^","D","v","P","*","X","h"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: True labels
    ax = axes[0]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette_cls[i], marker=markers[i % len(markers)],
                   label=cls, alpha=0.75, edgecolors="k", linewidth=0.4, s=55)
    ax.set_title("True labels (9 classes)", fontweight="bold")
    ax.legend(fontsize=6, loc="best", framealpha=0.8)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # Panel B: K-Means k=9
    ax = axes[1]
    for ci in range(9):
        mask = labels9 == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette9[ci], alpha=0.75, edgecolors="k",
                   linewidth=0.4, s=55, label=f"C{ci}")
    ax.set_title("K-Means k=9 clusters", fontweight="bold")
    ax.legend(fontsize=6, loc="best", framealpha=0.8)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # Panel C: K-Means k=best_k
    palette_best = sns.color_palette("Set2", best_k)
    ax = axes[2]
    for ci in range(best_k):
        mask = labels_best == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette_best[ci], alpha=0.75, edgecolors="k",
                   linewidth=0.4, s=55, label=f"C{ci}")
    ax.set_title(f"K-Means k={best_k} (best silhouette)", fontweight="bold")
    ax.legend(fontsize=6, loc="best", framealpha=0.8)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    ari9 = adjusted_rand_score(y, labels9)
    nmi9 = normalized_mutual_info_score(y, labels9)
    fig.suptitle(
        f"K-Means clustering vs true labels  |  k=9: ARI={ari9:.3f}  NMI={nmi9:.3f}",
        fontweight="bold"
    )
    fig.tight_layout()
    save(fig, "clustering_kmeans_pca.pdf", plt)

    return labels9, labels_best, X_pca


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Hierarchical dendrogram
# ─────────────────────────────────────────────────────────────────────────────
def fig_dendrogram(X, y, classes, session_ids):
    print("\n[3/6] Hierarchical clustering dendrogram...")
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    plt, sns = _plt()

    # Use session centroids (25 points) rather than all 130 — cleaner dendrogram
    all_sessions = np.unique(session_ids)
    centroids    = np.array([X[session_ids == s].mean(axis=0) for s in all_sessions])
    session_classes = [y[session_ids == s][0] for s in all_sessions]

    Z = linkage(centroids, method="ward")

    # Colour leaves by true class
    class_to_idx = {c: i for i, c in enumerate(classes)}
    palette = sns.color_palette("tab10", len(classes))

    # Build label colour list
    leaf_colours = [palette[class_to_idx[sc]] for sc in session_classes]
    short_labels = [s.replace("_S", "\nS") for s in all_sessions]

    fig, ax = plt.subplots(figsize=(14, 6))
    dend = dendrogram(Z, labels=short_labels, ax=ax,
                      color_threshold=0, above_threshold_color="gray",
                      leaf_rotation=60, leaf_font_size=7)

    # Colour x-tick labels by class
    for lbl, colour in zip(ax.get_xticklabels(),
                           [leaf_colours[i] for i in dend["leaves"]]):
        lbl.set_color(colour)

    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Hierarchical Clustering Dendrogram — session centroids\n"
                 "(leaf colour = true class)")

    # Class legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=palette[class_to_idx[c]], label=c) for c in classes]
    ax.legend(handles=legend_els, fontsize=7, loc="upper right",
              framealpha=0.9, ncol=2)
    fig.tight_layout()
    save(fig, "clustering_dendrogram.pdf", plt)

    # Also compute Ward k=9 clustering labels for metrics
    ward_labels = fcluster(Z, t=9, criterion="maxclust") - 1
    # Map back from sessions to cycles
    ward_cycle_labels = np.array([
        ward_labels[np.where(all_sessions == s)[0][0]] for s in session_ids
    ])
    return ward_cycle_labels


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Cross-tabulation heatmap (cluster x true class)
# ─────────────────────────────────────────────────────────────────────────────
def fig_crosstab_heatmap(y, labels9, classes):
    print("\n[4/6] Cross-tabulation heatmap (K-Means k=9)...")
    plt, sns = _plt()

    ct = pd.crosstab(
        pd.Series(labels9, name="K-Means cluster"),
        pd.Series(y,       name="True class"),
    )
    ct = ct[classes]   # enforce column order

    # Normalise by row (cluster composition)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Raw counts
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, ax=axes[0], cbar=False)
    axes[0].set_title("K-Means k=9 — raw cycle counts per cluster x class")
    axes[0].set_xlabel("True class")
    axes[0].set_ylabel("K-Means cluster")
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                             rotation=35, ha="right", fontsize=8)

    # Normalised
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("K-Means k=9 — fraction of cluster assigned to each class")
    axes[1].set_xlabel("True class")
    axes[1].set_ylabel("K-Means cluster")
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                             rotation=35, ha="right", fontsize=8)

    fig.tight_layout()
    save(fig, "clustering_heatmap.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Metrics summary bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_metrics_summary(metrics_rows):
    print("\n[5/6] Metrics summary bar chart...")
    plt, sns = _plt()

    df = pd.DataFrame(metrics_rows)
    valid = df[df["ARI"].notna() & df["NMI"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    palette = sns.color_palette("Set2", len(valid))

    for ax, metric in zip(axes, ["ARI", "NMI", "Silhouette"]):
        if metric not in valid.columns:
            continue
        vals = valid[metric].fillna(0)
        bars = ax.bar(valid["method"], vals, color=palette)
        ax.set_ylim(0, min(1.05, max(vals) * 1.15 + 0.05))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by clustering method")
        ax.set_xticklabels(valid["method"], rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Clustering quality metrics — hop e-nose data (9 true classes)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "clustering_metrics.pdf", plt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — PCA coloured by HOP VARIETY (not condition)
# ─────────────────────────────────────────────────────────────────────────────
def fig_variety_pca(X, X_pca, y, varieties, classes):
    print("\n[6/6] PCA coloured by hop variety...")
    plt, sns = _plt()

    unique_varieties = sorted(set(varieties))
    pal = sns.color_palette("Set1", len(unique_varieties))
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: coloured by variety
    ax = axes[0]
    for i, var in enumerate(unique_varieties):
        mask = varieties == var
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=pal[i], marker=markers[i % len(markers)],
                   label=var, alpha=0.75, edgecolors="k", linewidth=0.4, s=60)
    ax.set_title("PCA — coloured by hop variety\n(groups classes from same variety)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Right: coloured by class (original 9)
    ax = axes[1]
    pal9    = sns.color_palette("tab10", len(classes))
    mks     = ["o","s","^","D","v","P","*","X","h"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=pal9[i], marker=mks[i % len(mks)],
                   label=cls, alpha=0.75, edgecolors="k", linewidth=0.4, s=55)
    ax.set_title("PCA — coloured by class\n(variety + condition, 9 labels)")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Do E-nose signatures reveal hop variety vs condition structure?",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "clustering_variety_pca.pdf", plt)

    # ARI/NMI for variety-level labels
    km5 = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    labels5 = km5.fit_predict(X)
    ari5 = adjusted_rand_score(varieties, labels5)
    nmi5 = normalized_mutual_info_score(varieties, labels5)
    print(f"  K-Means k=5 (variety-level) -> ARI={ari5:.4f}  NMI={nmi5:.4f}")

    # ARI/NMI for condition labels (fresh/aged/dry)
    condition_map = {
        "Chinook_Fresh": "Fresh",  "Chinook_Aged":  "Aged",
        "Comet_2025":    "Fresh",  "Saaz_2005":     "Aged",
        "Saaz_Fresh":    "Fresh",  "Saaz_Aged":     "Aged",
        "Saaz_Dry":      "Dry",    "Vista_Fresh":   "Fresh",
        "Zeus_2025":     "Fresh",
    }
    conditions = np.array([condition_map[c] for c in y])
    km3 = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels3 = km3.fit_predict(X)
    ari3 = adjusted_rand_score(conditions, labels3)
    nmi3 = normalized_mutual_info_score(conditions, labels3)
    print(f"  K-Means k=3 (condition-level) -> ARI={ari3:.4f}  NMI={nmi3:.4f}")

    return ari5, nmi5, ari3, nmi3


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN parameter search
# ─────────────────────────────────────────────────────────────────────────────
def run_dbscan(X, y):
    print("\n  DBSCAN parameter search...")
    best = {"sil": -1, "params": None, "labels": None}

    for eps in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        for min_s in [2, 3, 5]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int((labels == -1).sum())
            if n_clusters < 2 or n_clusters > 20:
                continue
            valid = labels != -1
            if valid.sum() < n_clusters + 1:
                continue
            sil = silhouette_score(X[valid], labels[valid])
            if sil > best["sil"]:
                best = {"sil": sil, "params": (eps, min_s), "labels": labels,
                        "n_clusters": n_clusters, "n_noise": n_noise}

    if best["params"]:
        eps, min_s = best["params"]
        print(f"  Best DBSCAN: eps={eps}  min_samples={min_s}  "
              f"n_clusters={best['n_clusters']}  noise={best['n_noise']}  "
              f"silhouette={best['sil']:.4f}")
    else:
        print("  DBSCAN: no configuration found with 2-20 clusters")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  E-Nose Hop Clustering Analysis")
    print("=" * 65)

    X, y, session_ids, classes, varieties, data = load_and_flatten()
    print(f"\nX shape: {X.shape}  |  classes: {classes}")

    # PCA for visualisation (reused across figures)
    pca   = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)

    # ── Elbow ──────────────────────────────────────────────────────────────
    best_k = fig_elbow(X, y, classes)

    # ── K-Means k=9 ────────────────────────────────────────────────────────
    labels9, labels_best, _ = fig_kmeans_pca(X, y, classes, best_k)

    # ── Dendrogram ─────────────────────────────────────────────────────────
    ward_labels = fig_dendrogram(X, y, classes, session_ids)

    # ── Heatmap ────────────────────────────────────────────────────────────
    fig_crosstab_heatmap(y, labels9, classes)

    # ── DBSCAN ─────────────────────────────────────────────────────────────
    dbscan_best = run_dbscan(X, y)

    # ── Variety PCA ────────────────────────────────────────────────────────
    ari5, nmi5, ari3, nmi3 = fig_variety_pca(X, X_pca, y, varieties, classes)

    # ── Compute all metrics ────────────────────────────────────────────────
    print("\n  Computing cluster metrics...")
    metrics_rows = []

    # K-Means k=9
    metrics_rows.append(cluster_metrics(X, y, labels9,     "KMeans-9"))
    metrics_rows.append(cluster_metrics(X, y, labels_best, f"KMeans-{best_k}"))
    # K-Means k=5 (variety-level)
    km5 = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    metrics_rows.append(cluster_metrics(X, varieties, km5.fit_predict(X), "KMeans-5(variety)"))
    # Ward k=9
    metrics_rows.append(cluster_metrics(X, y, ward_labels, "Ward-9"))
    # DBSCAN best
    if dbscan_best["params"]:
        metrics_rows.append(cluster_metrics(
            X, y, dbscan_best["labels"],
            f"DBSCAN(eps={dbscan_best['params'][0]},min={dbscan_best['params'][1]})"
        ))

    fig_metrics_summary(metrics_rows)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CLUSTERING RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  {'Method':<35}  {'k':>3}  {'ARI':>6}  {'NMI':>6}  {'Silhouette':>10}")
    print("  " + "-" * 65)
    for r in metrics_rows:
        sil = f"{r['Silhouette']:.4f}" if r["Silhouette"] is not None else "  N/A"
        print(f"  {r['method']:<35}  {r['n_clusters']:>3}  "
              f"{r['ARI']:>6.4f}  {r['NMI']:>6.4f}  {sil:>10}")

    print(f"\n  Variety-level (k=5): ARI={ari5:.4f}  NMI={nmi5:.4f}")
    print(f"  Condition-level (k=3): ARI={ari3:.4f}  NMI={nmi3:.4f}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    results = {
        "metrics": metrics_rows,
        "variety_kmeans5": {"ARI": ari5, "NMI": nmi5},
        "condition_kmeans3": {"ARI": ari3, "NMI": nmi3},
        "best_k_silhouette": int(best_k),
    }
    out_path = os.path.join(RES_DIR, "clustering_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 65)
    print("  Clustering complete. Figures in figures/  results in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
