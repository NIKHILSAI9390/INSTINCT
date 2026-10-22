from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS = Path(
    r"C:\Research\INSTINCT_reproduction"
    r"\results\MouseBrain_Jiang2023"
)

PCA_FILE = RESULTS / "input_matrix_atac.npy"

META_FILE = RESULTS / "preprocessed_concat_atac.h5ad"


# ============================================================
# 1. Load PCA matrix + metadata
# ============================================================

import anndata as ad

X = np.load(PCA_FILE).astype(np.float64)

adata = ad.read_h5ad(
    META_FILE,
    backed="r"
)

slice_labels = (
    adata.obs["slice_name"]
    .astype(str)
    .to_numpy()
)

annotations = (
    adata.obs["Annotation_for_Combined"]
    .astype(str)
    .to_numpy()
)

assert X.shape == (7113, 100)

print("=" * 70)
print("PCA AUDIT")
print("=" * 70)

print("PCA matrix shape:", X.shape)

slice_order = [
    "E11_0-S1",
    "E13_5-S1",
    "E15_5-S1",
    "E18_5-S1"
]

for s in slice_order:
    print(
        s,
        np.sum(slice_labels == s)
    )


# ============================================================
# 2. Global PCA mean
#
# If PCA scores are centered globally, means should generally
# be close to zero.
# ============================================================

global_mean = X.mean(axis=0)

print("\n" + "=" * 70)
print("GLOBAL PC MEANS")
print("=" * 70)

print(
    "Mean absolute PC mean:",
    np.mean(np.abs(global_mean))
)

print(
    "Maximum absolute PC mean:",
    np.max(np.abs(global_mean))
)

largest_mean_pc = np.argmax(
    np.abs(global_mean)
)

print(
    "PC with largest absolute mean:",
    largest_mean_pc + 1,
    global_mean[largest_mean_pc]
)


# ============================================================
# 3. Variance of each retained PC
#
# IMPORTANT:
# This is variance among the 100 SAVED PC scores.
# It is NOT necessarily the exact original explained_variance_
# from the 64,537-dimensional peak matrix.
# ============================================================

pc_variance = X.var(
    axis=0,
    ddof=1
)

retained_variance_share = (
    pc_variance /
    pc_variance.sum()
)

cumulative_share = np.cumsum(
    retained_variance_share
)

print("\n" + "=" * 70)
print("RETAINED PC VARIANCE")
print("=" * 70)

print(
    "Total variance across 100 retained PCs:",
    pc_variance.sum()
)

for n in [1, 2, 5, 10, 20, 50, 100]:

    print(
        f"Variance share in first {n:3d} PCs:",
        cumulative_share[n - 1]
    )

print("\nTop 10 PCs by variance:")

top_var_idx = np.argsort(
    pc_variance
)[::-1][:10]

for pc in top_var_idx:

    print(
        f"PC{pc + 1:3d}"
        f" variance={pc_variance[pc]:.6f}"
        f" retained_share={retained_variance_share[pc]:.6f}"
    )


# ============================================================
# 4. Slice-wise PC mean + variance
# ============================================================

slice_means = {}
slice_vars = {}

for s in slice_order:

    Xs = X[
        slice_labels == s
    ]

    slice_means[s] = Xs.mean(axis=0)

    slice_vars[s] = Xs.var(
        axis=0,
        ddof=1
    )


# ============================================================
# 5. Which PCs separate slices most?
#
# Between-slice variance / within-slice variance
#
# This is an ANOVA-like descriptive score.
# Higher = stronger slice separation.
# ============================================================

overall_mean = X.mean(axis=0)

between = np.zeros(100)
within = np.zeros(100)

N = len(X)

for s in slice_order:

    mask = slice_labels == s

    ns = mask.sum()

    mean_s = slice_means[s]

    var_s = slice_vars[s]

    between += (
        ns *
        (mean_s - overall_mean) ** 2
    )

    within += (
        (ns - 1) *
        var_s
    )

between /= (
    len(slice_order) - 1
)

within /= (
    N - len(slice_order)
)

slice_separation_score = (
    between /
    np.clip(
        within,
        1e-12,
        None
    )
)

top_slice_pcs = np.argsort(
    slice_separation_score
)[::-1][:20]

print("\n" + "=" * 70)
print("PCS MOST ASSOCIATED WITH SLICE ID")
print("=" * 70)

for pc in top_slice_pcs[:15]:

    print(
        f"\nPC{pc + 1}"
        f" separation_score="
        f"{slice_separation_score[pc]:.4f}"
    )

    for s in slice_order:

        print(
            f"  {s:12s}"
            f" mean={slice_means[s][pc]:9.4f}"
            f" var={slice_vars[s][pc]:9.4f}"
        )


# ============================================================
# 6. Slice centroid distances in PCA space
# ============================================================

centroids = np.vstack([
    slice_means[s]
    for s in slice_order
])

centroid_euclidean = cdist(
    centroids,
    centroids,
    metric="euclidean"
)

centroid_cosine = cdist(
    centroids,
    centroids,
    metric="cosine"
)

print("\n" + "=" * 70)
print("SLICE CENTROID DISTANCES")
print("=" * 70)

print("\nEuclidean:")

print(
    pd.DataFrame(
        centroid_euclidean,
        index=slice_order,
        columns=slice_order
    ).round(4)
)

print("\nCosine:")

print(
    pd.DataFrame(
        centroid_cosine,
        index=slice_order,
        columns=slice_order
    ).round(4)
)


# ============================================================
# 7. E13-E15 cosine audit
#
# Does cosine actually assign lower distances to spots that
# share the same biological annotation?
# ============================================================

idx13 = np.where(
    slice_labels == "E13_5-S1"
)[0]

idx15 = np.where(
    slice_labels == "E15_5-S1"
)[0]

X13 = X[idx13]
X15 = X[idx15]

ann13 = annotations[idx13]
ann15 = annotations[idx15]

print("\n" + "=" * 70)
print("E13-E15 COSINE SIMILARITY AUDIT")
print("=" * 70)

D_cos = cdist(
    X13,
    X15,
    metric="cosine"
)

same_annotation = (
    ann13[:, None]
    ==
    ann15[None, :]
)

same_distances = D_cos[
    same_annotation
]

different_distances = D_cos[
    ~same_annotation
]

print(
    "Same-annotation pairs:",
    len(same_distances)
)

print(
    "Different-annotation pairs:",
    len(different_distances)
)

print(
    "\nSame annotation cosine distance:"
)

print(
    "mean / median:",
    same_distances.mean(),
    np.median(same_distances)
)

print(
    "\nDifferent annotation cosine distance:"
)

print(
    "mean / median:",
    different_distances.mean(),
    np.median(different_distances)
)

print(
    "\nDifference in mean distance "
    "(different - same):",
    different_distances.mean()
    -
    same_distances.mean()
)


# ============================================================
# 8. Per-annotation cosine behavior
# ============================================================

shared_annotations = sorted(
    set(ann13).intersection(
        set(ann15)
    )
)

annotation_rows = []

print("\n" + "=" * 70)
print("PER-ANNOTATION E13-E15 COSINE DISTANCE")
print("=" * 70)

for label in shared_annotations:

    mask13 = ann13 == label
    mask15 = ann15 == label

    same_label_dist = D_cos[
        np.ix_(
            mask13,
            mask15
        )
    ].ravel()

    # Compare source population against ALL E15 spots
    all_target_dist = D_cos[
        mask13,
        :
    ].ravel()

    row = {
        "annotation": label,
        "n_E13": int(mask13.sum()),
        "n_E15": int(mask15.sum()),
        "same_label_mean_cosine":
            float(same_label_dist.mean()),
        "same_label_median_cosine":
            float(np.median(same_label_dist)),
        "all_target_mean_cosine":
            float(all_target_dist.mean()),
        "difference":
            float(
                all_target_dist.mean()
                -
                same_label_dist.mean()
            ),
    }

    annotation_rows.append(row)

    print(
        f"\n{label}"
    )

    print(
        "  E13 / E15:",
        mask13.sum(),
        mask15.sum()
    )

    print(
        "  Same-label mean cosine distance:",
        same_label_dist.mean()
    )

    print(
        "  All-target mean cosine distance:",
        all_target_dist.mean()
    )

    print(
        "  Improvement:",
        row["difference"]
    )


# ============================================================
# 9. Save detailed tables
# ============================================================

pc_table = pd.DataFrame({
    "PC": np.arange(1, 101),
    "global_mean": global_mean,
    "variance": pc_variance,
    "retained_variance_share":
        retained_variance_share,
    "cumulative_retained_share":
        cumulative_share,
    "slice_separation_score":
        slice_separation_score,
})

for s in slice_order:

    pc_table[
        f"{s}_mean"
    ] = slice_means[s]

    pc_table[
        f"{s}_variance"
    ] = slice_vars[s]


pc_table.to_csv(
    ROOT / "pca_pc_statistics.csv",
    index=False
)

pd.DataFrame(
    centroid_euclidean,
    index=slice_order,
    columns=slice_order
).to_csv(
    ROOT / "slice_centroid_euclidean.csv"
)

pd.DataFrame(
    centroid_cosine,
    index=slice_order,
    columns=slice_order
).to_csv(
    ROOT / "slice_centroid_cosine.csv"
)

pd.DataFrame(
    annotation_rows
).to_csv(
    ROOT / "e13_e15_cosine_by_annotation.csv",
    index=False
)

adata.file.close()

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    ROOT / "pca_pc_statistics.csv"
)

print(
    ROOT / "slice_centroid_euclidean.csv"
)

print(
    ROOT / "slice_centroid_cosine.csv"
)

print(
    ROOT / "e13_e15_cosine_by_annotation.csv"
)

print("\nPCA audit completed.")