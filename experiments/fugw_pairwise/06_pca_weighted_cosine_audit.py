from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

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
ADATA_FILE = RESULTS / "preprocessed_concat_atac.h5ad"


# ============================================================
# 1. Load complete PCA matrix + labels
# ============================================================

X = np.load(
    PCA_FILE
).astype(np.float64)

adata = ad.read_h5ad(
    ADATA_FILE,
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


# ============================================================
# 2. PCA variance used for whitening
#
# Global PCA variance across ALL 7113 spots.
# ============================================================

pc_variance = X.var(
    axis=0,
    ddof=1
)

if np.any(pc_variance <= 0):
    raise ValueError(
        "One or more PCs have zero/non-positive variance."
    )

print("=" * 72)
print("PCA WEIGHTED COSINE AUDIT")
print("=" * 72)

print(
    "PC variance min / median / max:",
    pc_variance.min(),
    np.median(pc_variance),
    pc_variance.max()
)


# ============================================================
# 3. E13 and E15
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

print("\nE13:", X13.shape)
print("E15:", X15.shape)


# ============================================================
# 4. Ground-truth evaluation mask
#
# Labels are ONLY used for evaluation.
# They do not enter construction of molecular distance.
# ============================================================

same_annotation = (
    ann13[:, None]
    ==
    ann15[None, :]
)

shared_annotations = sorted(
    set(ann13).intersection(
        set(ann15)
    )
)

print(
    "\nSame-annotation pairs:",
    same_annotation.sum()
)

print(
    "Different-annotation pairs:",
    (~same_annotation).sum()
)


# ============================================================
# 5. Beta values
# ============================================================

BETAS = [
    0.00,
    0.10,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.75,
    1.00
]

# ============================================================
# 6. Storage
# ============================================================

global_rows = []
annotation_rows = []


# ============================================================
# 7. Evaluate every beta
# ============================================================

for beta in BETAS:

    print("\n")
    print("=" * 72)
    print(f"BETA = {beta}")
    print("=" * 72)

    # --------------------------------------------------------
    # Dimension weights
    #
    # z_k = x_k / lambda_k^(beta/2)
    # --------------------------------------------------------

    scale = (
        pc_variance **
        (beta / 2.0)
    )

    Z13 = X13 / scale
    Z15 = X15 / scale

    # --------------------------------------------------------
    # Confirm variance transformation
    #
    # beta=0 -> original variance
    # beta=1 -> approximately unit variance
    # --------------------------------------------------------

    Z_all = X / scale

    transformed_variance = Z_all.var(
        axis=0,
        ddof=1
    )

    print(
        "Transformed variance "
        "min / median / max:"
    )

    print(
        transformed_variance.min(),
        np.median(transformed_variance),
        transformed_variance.max()
    )

    # --------------------------------------------------------
    # Cross-slice cosine distance
    # --------------------------------------------------------

    print("Computing E13-E15 cosine matrix...")

    D = cdist(
        Z13,
        Z15,
        metric="cosine"
    )

    assert np.isfinite(D).all()

    same = D[
        same_annotation
    ]

    different = D[
        ~same_annotation
    ]

    same_mean = float(
        same.mean()
    )

    different_mean = float(
        different.mean()
    )

    same_median = float(
        np.median(same)
    )

    different_median = float(
        np.median(different)
    )

    mean_gap = (
        different_mean
        -
        same_mean
    )

    median_gap = (
        different_median
        -
        same_median
    )

    print(
        "\nSame annotation mean:",
        same_mean
    )

    print(
        "Different annotation mean:",
        different_mean
    )

    print(
        "Mean discrimination gap:",
        mean_gap
    )

    print(
        "\nSame annotation median:",
        same_median
    )

    print(
        "Different annotation median:",
        different_median
    )

    print(
        "Median discrimination gap:",
        median_gap
    )


    # ========================================================
    # 8. Nearest-neighbor biological agreement
    #
    # For each E13 spot:
    # which E15 spot has minimum molecular distance?
    # ========================================================

    nearest_idx = np.argmin(
        D,
        axis=1
    )

    nearest_labels = ann15[
        nearest_idx
    ]

    top1_accuracy = float(
        np.mean(
            nearest_labels == ann13
        )
    )

    nearest_distance = D[
        np.arange(len(X13)),
        nearest_idx
    ]

    print(
        "\nCross-slice molecular NN "
        "same-annotation fraction:",
        top1_accuracy
    )

    print(
        "Mean nearest-neighbor distance:",
        nearest_distance.mean()
    )


    # ========================================================
    # 9. Top-5 biological agreement
    #
    # Does at least one of the five closest target spots
    # share the source annotation?
    # ========================================================

    top5_idx = np.argpartition(
        D,
        kth=4,
        axis=1
    )[:, :5]

    top5_labels = ann15[
        top5_idx
    ]

    top5_match = np.any(
        top5_labels
        ==
        ann13[:, None],
        axis=1
    )

    top5_accuracy = float(
        top5_match.mean()
    )

    print(
        "Top-5 contains same annotation:",
        top5_accuracy
    )


    # ========================================================
    # 10. Per-annotation behavior
    # ========================================================

    print("\nPER-ANNOTATION")

    per_label_improvements = []

    for label in shared_annotations:

        mask13 = ann13 == label
        mask15 = ann15 == label

        same_label_dist = D[
            np.ix_(
                mask13,
                mask15
            )
        ].ravel()

        all_target_dist = D[
            mask13,
            :
        ].ravel()

        same_label_mean = float(
            same_label_dist.mean()
        )

        all_target_mean = float(
            all_target_dist.mean()
        )

        improvement = (
            all_target_mean
            -
            same_label_mean
        )

        per_label_improvements.append(
            improvement
        )

        # ----------------------------------------------------
        # NN accuracy specifically for this annotation
        # ----------------------------------------------------

        source_rows = np.where(
            mask13
        )[0]

        label_nn_accuracy = float(
            np.mean(
                nearest_labels[source_rows]
                ==
                label
            )
        )

        annotation_rows.append(
            {
                "beta": beta,
                "annotation": label,
                "n_E13": int(mask13.sum()),
                "n_E15": int(mask15.sum()),

                "same_label_mean_distance":
                    same_label_mean,

                "all_target_mean_distance":
                    all_target_mean,

                "improvement":
                    improvement,

                "nearest_neighbor_accuracy":
                    label_nn_accuracy,
            }
        )

        print(
            f"{label:32s}"
            f" improvement={improvement:8.4f}"
            f" NN_acc={label_nn_accuracy:7.4f}"
        )


    # ========================================================
    # 11. Global summary
    # ========================================================

    median_annotation_improvement = float(
        np.median(
            per_label_improvements
        )
    )

    mean_annotation_improvement = float(
        np.mean(
            per_label_improvements
        )
    )

    global_rows.append(
        {
            "beta": beta,

            "same_mean":
                same_mean,

            "different_mean":
                different_mean,

            "mean_gap":
                mean_gap,

            "same_median":
                same_median,

            "different_median":
                different_median,

            "median_gap":
                median_gap,

            "top1_accuracy":
                top1_accuracy,

            "top5_accuracy":
                top5_accuracy,

            "mean_nearest_distance":
                float(
                    nearest_distance.mean()
                ),

            "mean_annotation_improvement":
                mean_annotation_improvement,

            "median_annotation_improvement":
                median_annotation_improvement,

            "transformed_variance_min":
                float(
                    transformed_variance.min()
                ),

            "transformed_variance_max":
                float(
                    transformed_variance.max()
                ),
        }
    )

    del D
    del Z13
    del Z15
    del Z_all


# ============================================================
# 12. Final comparison
# ============================================================

summary = pd.DataFrame(
    global_rows
)

annotation_df = pd.DataFrame(
    annotation_rows
)

print("\n\n")
print("=" * 100)
print("FINAL MOLECULAR-SIMILARITY COMPARISON")
print("=" * 100)

print(
    summary[
        [
            "beta",
            "mean_gap",
            "median_gap",
            "top1_accuracy",
            "top5_accuracy",
            "mean_annotation_improvement",
            "median_annotation_improvement",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. Find best beta under several criteria
#
# We intentionally do NOT choose one automatically.
# ============================================================

best_mean_gap = summary.loc[
    summary["mean_gap"].idxmax(),
    "beta"
]

best_top1 = summary.loc[
    summary["top1_accuracy"].idxmax(),
    "beta"
]

best_top5 = summary.loc[
    summary["top5_accuracy"].idxmax(),
    "beta"
]

best_median_annotation = summary.loc[
    summary[
        "median_annotation_improvement"
    ].idxmax(),
    "beta"
]

print("\nBEST BETA BY CRITERION")

print(
    "Mean same-vs-different gap:",
    best_mean_gap
)

print(
    "Top-1 biological NN:",
    best_top1
)

print(
    "Top-5 biological NN:",
    best_top5
)

print(
    "Median per-annotation improvement:",
    best_median_annotation
)


# ============================================================
# 14. Save
# ============================================================

summary.to_csv(
    ROOT /
    "pca_weighted_cosine_summary.csv",
    index=False
)

annotation_df.to_csv(
    ROOT /
    "pca_weighted_cosine_by_annotation.csv",
    index=False
)

adata.file.close()

print("\nSaved:")

print(
    ROOT /
    "pca_weighted_cosine_summary.csv"
)

print(
    ROOT /
    "pca_weighted_cosine_by_annotation.csv"
)

print(
    "\nWeighted PCA cosine audit completed."
)

for beta in [0.0, 0.25, 0.35, 0.4, 0.5, 0.75, 1.0]:
    weights = pc_variance ** (-beta / 2)

    print(f"\nBETA = {beta}")
    print("PC   Variance        Weight")
    for k in range(20):
        print(
            f"{k+1:2d}   "
            f"{pc_variance[k]:10.4f}   "
            f"{weights[k]:10.4f}"
        )
        
        
