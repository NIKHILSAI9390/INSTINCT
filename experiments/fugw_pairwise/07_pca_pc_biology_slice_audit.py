from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

from scipy.stats import spearmanr


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
# 1. Load original PCA matrix + metadata
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

print("=" * 100)
print("PER-PC BIOLOGY / SLICE AUDIT")
print("=" * 100)

print("\nFull PCA matrix:", X.shape)


# ============================================================
# 2. Global PC variance
#
# Same variance definition as your cosine audit.
# No beta weighting is applied here.
# ============================================================

pc_variance = X.var(
    axis=0,
    ddof=1
)

if np.any(pc_variance <= 0):
    raise ValueError(
        "One or more PCs have zero/non-positive variance."
    )

print(
    "\nPC variance min / median / max:",
    pc_variance.min(),
    np.median(pc_variance),
    pc_variance.max()
)


# ============================================================
# 3. Select E13 and E15
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
# 4. Shared annotations
# ============================================================

shared_annotations = sorted(
    set(ann13).intersection(
        set(ann15)
    )
)

print(
    "\nShared annotations:",
    len(shared_annotations)
)

for label in shared_annotations:
    print(
        f"{label:32s} "
        f"E13={np.sum(ann13 == label):4d} "
        f"E15={np.sum(ann15 == label):4d}"
    )


# ============================================================
# Helper:
# annotation eta^2 inside ONE slice
#
# Measures how much variation along a PC is explained
# by annotation.
#
# eta^2 near 0:
# annotations overlap strongly
#
# eta^2 larger:
# PC separates annotations better
# ============================================================

def annotation_eta_squared(values, labels):

    grand_mean = np.mean(values)

    ss_total = np.sum(
        (values - grand_mean) ** 2
    )

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for label in np.unique(labels):

        group = values[
            labels == label
        ]

        if len(group) == 0:
            continue

        group_mean = np.mean(group)

        ss_between += (
            len(group)
            *
            (group_mean - grand_mean) ** 2
        )

    return float(
        ss_between / ss_total
    )


# ============================================================
# 5. Audit PCs
#
# Start with first 20 PCs because those contain most of the
# extreme variance structure.
#
# We can extend to all 100 later.
# ============================================================

N_PCS = 20

rows = []


for k in range(N_PCS):

    pc_number = k + 1

    x13 = X13[:, k]
    x15 = X15[:, k]


    # ========================================================
    # A. GLOBAL SLICE EFFECT
    #
    # How much does the entire E13 distribution shift
    # relative to E15?
    #
    # Standardized mean difference:
    #
    # |mean13 - mean15| / pooled SD
    #
    # LARGE = strong developmental/slice effect
    # ========================================================

    mean13 = np.mean(x13)
    mean15 = np.mean(x15)

    var13 = np.var(
        x13,
        ddof=1
    )

    var15 = np.var(
        x15,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (var13 + var15) / 2.0
    )

    if pooled_sd > 1e-12:

        slice_effect_d = (
            abs(mean13 - mean15)
            /
            pooled_sd
        )

    else:

        slice_effect_d = 0.0


    # ========================================================
    # B. ANNOTATION SIGNAL WITHIN EACH SLICE
    #
    # Important:
    # calculate separately for E13 and E15.
    #
    # This avoids calling something "annotation signal"
    # merely because annotation and developmental stage
    # happen to be confounded.
    # ========================================================

    eta13 = annotation_eta_squared(
        x13,
        ann13
    )

    eta15 = annotation_eta_squared(
        x15,
        ann15
    )

    mean_annotation_eta2 = (
        eta13 + eta15
    ) / 2.0


    # ========================================================
    # C. ANNOTATION CENTROIDS IN EACH SLICE
    #
    # Example:
    #
    # PC1:
    #
    # E13:
    # cartilage = high
    # brain     = low
    #
    # E15:
    # cartilage = high
    # brain     = low
    #
    # Then centroid correlation should be high.
    #
    # We use Spearman because relative annotation ordering
    # matters more than exact scale.
    # ========================================================

    centroids13 = []
    centroids15 = []

    for label in shared_annotations:

        vals13 = x13[
            ann13 == label
        ]

        vals15 = x15[
            ann15 == label
        ]

        if (
            len(vals13) > 0
            and
            len(vals15) > 0
        ):

            centroids13.append(
                np.mean(vals13)
            )

            centroids15.append(
                np.mean(vals15)
            )

    if len(centroids13) >= 3:

        centroid_corr, _ = spearmanr(
            centroids13,
            centroids15
        )

        centroid_corr = float(
            centroid_corr
        )

    else:

        centroid_corr = np.nan


    # ========================================================
    # D. SAME-ANNOTATION DEVELOPMENTAL SHIFT
    #
    # For each annotation:
    #
    # |mean_E13 - mean_E15|
    # --------------------
    # pooled within-label SD
    #
    # Then average over annotations.
    #
    # LOW = same biology stays aligned across slices
    # HIGH = annotation moves strongly between stages
    # ========================================================

    annotation_shifts = []

    for label in shared_annotations:

        vals13 = x13[
            ann13 == label
        ]

        vals15 = x15[
            ann15 == label
        ]

        if (
            len(vals13) < 2
            or
            len(vals15) < 2
        ):
            continue

        m13 = np.mean(vals13)
        m15 = np.mean(vals15)

        v13 = np.var(
            vals13,
            ddof=1
        )

        v15 = np.var(
            vals15,
            ddof=1
        )

        within_sd = np.sqrt(
            (v13 + v15) / 2.0
        )

        if within_sd > 1e-12:

            shift = (
                abs(m13 - m15)
                /
                within_sd
            )

            annotation_shifts.append(
                shift
            )

    if len(annotation_shifts) > 0:

        same_annotation_shift = float(
            np.mean(annotation_shifts)
        )

        median_annotation_shift = float(
            np.median(annotation_shifts)
        )

    else:

        same_annotation_shift = np.nan
        median_annotation_shift = np.nan


    # ========================================================
    # Store
    # ========================================================

    rows.append(
        {
            "PC":
                pc_number,

            "variance":
                float(
                    pc_variance[k]
                ),

            "slice_effect_d":
                float(
                    slice_effect_d
                ),

            "annotation_eta2_E13":
                eta13,

            "annotation_eta2_E15":
                eta15,

            "mean_annotation_eta2":
                mean_annotation_eta2,

            "centroid_corr":
                centroid_corr,

            "mean_same_annotation_shift":
                same_annotation_shift,

            "median_same_annotation_shift":
                median_annotation_shift,
        }
    )


# ============================================================
# 6. DataFrame
# ============================================================

df = pd.DataFrame(
    rows
)


# ============================================================
# 7. Main table
# ============================================================

print("\n\n")
print("=" * 110)
print("PER-PC AUDIT")
print("=" * 110)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:9.4f}"
    )
)


# ============================================================
# 8. Rankings
# ============================================================

print("\n\n")
print("=" * 110)
print("MOST SLICE / DEVELOPMENT DOMINATED PCs")
print("=" * 110)

print(
    df.sort_values(
        "slice_effect_d",
        ascending=False
    )[
        [
            "PC",
            "variance",
            "slice_effect_d",
            "mean_annotation_eta2",
            "centroid_corr",
            "mean_same_annotation_shift",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:9.4f}"
    )
)


print("\n\n")
print("=" * 110)
print("MOST ANNOTATION-INFORMATIVE PCs")
print("=" * 110)

print(
    df.sort_values(
        "mean_annotation_eta2",
        ascending=False
    )[
        [
            "PC",
            "variance",
            "slice_effect_d",
            "mean_annotation_eta2",
            "centroid_corr",
            "mean_same_annotation_shift",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:9.4f}"
    )
)


print("\n\n")
print("=" * 110)
print("MOST CROSS-SLICE STABLE BIOLOGICAL PCs")
print("=" * 110)

print(
    df.sort_values(
        [
            "centroid_corr",
            "mean_same_annotation_shift",
        ],
        ascending=[
            False,
            True,
        ]
    )[
        [
            "PC",
            "variance",
            "slice_effect_d",
            "mean_annotation_eta2",
            "centroid_corr",
            "mean_same_annotation_shift",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:9.4f}"
    )
)


# ============================================================
# 9. Save
# ============================================================

OUTPUT_FILE = (
    ROOT /
    "pca_pc_biology_slice_audit.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

adata.file.close()

print("\n\nSaved:")
print(OUTPUT_FILE)

print(
    "\nPC biology/slice audit completed."
)