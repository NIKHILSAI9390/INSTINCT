
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from umap import UMAP
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent

RESULTS = Path(
    r"C:\Research\INSTINCT_reproduction"
    r"\results\MouseBrain_Jiang2023"
)

# ==================================================
# Load JOINT GW result
# ==================================================

data = np.load(
    ROOT / "gw_joint_four_slices_results.npz"
)

indices = data["all_indices"].astype(int)

X_raw = data["X_before"]
X_gw = data["X_after"]

instinct_full = pd.read_csv(
    RESULTS / "comparison" / "INSTINCT"
    / "INSTINCT_embed_2.csv",
    header=None
).to_numpy(dtype=np.float64)

X_instinct = instinct_full[indices]

assert X_raw.shape == (800, 100)
assert X_gw.shape == (800, 200)
assert X_instinct.shape == (800, 30)

print("Raw PCA:", X_raw.shape)
print("Joint GW:", X_gw.shape)
print("INSTINCT:", X_instinct.shape)


# ==================================================
# Load matching metadata
# ==================================================

adata = ad.read_h5ad(
    RESULTS / "preprocessed_concat_atac.h5ad",
    backed="r"
)

slice_labels = (
    adata.obs["slice_name"]
    .astype(str)
    .to_numpy()[indices]
)

annotations = (
    adata.obs["Annotation_for_Combined"]
    .astype(str)
    .to_numpy()[indices]
)

adata.file.close()

slice_order = [
    "E11_0-S1",
    "E13_5-S1",
    "E15_5-S1",
    "E18_5-S1"
]

for i, name in enumerate(slice_order):
    assert np.all(
        slice_labels[i * 200:(i + 1) * 200] == name
    )

print("Spot ordering verified.")


# ==================================================
# UMAP
# ==================================================

def make_umap(X, metric):

    assert np.isfinite(X).all()

    model = UMAP(
        n_neighbors=30,
        n_components=2,
        metric=metric,
        min_dist=0.3,
        random_state=1234
    )

    return model.fit_transform(X)


print("\nCalculating Raw PCA UMAP...")
U_raw = make_umap(X_raw, "correlation")

print("Calculating JOINT GW UMAP...")
U_gw = make_umap(X_gw, "correlation")

print("Calculating INSTINCT UMAP...")
U_instinct = make_umap(X_instinct, "correlation")


representations = [
    ("Before alignment: Raw PCA", U_raw),
    ("After alignment: Joint GW", U_gw),
    ("After alignment: INSTINCT run 2", U_instinct)
]


# ==================================================
# Figure A — slice labels
# ==================================================

slice_colors = {
    "E11_0-S1": "#3B59C7",
    "E13_5-S1": "#C73B40",
    "E15_5-S1": "#39A65A",
    "E18_5-S1": "#989631"
}

point_colors = [
    slice_colors[x]
    for x in slice_labels
]

fig, axes = plt.subplots(
    1, 3, figsize=(16, 5)
)

for ax, (title, U) in zip(
    axes, representations
):

    ax.scatter(
        U[:, 0],
        U[:, 1],
        c=point_colors,
        s=10,
        alpha=0.8,
        linewidths=0
    )

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markersize=8,
        label=name
    )
    for name, color in slice_colors.items()
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

fig.tight_layout(rect=[0, 0.09, 1, 1])

fig.savefig(
    ROOT / "four_slices_joint_alignment_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ==================================================
# Figure B — biological annotation
# ==================================================

unique_labels = sorted(
    np.unique(annotations)
)

cmap = plt.get_cmap("tab20")

anno_map = {
    label: cmap(i % 20)
    for i, label in enumerate(unique_labels)
}

anno_colors = [
    anno_map[label]
    for label in annotations
]

fig, axes = plt.subplots(
    1, 3, figsize=(16, 5)
)

for ax, (title, U) in zip(
    axes, representations
):

    ax.scatter(
        U[:, 0],
        U[:, 1],
        c=anno_colors,
        s=10,
        alpha=0.8,
        linewidths=0
    )

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

anno_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=anno_map[label],
        markersize=7,
        label=label
    )
    for label in unique_labels
]

fig.legend(
    handles=anno_handles,
    loc="center left",
    bbox_to_anchor=(0.995, 0.5),
    fontsize=8,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 0.78, 1])

fig.savefig(
    ROOT / "four_slices_joint_annotation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved:")
print("four_slices_joint_alignment_comparison.png")
print("four_slices_joint_annotation_comparison.png")
print("\nPlotting completed.")
