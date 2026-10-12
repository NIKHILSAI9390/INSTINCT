from pathlib import Path 
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from umap import UMAP
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent

BASE = Path(
    r"C:\Research\INSTINCT_reproduction"
)

RESULTS = (
    BASE / "results" / "MouseBrain_Jiang2023"
)

pilot = np.load(ROOT / "ot_pilot_data.npz")
gw = np.load(ROOT / "gw_pilot_results.npz")

XA = pilot["XA"]
XB = pilot["XB"]

idx_A = pilot["idx_A"]
idx_B = pilot["idx_B"]

XA_aligned = gw["XA_on_B"]
T = gw["T"]

# validate each shape 
assert XA.shape == (200, 100)
assert XB.shape == (200, 100)
assert XA_aligned.shape == XA.shape
assert T.shape == (200, 200)

assert np.isfinite(XA_aligned).all()

print("Original E13.5:", XA.shape)
print("Original E15.5:", XB.shape)
print("GW aligned E13.5:", XA_aligned.shape)


#construct before and after representation
X_before = np.vstack([XA, XB])

X_after = np.vstack([
    XA_aligned,
    XB
])
slice_labels = np.array(
    ["E13.5"] * 200 + ["E15.5"] * 200
)

#load bio annotations 
adata = ad.read_h5ad(
    RESULTS / "preprocessed_concat_atac.h5ad",
    backed="r"
)

annotations = (
    adata.obs["Annotation_for_Combined"]
    .astype(str)
    .to_numpy()
)

annotation_labels = np.concatenate([
    annotations[idx_A],
    annotations[idx_B]
])

adata.file.close()


# Umap

def run_umap(X):
    reducer = UMAP(
        n_neighbors=30,
        n_components=2,
        metric="correlation",
        min_dist=0.3,
        random_state=1234
    )
    return reducer.fit_transform(X)

print("\nRunning UMAP before alignment...")
U_before = run_umap(X_before)

print("Running UMAP after GW alignment...")
U_after = run_umap(X_after)

#slice-color comparision

slice_colors = {
    "E13.5":"#C41230",
    "E15.5":"#29A329"
}

colors = [
    slice_colors[name] for name in slice_labels
]

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5)
)

for ax, U, title in zip(
    axes,
    [U_before, U_after],
    [
        "Before OT alignment\nOriginal PCA",
        "After GW alignment\nBarycentric projection"
    ]
):

    ax.scatter(
        U[:, 0],
        U[:, 1],
        c=colors,
        s=15,
        alpha=0.8
    )

    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

legend = [
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
    handles=legend,
    loc="lower center",
    ncol=2,
    frameon=False
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig(
    ROOT / "gw_alignment_before_after.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


#bio annotation figure

unique_labels = sorted(
    np.unique(annotation_labels)
)

cmap = plt.get_cmap("tab20")
annotation_map = {
    label: cmap(i % 20)
    for i, label in enumerate(unique_labels)
}

annotation_colors = [
    annotation_map[label]
    for label in annotation_labels
]

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    U_after[:, 0],
    U_after[:, 1],
    c=annotation_colors,
    s=15,
    alpha=0.85
)

ax.set_title(
    "GW alignment: Biological annotations",
    fontsize=14
)

ax.set_xticks([])
ax.set_yticks([])

handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=annotation_map[label],
        markersize=7,
        label=label
    )
    for label in unique_labels
]

ax.legend(
    handles=handles,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
    frameon=False
)

fig.tight_layout()

fig.savefig(
    ROOT / "gw_alignment_annotations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved figures:")
print("1. gw_alignment_before_after.png")
print("2. gw_alignment_annotations.png")
print("\nStep 3 completed.")