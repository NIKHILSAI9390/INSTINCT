import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from umap.umap_ import UMAP
from matplotlib.lines import Line2D


# --------------------------------------------------
# Paths
# --------------------------------------------------

save_dir = r"C:\Research\INSTINCT_reproduction\results\MouseBrain_Jiang2023"

adata_path = os.path.join(
    save_dir,
    "preprocessed_concat_atac.h5ad"
)

raw_pca_path = os.path.join(
    save_dir,
    "input_matrix_atac.npy"
)

instinct_path = os.path.join(
    save_dir,
    "comparison",
    "INSTINCT",
    "INSTINCT_embed_7.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

adata = ad.read_h5ad(adata_path)

raw_pca = np.load(raw_pca_path)

instinct_embed = pd.read_csv(
    instinct_path,
    header=None
).values


print("adata shape:", adata.shape)
print("Raw PCA shape:", raw_pca.shape)
print("INSTINCT embedding shape:", instinct_embed.shape)

print("\nSlice counts:")
print(adata.obs["slice_name"].value_counts())


# --------------------------------------------------
# Sanity check
# --------------------------------------------------

assert raw_pca.shape[0] == adata.shape[0]
assert instinct_embed.shape[0] == adata.shape[0]

print("\nAll row counts match.")


# --------------------------------------------------
# Same UMAP settings as authors' plot_umap.py
# --------------------------------------------------

def make_umap(X):

    reducer = UMAP(
        n_neighbors=30,
        n_components=2,
        metric="correlation",
        n_epochs=None,
        learning_rate=1.0,
        min_dist=0.3,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        repulsion_strength=1,
        negative_sample_rate=5,
        random_state=1234,
        angular_rp_forest=False,
        verbose=False
    )

    return reducer.fit_transform(X)


print("\nRunning UMAP on raw PCA...")
raw_umap = make_umap(raw_pca)

print("Running UMAP on INSTINCT latent representation...")
instinct_umap = make_umap(instinct_embed)


# --------------------------------------------------
# Slice colors
# --------------------------------------------------

slice_names = [
    "E11_0-S1",
    "E13_5-S1",
    "E15_5-S1",
    "E18_5-S1"
]

colors_for_slices = [
    [0.2298057, 0.29871797, 0.75368315],
    [0.70567316, 0.01555616, 0.15023281],
    [0.2298057, 0.70567316, 0.15023281],
    [0.5830223, 0.59200322, 0.12993134]
]

slice_cmap = {
    slice_names[i]: colors_for_slices[i]
    for i in range(len(slice_names))
}

point_colors = list(
    adata.obs["slice_name"].astype(str).map(slice_cmap)
)


# --------------------------------------------------
# Plot RAW vs INSTINCT
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

point_size = 8


axes[0].scatter(
    raw_umap[:, 0],
    raw_umap[:, 1],
    s=point_size,
    c=point_colors,
    alpha=0.8
)

axes[0].set_title(
    "Before integration\nRaw PCA",
    fontsize=14
)

axes[0].set_xticks([])
axes[0].set_yticks([])


axes[1].scatter(
    instinct_umap[:, 0],
    instinct_umap[:, 1],
    s=point_size,
    c=point_colors,
    alpha=0.8
)

axes[1].set_title(
    "After integration\nINSTINCT",
    fontsize=14
)

axes[1].set_xticks([])
axes[1].set_yticks([])


legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=slice_cmap[name],
        markersize=8,
        label=name
    )
    for name in slice_names
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

output_path = os.path.join(
    save_dir,
    "alignment_raw_vs_INSTINCT_run7.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved alignment figure to:")
print(output_path)

# --------------------------------------------------
# Biological annotation plot
# --------------------------------------------------

annotation = adata.obs["Annotation_for_Combined"].astype(str)

unique_annotations = sorted(annotation.unique())

print("\nNumber of biological annotations:", len(unique_annotations))
print(unique_annotations)

cmap = plt.cm.get_cmap(
    "tab20",
    len(unique_annotations)
)

annotation_color_map = {
    anno: cmap(i)
    for i, anno in enumerate(unique_annotations)
}

annotation_colors = [
    annotation_color_map[x]
    for x in annotation
]

plt.figure(figsize=(8, 7))

plt.scatter(
    instinct_umap[:, 0],
    instinct_umap[:, 1],
    s=8,
    c=annotation_colors,
    alpha=0.8
)

plt.title(
    "INSTINCT embedding\nColored by biological annotation",
    fontsize=14
)

plt.xticks([])
plt.yticks([])

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=annotation_color_map[name],
        markersize=7,
        label=name
    )
    for name in unique_annotations
]

plt.legend(
    handles=legend_handles,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
    frameon=False
)

annotation_output = os.path.join(
    save_dir,
    "INSTINCT_run7_biological_annotations.png"
)

plt.savefig(
    annotation_output,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved biological annotation figure to:")
print(annotation_output)



plt.show()

