import os
import csv
import torch
import numpy as np
import anndata as ad

from INSTINCT import *

import warnings
warnings.filterwarnings("ignore")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


slice_name_list = [
    "E11_0-S1",
    "E13_5-S1",
    "E15_5-S1",
    "E18_5-S1"
]


save_dir = (
    r"C:\Research\INSTINCT_reproduction"
    r"\results\MouseBrain_Jiang2023"
)

embed_dir = os.path.join(
    save_dir,
    "comparison",
    "INSTINCT"
)

os.makedirs(embed_dir, exist_ok=True)


# --------------------------------------------------
# Load already-preprocessed data
# --------------------------------------------------

cas_list = [
    ad.read_h5ad(
        os.path.join(
            save_dir,
            f"filtered_merged_{sample}_atac.h5ad"
        )
    )
    for sample in slice_name_list
]

adata_concat = ad.read_h5ad(
    os.path.join(
        save_dir,
        "preprocessed_concat_atac.h5ad"
    )
)


# --------------------------------------------------
# Load the existing 100-D PCA matrix
# --------------------------------------------------

input_matrix = np.load(
    os.path.join(
        save_dir,
        "input_matrix_atac.npy"
    )
)

adata_concat.obsm["X_pca"] = input_matrix


# --------------------------------------------------
# Rebuild spatial neighbor graph
# --------------------------------------------------

print("Creating neighbor graph...")

create_neighbor_graph(
    cas_list,
    adata_concat
)

print("Neighbor graph ready.")


# --------------------------------------------------
# Run the original 8 seeds
# --------------------------------------------------

num_iters = 8

for j in range(num_iters):

    output_file = os.path.join(
        embed_dir,
        f"INSTINCT_embed_{j}.csv"
    )

    print("\n" + "=" * 60)
    print(f"INSTINCT iteration {j}")
    print(f"Seed = {1234 + j}")

    model = INSTINCT_Model(
        cas_list,
        adata_concat,
        seed=1234 + j,
        device=device
    )

    model.train(
        report_loss=False,
        report_interval=100
    )

    model.eval(cas_list)

    result = ad.concat(
        cas_list,
        label="slice_name",
        keys=slice_name_list
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerows(
            result.obsm["INSTINCT_latent"]
        )

    print("Saved:", output_file)


print("\nAll 8 INSTINCT runs completed.")