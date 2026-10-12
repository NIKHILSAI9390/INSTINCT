
from pathlib import Path

import numpy as np
import anndata as ad

# ------------------------------------------
# Paths
# ------------------------------------------

ROOT = Path(__file__).resolve().parent

RESULTS = Path(
    r"C:\Research\INSTINCT_reproduction"
    r"\results\MouseBrain_Jiang2023"
)

# ------------------------------------------
# Load original INSTINCT PCA and metadata
# ------------------------------------------

X = np.load(
    RESULTS / "input_matrix_atac.npy"
)

adata = ad.read_h5ad(
    RESULTS / "preprocessed_concat_atac.h5ad",
    backed="r"
)

slice_labels = (
    adata.obs["slice_name"]
    .astype(str)
    .to_numpy()
)

assert X.shape == (7113, 100)
assert len(slice_labels) == len(X)

print("Original PCA:", X.shape)

# ------------------------------------------
# Reuse our existing two-slice pilot
# ------------------------------------------

pilot = np.load(
    ROOT / "ot_pilot_data.npz"
)

idx_E13 = pilot["idx_A"]
idx_E15 = pilot["idx_B"]

assert np.allclose(
    X[idx_E13], pilot["XA"]
)

assert np.allclose(
    X[idx_E15], pilot["XB"]
)

print("Existing E13/E15 pilot verified.")

# ------------------------------------------
# Sample E11 and E18
# ------------------------------------------

rng = np.random.default_rng(1234)

def select_spots(name, n=200):

    available = np.where(
        slice_labels == name
    )[0]

    assert len(available) >= n

    return np.sort(
        rng.choice(
            available,
            size=n,
            replace=False
        )
    )

idx_E11 = select_spots("E11_0-S1")
idx_E18 = select_spots("E18_5-S1")

# ------------------------------------------
# Collect all four samples
# ------------------------------------------

indices = {
    "E11": idx_E11,
    "E13": idx_E13,
    "E15": idx_E15,
    "E18": idx_E18,
}

features = {
    name: X[idx].astype(np.float64)
    for name, idx in indices.items()
}

expected_slices = {
    "E11": "E11_0-S1",
    "E13": "E13_5-S1",
    "E15": "E15_5-S1",
    "E18": "E18_5-S1",
}

for name, idx in indices.items():

    assert np.all(
        slice_labels[idx] == expected_slices[name]
    )

    assert features[name].shape == (200, 100)
    assert np.isfinite(features[name]).all()

    print(
        f"{name}: {features[name].shape}"
    )

# Preserve the chronological slice order
all_indices = np.concatenate([
    idx_E11,
    idx_E13,
    idx_E15,
    idx_E18
])

X_before = X[all_indices]

# ------------------------------------------
# Save all four OT inputs
# ------------------------------------------

output = ROOT / "ot_four_slices_data.npz"

np.savez_compressed(
    output,

    X_E11=features["E11"],
    X_E13=features["E13"],
    X_E15=features["E15"],
    X_E18=features["E18"],

    idx_E11=idx_E11,
    idx_E13=idx_E13,
    idx_E15=idx_E15,
    idx_E18=idx_E18,

    all_indices=all_indices,
    X_before=X_before,
)

adata.file.close()

print("\nCombined shape:", X_before.shape)
print("Saved:", output)
print("Four-slice preparation completed.")
