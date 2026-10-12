from pathlib import Path

import numpy as np
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
# 1. Load common ATAC PCA representation
# ============================================================

X = np.load(PCA_FILE).astype(np.float64)

adata = ad.read_h5ad(
    ADATA_FILE,
    backed="r"
)

slice_labels = (
    adata.obs["slice_name"]
    .astype(str)
    .to_numpy()
)

spatial = np.asarray(
    adata.obsm["spatial"],
    dtype=np.float64
)

assert X.shape == (7113, 100)
assert spatial.shape[0] == 7113

# ============================================================
# 2. Select ALL E13.5 and E15.5 spots
# ============================================================

idx13 = np.where(
    slice_labels == "E13_5-S1"
)[0]

idx15 = np.where(
    slice_labels == "E15_5-S1"
)[0]

F13 = X[idx13]
F15 = X[idx15]

R13 = spatial[idx13]
R15 = spatial[idx15]

print("E13 features:", F13.shape)
print("E15 features:", F15.shape)

print("E13 spatial:", R13.shape)
print("E15 spatial:", R15.shape)

assert F13.shape == (1777, 100)
assert F15.shape == (1949, 100)

assert R13.shape == (1777, 2)
assert R15.shape == (1949, 2)

# ============================================================
# 3. Check PCA vectors before cosine cost
# ============================================================

norm13 = np.linalg.norm(F13, axis=1)
norm15 = np.linalg.norm(F15, axis=1)

print("\nPCA vector norms")
print(
    "E13 min / median / max:",
    norm13.min(),
    np.median(norm13),
    norm13.max()
)

print(
    "E15 min / median / max:",
    norm15.min(),
    np.median(norm15),
    norm15.max()
)

if np.any(norm13 == 0) or np.any(norm15 == 0):
    raise ValueError(
        "Zero PCA vectors found; cosine distance undefined."
    )

# ============================================================
# 4. CROSS-SLICE MOLECULAR COST
#
# M[i,j] = cosine distance between:
#          E13 spot i and E15 spot j
# ============================================================

print("\nBuilding cross-slice molecular cost M...")

M_raw = cdist(
    F13,
    F15,
    metric="cosine"
)

assert M_raw.shape == (1777, 1949)
assert np.isfinite(M_raw).all()

# Numerical safety
M_raw = np.clip(
    M_raw,
    0.0,
    None
)

# ============================================================
# 5. WITHIN-SLICE PHYSICAL SPATIAL GEOMETRY
#
# C13[i,k] = Euclidean spatial distance
# C15[j,l] = Euclidean spatial distance
# ============================================================

print("Building E13 spatial geometry C13...")

C13_raw = cdist(
    R13,
    R13,
    metric="euclidean"
)

print("Building E15 spatial geometry C15...")

C15_raw = cdist(
    R15,
    R15,
    metric="euclidean"
)

assert C13_raw.shape == (1777, 1777)
assert C15_raw.shape == (1949, 1949)

assert np.isfinite(C13_raw).all()
assert np.isfinite(C15_raw).all()

np.fill_diagonal(C13_raw, 0.0)
np.fill_diagonal(C15_raw, 0.0)

# ============================================================
# 6. Diagnostics
# ============================================================

def print_stats(name, A):

    x = A.ravel()

    q = np.quantile(
        x,
        [0.0, 0.01, 0.05, 0.25,
         0.50, 0.75, 0.95, 0.99, 1.0]
    )

    print(f"\n{name}")
    print("shape:", A.shape)
    print("min :", q[0])
    print("1%  :", q[1])
    print("5%  :", q[2])
    print("25% :", q[3])
    print("50% :", q[4])
    print("75% :", q[5])
    print("95% :", q[6])
    print("99% :", q[7])
    print("max :", q[8])


print_stats(
    "Molecular cosine cost M",
    M_raw
)

print_stats(
    "E13 physical spatial distances C13",
    C13_raw
)

print_stats(
    "E15 physical spatial distances C15",
    C15_raw
)

# ============================================================
# 7. Simple scale diagnostics
#
# We are NOT committing to solver scaling yet.
# These versions are only useful for comparison.
# ============================================================

M_max = M_raw / M_raw.max()

C13_max = (
    C13_raw / C13_raw.max()
)

C15_max = (
    C15_raw / C15_raw.max()
)

print("\nAfter simple max normalization:")

print(
    "M median:",
    np.median(M_max)
)

print(
    "C13 median:",
    np.median(C13_max)
)

print(
    "C15 median:",
    np.median(C15_max)
)

# ============================================================
# 8. Check annotations
# ============================================================

annotations = (
    adata.obs["Annotation_for_Combined"]
    .astype(str)
    .to_numpy()
)

ann13 = annotations[idx13]
ann15 = annotations[idx15]

print("\nAnnotation counts E13:")

unique, counts = np.unique(
    ann13,
    return_counts=True
)

for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

print("\nAnnotation counts E15:")

unique, counts = np.unique(
    ann15,
    return_counts=True
)

for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

# ============================================================
# 9. Save prepared pair
#
# We deliberately save features/coordinates/labels,
# NOT several duplicate huge cost matrices.
# Costs are cheap enough to reconstruct exactly.
# ============================================================

np.savez_compressed(
    ROOT / "fugw_E13_E15_input.npz",

    F13=F13,
    F15=F15,

    R13=R13,
    R15=R15,

    idx13=idx13,
    idx15=idx15,

    ann13=ann13,
    ann15=ann15,
)

adata.file.close()

print(
    "\nSaved:",
    ROOT / "fugw_E13_E15_input.npz"
)

print("\nPreparation completed.")