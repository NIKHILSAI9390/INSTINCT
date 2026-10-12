from pathlib import Path 
import numpy as np
import anndata as ad

#existing instinct reproduction
BASE=Path(r"C:\Research\INSTINCT_reproduction")
DATA=BASE/ "results"/"MouseBrain_Jiang2023"
X=np.load(DATA/"input_matrix_atac.npy")

#read metadata without loading the huge peak matrix 
adata=ad.read_h5ad(DATA/"preprocessed_concat_atac.h5ad",backed="r")

labels=adata.obs["slice_name"].astype(str).to_numpy()

print ("Full PCA:", X.shape)
print("Metadata:", adata.shape)

assert X.shape[0] == len(labels)
assert np.isfinite(X).all()

print("\nAvailable slices:")
for name in np.unique(labels):
    print(name, ":", np.sum(labels == name))
    
# 2. Select two slices for the initial OT experiment
slice_A = "E13_5-S1"
slice_B = "E15_5-S1"

idx_A = np.where(labels == slice_A)[0]
idx_B = np.where(labels == slice_B)[0]

# 3. Reproducible random sampling
rng = np.random.default_rng(1234)

n = 200

selected_A = np.sort(
    rng.choice(idx_A, size=n, replace=False)
)

selected_B = np.sort(
    rng.choice(idx_B, size=n, replace=False)
)

XA = X[selected_A].astype(np.float64)
XB = X[selected_B].astype(np.float64)

print("\nOT pilot inputs:")
print("Slice A:", slice_A, XA.shape)
print("Slice B:", slice_B, XB.shape)

# 4. Save the pilot data and original row indices
np.savez_compressed(
    "ot_pilot_data.npz",
    XA=XA,
    XB=XB,
    idx_A=selected_A,
    idx_B=selected_B,
)

adata.file.close()

print("\nSaved: ot_pilot_data.npz")
print("Step 1 completed.")