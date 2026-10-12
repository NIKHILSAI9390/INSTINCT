from pathlib import Path
import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (shortest_path,connected_components,)

#load data from prev file

ROOT=Path(__file__).resolve().parent
data=np.load(ROOT / "ot_pilot_data.npz")

XA=data["XA"]
XB=data["XB"]

print("E13.5:", XA.shape)
print("E15.5:", XB.shape)


# graph distance construction

def build_graph_distance(X, k=20):
    n = X.shape[0]
    # Compute pairwise distances
    D = cdist(X, X, metric='correlation')
    if not np.isfinite(D).all():
        raise ValueError("Non-finite correlation in distance matrix.check for constant or invalid pca rows")
    
    #do not slect cell as it own neighbour
    
    np.fill_diagonal(D, np.inf)
    
    #select k nn for every spot 
    neighbors = np.argsort(D, axis=1)[:, :k]
     # Binary adjacency matrix
    G = np.zeros((n, n), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    G[rows, neighbors] = 1
    
    # Symmetrize the graph
    G = np.maximum(G, G.T)
    
    #check connectivity
    n_components, _ = connected_components(
        csr_matrix(G),
        directed=False
    )
    
    print("Connected components:", n_components)
   
    # Shortest-path distances 
    C = shortest_path(
        csr_matrix(G),
        directed=False,
        unweighted=True
    )
    # for disconnected or far-apart nodes, 
    finite = np.isfinite(C)
    max_finite = C[finite].max()
    
    if max_finite <= 0:
        raise ValueError("Degenerate graph distances.")
    
    if not finite.all():
        print("Warning: disconnected pairs found.")
        C[~finite] = max_finite
    #normalisation     
    C = C / max_finite   
    
    np.fill_diagonal(C, 0)

    return C, G
print("\nBuilding graph for E13.5...")
CA, GA = build_graph_distance(XA, k=20)

print("\nBuilding graph for E15.5...")
CB, GB = build_graph_distance(XB, k=20)

print("\nGraph distance matrices:")
print("CA:", CA.shape)
print("CB:", CB.shape)

# uniform probablity distributions

p = ot.unif(XA.shape[0])
q = ot.unif(XB.shape[0])

#entropic-gw ot

print("\nStarting GW alignment...")

T, gw_log = ot.gromov.entropic_gromov_wasserstein(CA,CB,p,q,loss_fun='square_loss',epsilon=0.05,max_iter=200,tol=1e-7,verbose=False,solver="PGD",log=True)

if not np.isfinite(T).all():
    raise ValueError(
        "GW returned non-finite transport values."
    )   

print("\nGW optimization finished.")

print("Transport shape:", T.shape)
print("Transport total mass:", T.sum())

row_error = np.max(np.abs(T.sum(axis=1) - p))
col_error = np.max(np.abs(T.sum(axis=0) - q))

print("Maximum row marginal error:", row_error)
print("Maximum column marginal error:", col_error)

if "gw_dist" in gw_log:
    print("GW objective:", gw_log["gw_dist"])   
    
# barycentric projection

XA_on_B=(T @ XB) / p[:, None]

print("\nProjected E13.5 shape:", XA_on_B.shape)


#save results 

np.savez_compressed(ROOT /"gw_pilot_results.npz",XA_on_B=XA_on_B, T=T, CA=CA, CB=CB, GA=GA, GB=GB, XA=XA, XB=XB)
print("Saved: gw_pilot_results.npz")

#visualizatio of transport matrix

conditional_T = T / p[:, None]

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    conditional_T,
    aspect="auto",
    interpolation="nearest"
)

ax.set_title("GW transport plan: E13.5 to E15.5")
ax.set_xlabel("Target spots (E15.5)")
ax.set_ylabel("Source spots (E13.5)")

fig.colorbar(
    im,
    ax=ax,
    label="Conditional transport probability"
)

fig.tight_layout()

fig.savefig(
    ROOT / "gw_pilot_coupling.png",
    dpi=300
)

plt.close(fig)

print("Saved: gw_pilot_coupling.png")
print("\nStep 2 completed.")
