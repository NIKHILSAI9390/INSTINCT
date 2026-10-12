
from pathlib import Path

import numpy as np
import ot

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    shortest_path,
    connected_components,
)


# ==================================================
# 1. Load the SAME 800 prepared spots
# ==================================================

ROOT = Path(__file__).resolve().parent

data = np.load(
    ROOT / "ot_four_slices_data.npz"
)

names = ["E11", "E13", "E15", "E18"]

Xs = [
    data["X_E11"].astype(np.float64),
    data["X_E13"].astype(np.float64),
    data["X_E15"].astype(np.float64),
    data["X_E18"].astype(np.float64),
]

for name, X in zip(names, Xs):
    assert X.shape == (200, 100)
    assert np.isfinite(X).all()
    print(f"{name}: {X.shape}")


# ==================================================
# 2. Build molecular kNN graph for EACH slice
#    Same methodology as our original GW pilot
# ==================================================

def build_graph_distance(X, k=20):

    n = X.shape[0]

    D = cdist(
        X,
        X,
        metric="correlation"
    )

    if not np.isfinite(D).all():
        raise ValueError(
            "Invalid correlation distances."
        )

    np.fill_diagonal(D, np.inf)

    neighbors = np.argsort(
        D, axis=1
    )[:, :k]

    G = np.zeros(
        (n, n),
        dtype=np.uint8
    )

    G[
        np.arange(n)[:, None],
        neighbors
    ] = 1

    G = np.maximum(G, G.T)

    components, _ = connected_components(
        csr_matrix(G),
        directed=False
    )

    print("Connected components:", components)

    C = shortest_path(
        csr_matrix(G),
        directed=False,
        unweighted=True
    )

    finite = np.isfinite(C)

    max_finite = C[finite].max()

    if max_finite <= 0:
        raise ValueError("Degenerate graph distances.")

    if not finite.all():
        print("Warning: disconnected graph.")
        C[~finite] = max_finite

    C /= max_finite

    np.fill_diagonal(C, 0)

    return C


Cs = []

for name, X in zip(names, Xs):

    print(f"\nBuilding {name} molecular graph...")

    C = build_graph_distance(X, k=20)

    assert C.shape == (200, 200)
    assert np.isfinite(C).all()

    Cs.append(C)


# ==================================================
# 3. Set up JOINT GW barycenter
# ==================================================

N = 200

# Uniform masses for all four samples
ps = [
    ot.unif(len(X))
    for X in Xs
]

# Barycenter itself also contains 200 atoms
p_star = ot.unif(N)

# Equal importance for ALL FOUR slices
lambdas = [0.25, 0.25, 0.25, 0.25]

EPSILON = 0.05

print("\n" + "=" * 60)
print("STARTING JOINT FOUR-SLICE GW BARYCENTER")
print("No fixed developmental-stage reference.")
print("=" * 60)


# ==================================================
# 4. Joint optimization
# ==================================================

C_star, bary_log = (
    ot.gromov.entropic_gromov_barycenters(
        N=N,
        Cs=Cs,
        ps=ps,
        p=p_star,
        lambdas=lambdas,
        loss_fun="square_loss",
        epsilon=EPSILON,
        symmetric=True,
        max_iter=60,
        tol=1e-5,
        stop_criterion="barycenter",
        warmstartT=False,
        random_state=1234,
        solver="PGD",
        verbose=False,
        log=True,
    )
)

assert C_star.shape == (200, 200)
assert np.isfinite(C_star).all()

errors = np.asarray(
    bary_log.get("err", []),
    dtype=float
)

print("\nJOINT BARYCENTER COMPLETED")
print("C_star shape:", C_star.shape)

print("Barycenter range:",
      C_star.min(), C_star.max())

print("Number of barycenter updates:", len(errors))

if len(errors):
    print("Initial update error:", errors[0])
    print("Final update error:", errors[-1])

    if errors[-1] > 1e-5:
        print(
            "NOTE: Tolerance was not reached. "
            "The result is usable for inspection, "
            "but convergence needs further checking."
        )


# ==================================================
# 5. Obtain final transport plans
#
# We solve against the FINAL C_star, not an earlier
# intermediate barycenter.
#
# Source spots -> shared barycenter atoms
# ==================================================

transport_plans = []
profiles = []

for name, C, p_source in zip(names, Cs, ps):

    print("\n" + "-" * 55)
    print(f"Aligning {name} to JOINT barycenter...")

    T, log = ot.gromov.entropic_gromov_wasserstein(
        C,
        C_star,
        p_source,
        p_star,
        loss_fun="square_loss",
        epsilon=EPSILON,
        max_iter=200,
        tol=1e-7,
        solver="PGD",
        verbose=False,
        log=True,
    )

    assert T.shape == (200, 200)
    assert np.isfinite(T).all()

    row_error = np.max(
        np.abs(T.sum(axis=1) - p_source)
    )

    col_error = np.max(
        np.abs(T.sum(axis=0) - p_star)
    )

    print("Total mass:", T.sum())
    print("Row marginal error:", row_error)
    print("Column marginal error:", col_error)
    print("GW objective:", log.get("gw_dist"))

    if max(row_error, col_error) > 1e-5:
        raise RuntimeError(
            f"Invalid coupling for {name}"
        )

    # Each source spot is represented as a probability
    # distribution over the SAME 200 barycenter atoms.
    P = T / p_source[:, None]

    assert np.allclose(
        P.sum(axis=1),
        1.0,
        atol=1e-5
    )

    entropy = -np.sum(
        P * np.log(np.clip(P, 1e-15, None)),
        axis=1
    )

    effective_targets = np.exp(entropy).mean()

    # How far are profiles from uniform transport?
    nonuniformity = np.linalg.norm(
        P - p_star[None, :],
        axis=1
    ).mean()

    print(
        "Effective barycenter targets:",
        round(effective_targets, 3)
    )

    print(
        "Profile nonuniformity:",
        round(nonuniformity, 8)
    )

    transport_plans.append(T)
    profiles.append(P)


# ==================================================
# 6. Joint representation for all 800 spots
# ==================================================

# Rows from every slice now refer to the same
# 200 learned barycenter atoms.
#
# We use square-rooted coupling probabilities
# as a visualization representation.
#
# Euclidean distances between these vectors are
# proportional to Hellinger distances between
# the underlying probability distributions.

joint_profiles = np.vstack(profiles)

X_after = np.sqrt(
    np.clip(joint_profiles, 0, None)
)

X_before = data["X_before"]

assert joint_profiles.shape == (800, 200)
assert X_after.shape == (800, 200)
assert X_before.shape == (800, 100)

assert np.isfinite(X_after).all()

print("\n" + "=" * 60)
print("FINAL JOINT ALIGNMENT")
print("Before:", X_before.shape)
print("Joint coupling profiles:", joint_profiles.shape)
print("Visualization features:", X_after.shape)

# A basic diagnostic for representation collapse
profile_spread = np.mean(
    np.linalg.norm(
        joint_profiles - joint_profiles.mean(axis=0),
        axis=1
    )
)

print("Joint profile spread:", profile_spread)

if profile_spread < 1e-6:
    print(
        "WARNING: Transport profiles are almost "
        "identical. A UMAP may be uninformative."
    )


# ==================================================
# 7. Save the NEW joint experiment
# ==================================================

output = ROOT / "gw_joint_four_slices_results.npz"

np.savez_compressed(
    output,

    X_before=X_before,
    X_after=X_after,

    C_star=C_star,

    C11=Cs[0],
    C13=Cs[1],
    C15=Cs[2],
    C18=Cs[3],

    T11=transport_plans[0],
    T13=transport_plans[1],
    T15=transport_plans[2],
    T18=transport_plans[3],

    P11=profiles[0],
    P13=profiles[1],
    P15=profiles[2],
    P18=profiles[3],

    joint_profiles=joint_profiles,

    all_indices=data["all_indices"],
    barycenter_errors=errors,
)

print("\nSaved:", output)
print("JOINT FOUR-SLICE GW COMPLETED.")
