from pathlib import Path
import numpy as np
import ot

from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent

inp = np.load(
    ROOT / "fugw_E13_E15_input.npz",
    allow_pickle=True
)

res = np.load(
    ROOT / "fugw_E13_E15_results.npz",
    allow_pickle=True
)

F13 = inp["F13"].astype(np.float64)
F15 = inp["F15"].astype(np.float64)

R13 = inp["R13"].astype(np.float64)
R15 = inp["R15"].astype(np.float64)

ann13 = inp["ann13"].astype(str)
ann15 = inp["ann15"].astype(str)

Gamma = res["Gamma"]

# ------------------------------------------------------------
# Re-run only to recover Gamma2 and the correct log entries.
#
# Same exact inputs and hyperparameters as experiment 02.
# ------------------------------------------------------------

M_raw = cdist(
    F13,
    F15,
    metric="cosine"
)

M_raw = np.clip(M_raw, 0, None)

C13_raw = cdist(
    R13,
    R13,
    metric="euclidean"
)

C15_raw = cdist(
    R15,
    R15,
    metric="euclidean"
)

molecular_scale = float(
    res["molecular_scale"]
)

spatial_scale = float(
    res["spatial_scale"]
)

M = M_raw / molecular_scale
C13 = C13_raw / spatial_scale
C15 = C15_raw / spatial_scale

n13 = len(F13)
n15 = len(F15)

p = np.ones(n13) / n13
q = np.ones(n15) / n15

alpha = float(res["alpha_pot"])
rho = float(res["rho"])
epsilon = float(res["epsilon"])

print("Recomputing exact saved FUGW configuration...")

G1, G2, log = (
    ot.gromov.fused_unbalanced_gromov_wasserstein(
        C13,
        C15,
        wx=p,
        wy=q,
        reg_marginals=rho,
        epsilon=epsilon,
        divergence="kl",
        unbalanced_solver="sinkhorn_log",
        alpha=alpha,
        M=M,
        max_iter=100,
        tol=1e-7,
        max_iter_ot=500,
        tol_ot=1e-7,
        log=True,
        verbose=False,
    )
)

# ------------------------------------------------------------
# 1. Did we reproduce Gamma?
# ------------------------------------------------------------

print("\n==============================")
print("PLAN CONSISTENCY")
print("==============================")

print(
    "Saved Gamma vs recomputed G1 max diff:",
    np.max(np.abs(Gamma - G1))
)

# P and Q should be inspected because POT's FUGW
# lower-bound solver works with two couplings.

fro_diff = np.linalg.norm(G1 - G2)

relative_fro_diff = (
    fro_diff /
    max(np.linalg.norm(G1), 1e-15)
)

print("||G1-G2||_F:", fro_diff)
print(
    "Relative ||G1-G2||:",
    relative_fro_diff
)

print("G1 total mass:", G1.sum())
print("G2 total mass:", G2.sum())

# ------------------------------------------------------------
# 2. Correct solver logs
# ------------------------------------------------------------

print("\n==============================")
print("SOLVER LOG")
print("==============================")

print(
    "Linear molecular cost:",
    log.get("linear_cost")
)

print(
    "Total FUGW cost:",
    log.get("fugw_cost")
)

errors = np.asarray(
    log.get("error", []),
    dtype=float
)

if len(errors):
    print("BCD iterations:", len(errors))
    print("Final error:", errors[-1])

# ------------------------------------------------------------
# 3. Source and target marginal masses
# ------------------------------------------------------------

row_mass = G1.sum(axis=1)
col_mass = G1.sum(axis=0)

relative_row_mass = row_mass / p
relative_col_mass = col_mass / q

print("\n==============================")
print("SOURCE MASS — E13")
print("==============================")

for label in np.unique(ann13):

    mask = ann13 == label

    x = relative_row_mass[mask]

    print(
        f"{label:32s}"
        f" n={mask.sum():4d}"
        f" median={np.median(x):.4f}"
        f" mean={np.mean(x):.4f}"
        f" min={np.min(x):.4f}"
    )

print("\n==============================")
print("TARGET MASS — E15")
print("==============================")

for label in np.unique(ann15):

    mask = ann15 == label

    x = relative_col_mass[mask]

    print(
        f"{label:32s}"
        f" n={mask.sum():4d}"
        f" median={np.median(x):.4f}"
        f" mean={np.mean(x):.4f}"
        f" min={np.min(x):.4f}"
    )

# ------------------------------------------------------------
# 4. Top-1 correspondence annotation accuracy
#
# Not the final metric, but very interpretable.
# ------------------------------------------------------------

best_target = np.argmax(
    G1,
    axis=1
)

predicted_label = ann15[
    best_target
]

top1_match = np.mean(
    predicted_label == ann13
)

print("\n==============================")
print("TOP-1 CORRESPONDENCE")
print("==============================")

print(
    "Top-1 same annotation fraction:",
    top1_match
)

# ------------------------------------------------------------
# 5. Annotation transport matrix
# ------------------------------------------------------------

labels13 = np.unique(ann13)
labels15 = np.unique(ann15)

print("\n==============================")
print("ANNOTATION TRANSPORT")
print("==============================")

for source_label in labels13:

    source_mask = (
        ann13 == source_label
    )

    source_transport = (
        G1[source_mask, :].sum(axis=0)
    )

    mass = source_transport.sum()

    if mass <= 0:
        continue

    target_masses = []

    for target_label in labels15:

        target_mask = (
            ann15 == target_label
        )

        frac = (
            source_transport[target_mask].sum()
            / mass
        )

        target_masses.append(
            (target_label, frac)
        )

    target_masses.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(
        f"\n{source_label} "
        f"(n={source_mask.sum()})"
    )

    for target_label, frac in target_masses[:5]:

        print(
            f"  -> {target_label:30s}"
            f" {frac:.4f}"
        )

# ------------------------------------------------------------
# 6. Save Gamma2 too
# ------------------------------------------------------------

np.savez_compressed(
    ROOT / "fugw_E13_E15_diagnostics.npz",

    Gamma1=G1,
    Gamma2=G2,

    relative_row_mass=relative_row_mass,
    relative_col_mass=relative_col_mass,

    best_target=best_target,
    predicted_label=predicted_label,

    relative_gamma_difference=relative_fro_diff,
)

print(
    "\nSaved fugw_E13_E15_diagnostics.npz"
)