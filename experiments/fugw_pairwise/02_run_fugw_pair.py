from pathlib import Path

import numpy as np
import ot

from scipy.spatial.distance import cdist


# ============================================================
# 1. Load prepared full slices
# ============================================================

ROOT = Path(__file__).resolve().parent

data = np.load(
    ROOT / "fugw_E13_E15_input.npz",
    allow_pickle=True
)

F13 = data["F13"].astype(np.float64)
F15 = data["F15"].astype(np.float64)

R13 = data["R13"].astype(np.float64)
R15 = data["R15"].astype(np.float64)

ann13 = data["ann13"].astype(str)
ann15 = data["ann15"].astype(str)

idx13 = data["idx13"]
idx15 = data["idx15"]

n13 = len(F13)
n15 = len(F15)

print("POT version:", ot.__version__)
print("E13:", F13.shape)
print("E15:", F15.shape)

if not hasattr(
    ot.gromov,
    "fused_unbalanced_gromov_wasserstein"
):
    raise RuntimeError(
        "Installed POT version does not contain "
        "fused_unbalanced_gromov_wasserstein."
    )


# ============================================================
# 2. Molecular cross-slice cost
# ============================================================

print("\nConstructing molecular cosine cost...")

M_raw = cdist(
    F13,
    F15,
    metric="cosine"
)

M_raw = np.clip(
    M_raw,
    0.0,
    None
)


# ============================================================
# 3. Physical within-slice geometry
# ============================================================

print("Constructing physical geometries...")

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

np.fill_diagonal(C13_raw, 0.0)
np.fill_diagonal(C15_raw, 0.0)


# ============================================================
# 4. Robust normalization
#
# Molecular and spatial quantities use different units.
#
# Molecular:
# divide by its 95th percentile.
#
# Spatial:
# use ONE COMMON scale for both slices.
#
# This preserves E13-vs-E15 relative tissue scale rather than
# normalizing each slice independently.
# ============================================================

molecular_scale = np.quantile(
    M_raw,
    0.95
)

spatial_scale_13 = np.quantile(
    C13_raw,
    0.95
)

spatial_scale_15 = np.quantile(
    C15_raw,
    0.95
)

spatial_scale = (
    spatial_scale_13 +
    spatial_scale_15
) / 2.0

M = M_raw / molecular_scale

C13 = C13_raw / spatial_scale
C15 = C15_raw / spatial_scale

print("\nNORMALIZATION")
print("Molecular scale:", molecular_scale)
print("Common spatial scale:", spatial_scale)

print(
    "Normalized M median:",
    np.median(M)
)

print(
    "Normalized C13 median:",
    np.median(C13)
)

print(
    "Normalized C15 median:",
    np.median(C15)
)


# ============================================================
# 5. Uniform prior masses
#
# These are prior masses.
# FUGW is allowed to deviate from them.
# ============================================================

p = np.ones(n13) / n13
q = np.ones(n15) / n15


# ============================================================
# 6. Determine molecular-vs-geometry numerical scale
#
# Start from independent diffuse coupling:
#
# P0 = p q^T
#
# Molecular contribution:
#
#     <M, P0>
#
# GW contribution:
#
# E[(C13 - C15)^2]
#
# With independent uniform pairs this can be calculated
# without constructing a 4-D tensor.
# ============================================================

molecular_initial = np.sum(
    p[:, None] *
    q[None, :] *
    M
)

mean_C13 = np.sum(
    p[:, None] *
    p[None, :] *
    C13
)

mean_C15 = np.sum(
    q[:, None] *
    q[None, :] *
    C15
)

mean_C13_sq = np.sum(
    p[:, None] *
    p[None, :] *
    C13**2
)

mean_C15_sq = np.sum(
    q[:, None] *
    q[None, :] *
    C15**2
)

gw_initial = (
    mean_C13_sq
    + mean_C15_sq
    - 2.0 * mean_C13 * mean_C15
)

print("\nINITIAL DIFFUSE-PLAN COST SCALES")
print("Molecular term:", molecular_initial)
print("GW structural term:", gw_initial)


# ============================================================
# 7. POT alpha
#
# POT objective is approximately:
#
#       GW + alpha * molecular
#
# So choose alpha such that the two terms have equal
# numerical contribution at the diffuse initialization.
#
# This is NOT a tuned biological optimum.
# It is only a principled initial scale calibration.
# ============================================================

alpha_pot = (
    gw_initial /
    molecular_initial
)

print("\nScale-balanced POT alpha:", alpha_pot)


# ============================================================
# 8. First FUGW configuration
#
# rho = 1:
# literature-supported genuinely unbalanced starting point.
#
# epsilon = 1e-3:
# small entropy regularization used in original FUGW work.
#
# We are NOT sweeping parameters yet.
# ============================================================

RHO = 1.0
EPSILON = 1e-3

print("\n" + "=" * 65)
print("RUNNING FULL E13 -> E15 FUGW")
print("=" * 65)

print("Source spots:", n13)
print("Target spots:", n15)

print("POT alpha:", alpha_pot)
print("rho:", RHO)
print("epsilon:", EPSILON)


# ============================================================
# 9. Solve
# ============================================================

Gamma, Gamma2, log = (
    ot.gromov.fused_unbalanced_gromov_wasserstein(
        C13,
        C15,

        wx=p,
        wy=q,

        reg_marginals=RHO,

        epsilon=EPSILON,

        divergence="kl",

        unbalanced_solver="sinkhorn_log",

        alpha=alpha_pot,

        M=M,

        max_iter=100,
        tol=1e-7,

        max_iter_ot=500,
        tol_ot=1e-7,

        log=True,
        verbose=False,
    )
)

assert Gamma.shape == (n13, n15)
assert np.isfinite(Gamma).all()
assert np.all(Gamma >= 0)


# ============================================================
# 10. Basic transport diagnostics
# ============================================================

row_mass = Gamma.sum(axis=1)
col_mass = Gamma.sum(axis=0)

total_mass = Gamma.sum()

relative_row_mass = (
    row_mass / p
)

relative_col_mass = (
    col_mass / q
)

print("\n" + "=" * 65)
print("TRANSPORT DIAGNOSTICS")
print("=" * 65)

print("Gamma shape:", Gamma.shape)
print("Total transported mass:", total_mass)

print(
    "\nE13 relative transported mass:"
)

print(
    "min / median / max:",
    relative_row_mass.min(),
    np.median(relative_row_mass),
    relative_row_mass.max()
)

print(
    "\nE15 relative received mass:"
)

print(
    "min / median / max:",
    relative_col_mass.min(),
    np.median(relative_col_mass),
    relative_col_mass.max()
)


# ============================================================
# 11. Transport diffuseness
# ============================================================

valid = row_mass > 0

P_cond = np.zeros_like(Gamma)

P_cond[valid] = (
    Gamma[valid] /
    row_mass[valid, None]
)

entropy = np.zeros(n13)

entropy[valid] = -np.sum(
    P_cond[valid] *
    np.log(
        np.clip(
            P_cond[valid],
            1e-300,
            None
        )
    ),
    axis=1
)

normalized_entropy = (
    entropy / np.log(n15)
)

effective_targets = np.exp(entropy)

print(
    "\nNormalized transport entropy "
    "min / median / max:"
)

print(
    normalized_entropy[valid].min(),
    np.median(normalized_entropy[valid]),
    normalized_entropy[valid].max()
)

print(
    "Effective target spots "
    "min / median / max:"
)

print(
    effective_targets[valid].min(),
    np.median(effective_targets[valid]),
    effective_targets[valid].max()
)


# ============================================================
# 12. Molecular quality
#
# Compare transported pairs against completely independent
# cross-slice matching.
# ============================================================

weighted_molecular_cost = (
    (Gamma * M_raw).sum()
    / total_mass
)

random_molecular_cost = (
    M_raw.mean()
)

print("\nMOLECULAR MATCHING")

print(
    "Transport-weighted cosine distance:",
    weighted_molecular_cost
)

print(
    "Independent/random cosine distance:",
    random_molecular_cost
)


# ============================================================
# 13. Annotation agreement
#
# Annotation is ONLY evaluation.
# It was never given to FUGW.
# ============================================================

same_label = (
    ann13[:, None]
    ==
    ann15[None, :]
)

same_annotation_mass = (
    Gamma[same_label].sum()
    / total_mass
)

# Random expectation from empirical label proportions

labels = np.union1d(
    np.unique(ann13),
    np.unique(ann15)
)

random_annotation_match = 0.0

for label in labels:

    f13 = np.mean(
        ann13 == label
    )

    f15 = np.mean(
        ann15 == label
    )

    random_annotation_match += (
        f13 * f15
    )

print("\nANNOTATION AGREEMENT")

print(
    "FUGW same-annotation transported mass:",
    same_annotation_mass
)

print(
    "Random independent expectation:",
    random_annotation_match
)


# ============================================================
# 14. Per-annotation transported mass
#
# This is particularly important for rare populations.
# ============================================================

print("\nE13 RELATIVE MASS BY ANNOTATION")

for label in np.unique(ann13):

    mask = (
        ann13 == label
    )

    values = relative_row_mass[mask]

    print(
        f"{label:32s}"
        f" n={mask.sum():4d}"
        f" median={np.median(values):.4f}"
        f" mean={np.mean(values):.4f}"
    )


# ============================================================
# 15. Solver information
# ============================================================

errors = np.asarray(
    log.get("error", []),
    dtype=float
)

print("\nSOLVER")

if len(errors):

    print(
        "BCD iterations:",
        len(errors)
    )

    print(
        "Initial coupling change:",
        errors[0]
    )

    print(
        "Final coupling change:",
        errors[-1]
    )

print(
    "Linear FUGW component:",
    log.get("linear")
)

print(
    "Total FUGW cost:",
    log.get("fugw_cost")
)


# ============================================================
# 16. Save
# ============================================================

np.savez_compressed(
    ROOT / "fugw_E13_E15_results.npz",

    Gamma=Gamma,

    row_mass=row_mass,
    col_mass=col_mass,

    relative_row_mass=relative_row_mass,
    relative_col_mass=relative_col_mass,

    entropy=entropy,
    normalized_entropy=normalized_entropy,

    M=M,
    C13=C13,
    C15=C15,

    molecular_scale=molecular_scale,
    spatial_scale=spatial_scale,

    alpha_pot=alpha_pot,
    rho=RHO,
    epsilon=EPSILON,

    idx13=idx13,
    idx15=idx15,

    ann13=ann13,
    ann15=ann15,
)

print(
    "\nSaved:",
    ROOT / "fugw_E13_E15_results.npz"
)

print("\nFUGW completed.")