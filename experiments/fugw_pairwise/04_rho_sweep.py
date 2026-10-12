from pathlib import Path
import csv

import numpy as np
import ot

from scipy.spatial.distance import cdist


# ============================================================
# 1. Load inputs and previous configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

inp = np.load(
    ROOT / "fugw_E13_E15_input.npz",
    allow_pickle=True
)

prev = np.load(
    ROOT / "fugw_E13_E15_results.npz",
    allow_pickle=True
)

F13 = inp["F13"].astype(np.float64)
F15 = inp["F15"].astype(np.float64)

R13 = inp["R13"].astype(np.float64)
R15 = inp["R15"].astype(np.float64)

ann13 = inp["ann13"].astype(str)
ann15 = inp["ann15"].astype(str)

n13 = len(F13)
n15 = len(F15)

p = np.ones(n13) / n13
q = np.ones(n15) / n15


# ============================================================
# 2. Reconstruct EXACT same costs
# ============================================================

print("Constructing costs once...")

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

molecular_scale = float(prev["molecular_scale"])
spatial_scale = float(prev["spatial_scale"])

M = M_raw / molecular_scale
C13 = C13_raw / spatial_scale
C15 = C15_raw / spatial_scale

alpha = float(prev["alpha_pot"])
epsilon = float(prev["epsilon"])

print("POT alpha:", alpha)
print("epsilon:", epsilon)


# ============================================================
# 3. ONLY parameter being changed
# ============================================================

RHO_VALUES = [
    1.0,
    0.3,
    0.1,
    0.03,
]


# ============================================================
# 4. Useful annotation sets
# ============================================================

labels13 = set(np.unique(ann13))
labels15 = set(np.unique(ann15))

shared_labels = sorted(
    labels13.intersection(labels15)
)

target_only_labels = sorted(
    labels15 - labels13
)

source_only_labels = sorted(
    labels13 - labels15
)

print("\nShared annotations:")
print(shared_labels)

print("\nE15-only annotations:")
print(target_only_labels)

print("\nE13-only annotations:")
print(source_only_labels)


# ============================================================
# 5. Random annotation baseline
# ============================================================

all_labels = sorted(
    labels13.union(labels15)
)

random_annotation_match = 0.0

for label in all_labels:

    random_annotation_match += (
        np.mean(ann13 == label)
        *
        np.mean(ann15 == label)
    )

print(
    "\nRandom annotation expectation:",
    random_annotation_match
)


# ============================================================
# 6. Helper
# ============================================================

def annotation_mass_summary(
    relative_mass,
    annotations,
    labels
):
    values = {}

    for label in labels:

        mask = annotations == label

        values[label] = {
            "n": int(mask.sum()),
            "median": float(
                np.median(relative_mass[mask])
            ),
            "mean": float(
                np.mean(relative_mass[mask])
            ),
        }

    return values


# ============================================================
# 7. Run rho sweep
# ============================================================

summaries = []

all_row_masses = {}
all_col_masses = {}

for rho in RHO_VALUES:

    print("\n")
    print("=" * 70)
    print(f"RUNNING rho = {rho}")
    print("=" * 70)

    # IMPORTANT:
    # init_pi is deliberately omitted.
    # Every rho starts from POT's same neutral
    # product-mass initialization.

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

    assert np.isfinite(G1).all()
    assert np.isfinite(G2).all()

    # --------------------------------------------------------
    # Solver consistency
    # --------------------------------------------------------

    relative_plan_difference = (
        np.linalg.norm(G1 - G2)
        /
        max(
            np.linalg.norm(G1),
            1e-15
        )
    )

    errors = np.asarray(
        log.get("error", []),
        dtype=float
    )

    final_error = (
        float(errors[-1])
        if len(errors)
        else np.nan
    )

    # --------------------------------------------------------
    # Marginal mass
    # --------------------------------------------------------

    row_mass = G1.sum(axis=1)
    col_mass = G1.sum(axis=0)

    relative_row_mass = row_mass / p
    relative_col_mass = col_mass / q

    total_mass = float(G1.sum())

    all_row_masses[str(rho)] = relative_row_mass
    all_col_masses[str(rho)] = relative_col_mass

    # --------------------------------------------------------
    # Transport entropy
    # --------------------------------------------------------

    valid = row_mass > 0

    conditional = np.zeros_like(G1)

    conditional[valid] = (
        G1[valid]
        /
        row_mass[valid, None]
    )

    entropy = np.zeros(n13)

    entropy[valid] = -np.sum(
        conditional[valid]
        *
        np.log(
            np.clip(
                conditional[valid],
                1e-300,
                None
            )
        ),
        axis=1
    )

    effective_targets = np.exp(
        entropy[valid]
    )

    median_effective_targets = float(
        np.median(effective_targets)
    )

    # --------------------------------------------------------
    # Molecular matching quality
    # --------------------------------------------------------

    transported_molecular_cost = float(
        np.sum(G1 * M_raw)
        /
        total_mass
    )

    # --------------------------------------------------------
    # Annotation matching
    # --------------------------------------------------------

    same_annotation = (
        ann13[:, None]
        ==
        ann15[None, :]
    )

    same_annotation_mass = float(
        G1[same_annotation].sum()
        /
        total_mass
    )

    # --------------------------------------------------------
    # Top-1 annotation
    # --------------------------------------------------------

    best_target = np.argmax(
        G1,
        axis=1
    )

    top1_annotation_match = float(
        np.mean(
            ann13
            ==
            ann15[best_target]
        )
    )

    # --------------------------------------------------------
    # Population mass summaries
    # --------------------------------------------------------

    source_stats = annotation_mass_summary(
        relative_row_mass,
        ann13,
        sorted(labels13)
    )

    target_stats = annotation_mass_summary(
        relative_col_mass,
        ann15,
        sorted(labels15)
    )

    shared_target_medians = [
        target_stats[label]["median"]
        for label in shared_labels
    ]

    target_only_medians = [
        target_stats[label]["median"]
        for label in target_only_labels
    ]

    median_shared_target_mass = float(
        np.median(shared_target_medians)
    )

    median_target_only_mass = float(
        np.median(target_only_medians)
    )

    # Ratio < 1 means target-only populations receive
    # less relative mass than shared populations.

    target_only_vs_shared_ratio = (
        median_target_only_mass
        /
        median_shared_target_mass
    )

    # --------------------------------------------------------
    # Print clean summary
    # --------------------------------------------------------

    print("Total transported mass:", total_mass)

    print(
        "Relative source mass "
        "(min / median / max):",
        relative_row_mass.min(),
        np.median(relative_row_mass),
        relative_row_mass.max()
    )

    print(
        "Relative target mass "
        "(min / median / max):",
        relative_col_mass.min(),
        np.median(relative_col_mass),
        relative_col_mass.max()
    )

    print(
        "Median effective targets:",
        median_effective_targets
    )

    print(
        "Transport molecular cosine cost:",
        transported_molecular_cost
    )

    print(
        "Same-annotation mass:",
        same_annotation_mass
    )

    print(
        "Top-1 same annotation:",
        top1_annotation_match
    )

    print(
        "Median shared-population target mass:",
        median_shared_target_mass
    )

    print(
        "Median target-only population mass:",
        median_target_only_mass
    )

    print(
        "Target-only/shared mass ratio:",
        target_only_vs_shared_ratio
    )

    print(
        "G1/G2 relative difference:",
        relative_plan_difference
    )

    print(
        "Final solver error:",
        final_error
    )

    print("\nE15 TARGET-ONLY POPULATIONS")

    for label in target_only_labels:

        s = target_stats[label]

        print(
            f"{label:25s}"
            f" n={s['n']:4d}"
            f" median={s['median']:.4f}"
            f" mean={s['mean']:.4f}"
        )

    print("\nSELECTED SHARED POPULATIONS")

    selected_shared = [
        "DPallm",
        "Diencephalon_and_hindbrain",
        "Midbrain",
        "Mesenchyme",
        "Muscle",
        "Cartilage_2",
    ]

    for label in selected_shared:

        if label not in target_stats:
            continue

        s = target_stats[label]

        print(
            f"{label:32s}"
            f" n={s['n']:4d}"
            f" median={s['median']:.4f}"
            f" mean={s['mean']:.4f}"
        )

    summaries.append(
        {
            "rho": rho,
            "total_mass": total_mass,

            "source_mass_median":
                float(np.median(relative_row_mass)),

            "target_mass_median":
                float(np.median(relative_col_mass)),

            "median_effective_targets":
                median_effective_targets,

            "molecular_cost":
                transported_molecular_cost,

            "same_annotation_mass":
                same_annotation_mass,

            "top1_annotation_match":
                top1_annotation_match,

            "shared_target_mass":
                median_shared_target_mass,

            "target_only_mass":
                median_target_only_mass,

            "target_only_shared_ratio":
                target_only_vs_shared_ratio,

            "plan_difference":
                relative_plan_difference,

            "final_error":
                final_error,
        }
    )

    # Free the huge transport matrices before next rho.
    del G1
    del G2
    del conditional


# ============================================================
# 8. Final comparison table
# ============================================================

print("\n\n")
print("=" * 105)
print("FINAL RHO COMPARISON")
print("=" * 105)

header = (
    f"{'rho':>6}"
    f"{' mass':>10}"
    f"{' mol.cost':>11}"
    f"{' anno.mass':>11}"
    f"{' top1':>9}"
    f"{' eff.tgt':>10}"
    f"{' shared':>10}"
    f"{' tgt-only':>11}"
    f"{' ratio':>9}"
)

print(header)

for s in summaries:

    print(
        f"{s['rho']:6.2f}"
        f"{s['total_mass']:10.4f}"
        f"{s['molecular_cost']:11.4f}"
        f"{s['same_annotation_mass']:11.4f}"
        f"{s['top1_annotation_match']:9.4f}"
        f"{s['median_effective_targets']:10.2f}"
        f"{s['shared_target_mass']:10.4f}"
        f"{s['target_only_mass']:11.4f}"
        f"{s['target_only_shared_ratio']:9.4f}"
    )


# ============================================================
# 9. Save summary CSV
# ============================================================

csv_path = ROOT / "rho_sweep_summary.csv"

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summaries[0].keys()
    )

    writer.writeheader()
    writer.writerows(summaries)


# ============================================================
# 10. Save marginal masses only
#
# We do NOT save all four gigantic Gamma matrices.
# ============================================================

np.savez_compressed(
    ROOT / "rho_sweep_marginals.npz",

    rho_values=np.array(RHO_VALUES),

    row_mass_rho_1=all_row_masses["1.0"],
    row_mass_rho_03=all_row_masses["0.3"],
    row_mass_rho_01=all_row_masses["0.1"],
    row_mass_rho_003=all_row_masses["0.03"],

    col_mass_rho_1=all_col_masses["1.0"],
    col_mass_rho_03=all_col_masses["0.3"],
    col_mass_rho_01=all_col_masses["0.1"],
    col_mass_rho_003=all_col_masses["0.03"],
)

print("\nSaved:")
print(csv_path)
print(ROOT / "rho_sweep_marginals.npz")

print("\nRHO DIAGNOSIS COMPLETED.")