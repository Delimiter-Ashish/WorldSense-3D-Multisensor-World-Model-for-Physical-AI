import json
from pathlib import Path

import matplotlib.pyplot as plt

results = json.loads(
    Path("results/v0.2_ablation_results.json").read_text()
)

order = [
    ("state_intervention", "State + Intervention"),
    ("global_visual", "Global Visual"),
    ("target_aware_multimodal", "Target-Aware Multimodal"),
    ("no_intervention", "No Intervention"),
]

labels = [label for _, label in order]

target_ade = [
    results[k]["overall"]["target_ade"]
    for k, _ in order
]

shift_mae = [
    results[k]["overall"]["shift_mae"]
    for k, _ in order
]

plt.figure(figsize=(9, 5))
plt.bar(labels, target_ade)
plt.ylabel("Target ADE ↓")
plt.title("Counterfactual Target-Trajectory Prediction")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    "docs/figures/v02_target_ade.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(labels, shift_mae)
plt.ylabel("Counterfactual Shift MAE ↓")
plt.title("Intervention-Effect Prediction")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    "docs/figures/v02_shift_mae.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

best = results["state_intervention"]["by_intervention"]

interventions = [
    "force",
    "mass",
    "friction",
    "cancel_action",
]

values = [
    best[k]["target_ade"]
    for k in interventions
]

display_names = [
    "Force",
    "Mass",
    "Friction",
    "Cancel Action",
]

plt.figure(figsize=(8, 5))
plt.bar(display_names, values)
plt.ylabel("Target ADE ↓")
plt.title("Difficulty by Counterfactual Intervention")
plt.tight_layout()
plt.savefig(
    "docs/figures/v02_intervention_breakdown.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

rows = []
for key, label in order:
    m = results[key]["overall"]
    rows.append(
        (
            label,
            m["ade"],
            m["fde"],
            m["target_ade"],
            m["target_fde"],
            m["shift_mae"],
            m["final_shift_mae"],
        )
    )

best_target = results["state_intervention"]["overall"]["target_ade"]
no_int_target = results["no_intervention"]["overall"]["target_ade"]

target_gain = (
    (no_int_target - best_target)
    / no_int_target
    * 100
)

best_shift = results["state_intervention"]["overall"]["shift_mae"]
no_int_shift = results["no_intervention"]["overall"]["shift_mae"]

shift_gain = (
    (no_int_shift - best_shift)
    / no_int_shift
    * 100
)

lines = [
    "# WorldSense-3D v0.2 Counterfactual Benchmark",
    "",
    "## Experimental Setup",
    "",
    "- 5,000 paired factual/counterfactual episodes",
    "- 4,000 train / 500 validation / 500 test",
    "- Counterfactual interventions: force, mass, friction, action cancellation",
    "- Future 3D object trajectory forecasting",
    "- Evaluation includes scene trajectory error and intervention-effect error",
    "",
    "## Ablation Results",
    "",
    "| Model | ADE ↓ | FDE ↓ | Target ADE ↓ | Target FDE ↓ | Shift MAE ↓ | Final Shift MAE ↓ |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    lines.append(
        f"| {row[0]} | "
        f"{row[1]:.4f} | "
        f"{row[2]:.4f} | "
        f"{row[3]:.4f} | "
        f"{row[4]:.4f} | "
        f"{row[5]:.4f} | "
        f"{row[6]:.4f} |"
    )

lines += [
    "",
    "## Main Findings",
    "",
    f"- Explicit intervention conditioning reduces target ADE by approximately {target_gain:.1f}% relative to the no-intervention model.",
    f"- Intervention conditioning reduces counterfactual shift MAE by approximately {shift_gain:.1f}%.",
    "- Structured state + intervention conditioning achieves the strongest overall result in the current simulator.",
    "- RGB/depth features do not improve over privileged structured physical state under the current synthetic setup.",
    "- Mass interventions are the most difficult among the tested intervention families.",
    "- Action cancellation is the easiest intervention to predict.",
    "",
    "## Interpretation",
    "",
    "These results suggest that explicit intervention information is essential for counterfactual physical forecasting. When accurate structured physical state is already available, visual observations can be redundant. This motivates future experiments where physical state is partial, noisy, or unavailable and must instead be inferred from perception.",
    "",
    "## Figures",
    "",
    "![Target ADE](figures/v02_target_ade.png)",
    "",
    "![Shift MAE](figures/v02_shift_mae.png)",
    "",
    "![Intervention Breakdown](figures/v02_intervention_breakdown.png)",
    "",
]

Path("docs/V0.2_RESULTS.md").write_text(
    "\n".join(lines) + "\n"
)

print("Created:")
print(" docs/V0.2_RESULTS.md")
print(" docs/figures/v02_target_ade.png")
print(" docs/figures/v02_shift_mae.png")
print(" docs/figures/v02_intervention_breakdown.png")
