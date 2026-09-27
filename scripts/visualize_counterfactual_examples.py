from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("data/worldsense3d-cf-v0.2-full/test")
OUT = Path("docs/figures/counterfactual_examples")
OUT.mkdir(parents=True, exist_ok=True)

CONTEXT_FRAMES = 4


def load_examples():
    best = {}

    for path in ROOT.glob("pair_*.npz"):
        with np.load(path) as d:
            intervention = str(d["intervention_type"])
            shift = float(d["mean_target_shift"])

        if intervention not in best or shift > best[intervention][0]:
            best[intervention] = (shift, path)

    return best


def trajectory_plot(path, intervention, shift):
    with np.load(path) as d:
        factual = d["factual_states"].astype(np.float32)
        counterfactual = d["counterfactual_states"].astype(np.float32)
        target = int(d["target_index"])
        num_objects = int(d["num_objects"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    for i in range(num_objects):
        f = factual[:, i, :2]
        c = counterfactual[:, i, :2]

        if i == target:
            ax.plot(
                f[:, 0],
                f[:, 1],
                marker="o",
                linewidth=2.5,
                label="Target factual",
            )
            ax.plot(
                c[:, 0],
                c[:, 1],
                marker="o",
                linestyle="--",
                linewidth=2.5,
                label="Target counterfactual",
            )

            start = factual[CONTEXT_FRAMES - 1, i, :2]
            ax.scatter(
                start[0],
                start[1],
                s=120,
                marker="*",
                label="Intervention point",
            )
        else:
            ax.plot(
                f[:, 0],
                f[:, 1],
                linewidth=1.0,
                alpha=0.35,
            )

    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_title(
        f"{intervention.replace('_', ' ').title()} Counterfactual\n"
        f"Mean target trajectory shift = {shift:.3f} m"
    )
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUT / f"{intervention}_trajectory.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def add_label(img, text):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, img.width, 26],
        fill=(0, 0, 0),
    )
    draw.text(
        (8, 6),
        text,
        fill=(255, 255, 255),
    )

    return img


def make_gif(path, intervention):
    with np.load(path) as d:
        factual = d["factual_rgb"]
        counterfactual = d["counterfactual_rgb"]

    frames = []

    for t in range(factual.shape[0]):
        f = add_label(
            Image.fromarray(factual[t]),
            f"FACTUAL  |  t={t}",
        )

        c = add_label(
            Image.fromarray(counterfactual[t]),
            f"COUNTERFACTUAL  |  t={t}",
        )

        canvas = Image.new(
            "RGB",
            (f.width * 2, f.height),
        )

        canvas.paste(f, (0, 0))
        canvas.paste(c, (f.width, 0))

        frames.append(canvas.resize((768, 384)))

    frames[0].save(
        OUT / f"{intervention}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        optimize=True,
    )


def comparison_panel(examples):
    interventions = [
        "force",
        "mass",
        "friction",
        "cancel_action",
    ]

    fig = plt.figure(figsize=(12, 10))

    for j, intervention in enumerate(interventions, start=1):
        shift, path = examples[intervention]

        with np.load(path) as d:
            factual = d["factual_states"]
            counterfactual = d["counterfactual_states"]
            target = int(d["target_index"])

        f = factual[:, target, :2]
        c = counterfactual[:, target, :2]

        ax = fig.add_subplot(2, 2, j)

        ax.plot(
            f[:, 0],
            f[:, 1],
            marker="o",
            label="Factual",
        )

        ax.plot(
            c[:, 0],
            c[:, 1],
            marker="o",
            linestyle="--",
            label="Counterfactual",
        )

        start = factual[CONTEXT_FRAMES - 1, target, :2]
        ax.scatter(
            start[0],
            start[1],
            marker="*",
            s=100,
        )

        ax.set_title(
            f"{intervention.replace('_', ' ').title()}\n"
            f"Shift={shift:.3f} m"
        )

        ax.axis("equal")
        ax.grid(alpha=0.25)

        if j == 1:
            ax.legend()

    fig.suptitle(
        "WorldSense-3D: Factual vs Counterfactual Physical Futures",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "counterfactual_overview.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    examples = load_examples()

    print("Selected qualitative examples:")

    for intervention, (shift, path) in examples.items():
        print(
            f"{intervention:14s} "
            f"shift={shift:.4f} "
            f"{path.name}"
        )

        trajectory_plot(
            path,
            intervention,
            shift,
        )

        make_gif(
            path,
            intervention,
        )

    comparison_panel(examples)

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
