"""
Compare nanoGPT runs by parsing their training logs.

Usage:
    python plot_loss.py logs/baseline_123.out:baseline logs/rmsnorm_456.out:rmsnorm -o compare_norm.png
    (label after ':' is optional; defaults to the log file name)
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

EVAL_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def parse_log(path):
    steps, train, val = [], [], []
    with open(path) as f:
        for m in EVAL_RE.finditer(f.read()):
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    if not steps:
        raise ValueError(f"No eval lines found in {path}")
    return steps, train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="log_path[:label]")
    p.add_argument("-o", "--out", default="compare.png")
    p.add_argument("--title", default="Train / Val loss comparison")
    p.add_argument("--zoom_from", type=int, default=500,
                   help="right panel only shows steps >= this, to make small gaps visible")
    args = p.parse_args()

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    summary = []

    for i, spec in enumerate(args.runs):
        path, _, label = spec.partition(":")
        label = label or os.path.splitext(os.path.basename(path))[0]
        steps, train, val = parse_log(path)
        c = colors[i % len(colors)]

        best_i = min(range(len(val)), key=val.__getitem__)
        summary.append((label, val[best_i], steps[best_i], train[best_i], train[-1], val[-1]))

        # left: full curves (val solid, train dashed, same color per run)
        ax_full.plot(steps, val, "-o", color=c, ms=3, label=f"{label} val")
        ax_full.plot(steps, train, "--", color=c, alpha=0.7, label=f"{label} train")
        ax_full.scatter(steps[best_i], val[best_i], color=c, s=120, marker="*", zorder=5)

        # right: zoomed view of val loss only
        z = [k for k, s in enumerate(steps) if s >= args.zoom_from]
        ax_zoom.plot([steps[k] for k in z], [val[k] for k in z], "-o", color=c, ms=3,
                     label=f"{label} (best {val[best_i]:.4f} @ {steps[best_i]})")
        ax_zoom.scatter(steps[best_i], val[best_i], color=c, s=120, marker="*", zorder=5)

    ax_full.set_title(args.title)
    ax_zoom.set_title(f"Val loss (zoomed, step >= {args.zoom_from}); * = best")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}\n")

    # summary table for the report
    header = f"{'run':<15}{'best val':>10}{'@step':>8}{'train@best':>12}{'final train':>13}{'final val':>11}"
    print(header)
    print("-" * len(header))
    for label, bv, bs, tb, ft, fv in summary:
        print(f"{label:<15}{bv:>10.4f}{bs:>8}{tb:>12.4f}{ft:>13.4f}{fv:>11.4f}")


if __name__ == "__main__":
    main()