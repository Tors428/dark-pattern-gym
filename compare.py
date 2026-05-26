"""
Compare two agents on the SAME UI, side by side.

Typical use: compare the keyword baseline against the Gemini agent on the
dark UI (the question that matters -- does the smart agent get fooled more?).

    python compare.py --baseline results/results.json \
                      --llm results/gemini_dark.json --ui dark

Each file is produced by run_experiment.py. The script reads each file's
summary for the chosen UI and prints a side-by-side table, then saves
results/compare.png.
"""

import argparse
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_summary(path: str, ui: str):
    with open(path) as f:
        data = json.load(f)
    if ui not in data["by_ui"]:
        raise SystemExit(f"File {path} has no '{ui}' UI data (has: {list(data['by_ui'])})")
    return data.get("agent", os.path.basename(path)), data["by_ui"][ui]["summary"]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two agents on one UI")
    p.add_argument("--baseline", default=os.path.join(RESULTS_DIR, "results.json"))
    p.add_argument("--llm", default=os.path.join(RESULTS_DIR, "gemini_dark.json"))
    p.add_argument("--ui", default="dark", choices=["clean", "dark"])
    args = p.parse_args()

    name_a, a = load_summary(args.baseline, args.ui)
    name_b, b = load_summary(args.llm, args.ui)

    rows = [
        ("Success rate", "success_rate", "{:.0%}"),
        ("Tricked into keeping", "failure_rate", "{:.0%}"),
        ("Timeout rate", "timeout_rate", "{:.0%}"),
        ("Avg steps", "avg_steps", "{:.2f}"),
        ("Avg dark-pattern traps hit", "avg_trap_hits", "{:.2f}"),
        ("Avg reward (return)", "avg_reward", "{:+.2f}"),
        ("Episodes", "episodes", "{}"),
    ]
    w = 28
    print(f"\nComparison on the '{args.ui}' UI")
    print(f"{'Metric':<{w}}{name_a:>20}{name_b:>20}")
    print("-" * (w + 40))
    for label, key, fmt in rows:
        print(f"{label:<{w}}{fmt.format(a[key]):>20}{fmt.format(b[key]):>20}")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return

    labels = ["Success\nrate", "Tricked into\nkeeping", "Avg traps\nhit"]
    a_vals = [a["success_rate"], a["failure_rate"], a["avg_trap_hits"]]
    b_vals = [b["success_rate"], b["failure_rate"], b["avg_trap_hits"]]
    x = range(len(labels))
    wbar = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - wbar / 2 for i in x], a_vals, wbar, label=name_a, color="#5577AA")
    ax.bar([i + wbar / 2 for i in x], b_vals, wbar, label=name_b, color="#C44E52")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_title(f"Dumb vs smart agent on the dark UI ('{args.ui}')")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (va, vb) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - wbar / 2, va, f"{va:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + wbar / 2, vb, f"{vb:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare.png")
    fig.savefig(out, dpi=130)
    print(f"Saved chart to {out}")


if __name__ == "__main__":
    main()
