"""
Read results.json and produce (1) a printed comparison table and (2) a bar
chart saved to results/comparison.png comparing the clean vs dark UI.

Usage:  python analyze.py
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")


def print_table(data: dict) -> None:
    agent = data.get("agent", "?")
    print(f"\nAgent: {agent}")
    metrics = [
        ("Success rate", "success_rate", "{:.0%}"),
        ("Failure (tricked)", "failure_rate", "{:.0%}"),
        ("Timeout rate", "timeout_rate", "{:.0%}"),
        ("Avg steps", "avg_steps", "{:.2f}"),
        ("Avg dark-pattern traps hit", "avg_trap_hits", "{:.2f}"),
        ("Avg reward (return)", "avg_reward", "{:+.2f}"),
    ]
    clean = data["by_ui"]["clean"]["summary"]
    dark = data["by_ui"]["dark"]["summary"]
    width = 28
    print(f"{'Metric':<{width}}{'Clean UI':>12}{'Dark UI':>12}")
    print("-" * (width + 24))
    for label, key, fmt in metrics:
        print(f"{label:<{width}}{fmt.format(clean[key]):>12}{fmt.format(dark[key]):>12}")
    print()


def make_chart(data: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart. (pip install matplotlib)")
        return ""

    clean = data["by_ui"]["clean"]["summary"]
    dark = data["by_ui"]["dark"]["summary"]
    labels = ["Success\nrate", "Tricked into\nkeeping", "Avg traps\nhit"]
    clean_vals = [clean["success_rate"], clean["failure_rate"], clean["avg_trap_hits"]]
    dark_vals = [dark["success_rate"], dark["failure_rate"], dark["avg_trap_hits"]]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], clean_vals, w, label="Clean UI", color="#4C9F70")
    ax.bar([i + w / 2 for i in x], dark_vals, w, label="Dark UI", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"UI agent performance: clean vs dark patterns\nagent: {data.get('agent','?')}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (c, d) in enumerate(zip(clean_vals, dark_vals)):
        ax.text(i - w / 2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, d, f"{d:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison.png")
    fig.savefig(out, dpi=130)
    return out


def main() -> None:
    if not os.path.exists(RESULTS_PATH):
        print("No results found. Run:  python run_experiment.py --agent keyword")
        return
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    print_table(data)
    out = make_chart(data)
    if out:
        print(f"Saved chart to {out}")


if __name__ == "__main__":
    main()
