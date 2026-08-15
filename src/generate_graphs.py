"""
generate_graphs.py
=====================
Produces the plots referenced by "Proposed Paper Figure Sequence"
(Figures 10 and 11) plus supporting diagnostic plots from the
detection_results.csv / method_comparison.csv produced earlier.

Run AFTER evaluate_pipeline.py and compare_methods.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def main(results_dir="results", figures_dir="figures"):
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(results_dir, "detection_results.csv"))
    cmp_df = pd.read_csv(os.path.join(results_dir, "method_comparison.csv"))

    # Figure: localization error per sample (bar), success/failure colored
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2e7d32" if s else "#c62828" for s in df["success"]]
    ax.bar(df["sample_id"], df["euclidean_error_px"], color=colors)
    ax.set_xlabel("Sample ID")
    ax.set_ylabel("Euclidean localization error (px)")
    ax.set_title("DRIFT-SENSE localization error per test case (green=success, red=failure)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "localization_error_per_sample.png"), dpi=150)
    plt.close(fig)

    # Figure: confidence vs error scatter (sanity check that confidence is meaningful)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["confidence"], df["euclidean_error_px"], c=colors)
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Euclidean error (px)")
    ax.set_title("Confidence vs. localization error")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "confidence_vs_error.png"), dpi=150)
    plt.close(fig)

    # Figure 10: accuracy / error comparison across methods
    summary = cmp_df.groupby("method")["localization_error_px"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="barh", ax=ax, color="#1565c0")
    ax.set_xlabel("Median localization error (px)")
    ax.set_title("Figure 10: Localization error comparison across methods")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "final_method_localization_error.png"), dpi=150)
    plt.close(fig)

    # Figure: matching score comparison (using 1/(1+error) as a simple proxy "score")
    score = 1 / (1 + summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    score.sort_values().plot(kind="barh", ax=ax, color="#6a1b9a")
    ax.set_xlabel("Relative matching quality (higher is better)")
    ax.set_title("Final method matching-quality comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "final_method_matching_score.png"), dpi=150)
    plt.close(fig)

    # Figure 11: runtime vs method (proxy for "runtime vs resolution")
    runtime = cmp_df.groupby("method")["runtime_ms"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    runtime.plot(kind="barh", ax=ax, color="#ef6c00")
    ax.set_xlabel("Mean runtime (ms)")
    ax.set_title("Figure 11: Runtime comparison across methods")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "runtime_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"Figures written to {figures_dir}/")


if __name__ == "__main__":
    main()
