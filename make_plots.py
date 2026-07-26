#!/usr/bin/env python3
"""
Generates the four summary plots referenced in the README (plots/*.png)
directly from benchmark_results.csv. Re-run this after regenerating the
dataset to refresh the plots.

Usage:
    python make_plots.py
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")

MODELS = [
    "Qwen2.5-1.5B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Gemma-2-2B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]
GPUS = ["NVIDIA-L4-24GB", "NVIDIA-RTX-PRO-96GB"]
COLORS = {"NVIDIA-L4-24GB": "#1f77b4", "NVIDIA-RTX-PRO-96GB": "#d62728"}


def plot_metric(column, ylabel, title, filename, logy=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        for gpu in GPUS:
            gsub = sub[sub["gpu"] == gpu].sort_values("max_num_seqs")
            ax.plot(
                gsub["max_num_seqs"], gsub[column],
                marker="o", label=gpu, color=COLORS[gpu],
            )
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("max_num_seqs")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"plots/{filename}", dpi=150)
    plt.close(fig)
    print(f"Wrote plots/{filename}")


if __name__ == "__main__":
    plot_metric("throughput_req_per_sec", "Throughput (req/s)",
                "Throughput vs. max_num_seqs", "throughput.png")
    plot_metric("mean_latency_sec", "Mean latency (s)",
                "Mean Latency vs. max_num_seqs", "latency.png", logy=True)
    plot_metric("p95_latency_sec", "P95 latency (s)",
                "P95 Latency vs. max_num_seqs", "p95_latency.png", logy=True)
    plot_metric("p99_latency_sec", "P99 latency (s)",
                "P99 Latency vs. max_num_seqs", "p99_latency.png", logy=True)
