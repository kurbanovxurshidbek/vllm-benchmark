#!/usr/bin/env python3
"""
Empirical calibration rule for max_num_seqs
============================================

Reproduces the marginal-throughput-gain analysis (g(N)) and the resulting
N* diagnostic value reported in the paper, directly from
data/benchmark_results.csv. No model-architecture information is required:
the rule is computed purely from observed throughput.

    g(N)  = [Thr(2N) - Thr(N)] / Thr(N)
    N*(e) = min{ N in {16, 32, 64, 128, 256} : g(N) < e }

Usage
-----
    python analysis/calibration_rule.py --csv data/benchmark_results.csv --epsilon 0.15
"""

import argparse
import csv
from collections import defaultdict

SEQ_LEVELS = [16, 32, 64, 128, 256]


def load_throughput(csv_path):
    """Return {(model, gpu): {max_num_seqs: throughput}}"""
    table = defaultdict(dict)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["gpu"])
            table[key][int(row["max_num_seqs"])] = float(row["throughput_req_per_sec"])
    return table


def marginal_gains(throughput_by_n):
    """Given {N: throughput}, return list of (N, N*2, gain) for each doubling step."""
    gains = []
    for n in SEQ_LEVELS[:-1]:
        n2 = n * 2
        if n in throughput_by_n and n2 in throughput_by_n:
            thr_n = throughput_by_n[n]
            thr_n2 = throughput_by_n[n2]
            gain = (thr_n2 - thr_n) / thr_n
            gains.append((n, n2, gain))
    return gains


def diagnostic_n_star(gains, epsilon):
    """Smallest N at which the gain to 2N drops below epsilon."""
    for n, n2, gain in gains:
        if gain < epsilon:
            return n
    return gains[-1][1] if gains else None  # never saturates within tested range


def main():
    parser = argparse.ArgumentParser(description="Compute the N* diagnostic rule from benchmark data")
    parser.add_argument("--csv", default="benchmark_results.csv")
    parser.add_argument("--epsilon", type=float, default=0.15,
                         help="Marginal-gain threshold (default: 0.15, i.e. 15%%)")
    args = parser.parse_args()

    table = load_throughput(args.csv)

    print(f"{'Model':<26}{'GPU':<20}{'g(16->32)':<12}{'g(32->64)':<12}{'g(64->128)':<13}{'g(128->256)':<13}{'N*'}")
    print("-" * 110)

    for (model, gpu), throughput_by_n in sorted(table.items()):
        gains = marginal_gains(throughput_by_n)
        n_star = diagnostic_n_star(gains, args.epsilon)
        gain_strs = [f"{g * 100:+.1f}%" for _, _, g in gains]
        gain_strs += [""] * (4 - len(gain_strs))
        print(f"{model:<26}{gpu:<20}{gain_strs[0]:<12}{gain_strs[1]:<12}{gain_strs[2]:<13}{gain_strs[3]:<13}{n_star}")


if __name__ == "__main__":
    main()
