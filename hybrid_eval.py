"""
Computes the lambda sweep results for the stochastic routing hybrid,
following the objective defined in the dissertation methodology:

    L(lambda) = lambda * L1 + (1 - lambda) * L2      (expected loss)
    C(lambda) = lambda * C1 + (1 - lambda) * C2      (expected cost)
    J(lambda) = L(lambda) + beta * C(lambda)

where L1, L2 are the measured validation losses of the Transformer and
Mamba baselines (from train.py's evaluate()), and C1, C2 are their
measured per-token latencies (from measure_latency.py).

This does NOT require running any model - it only combines the four
already-measured numbers according to the formulas above, exactly as
described in the methodology. Run this after both baselines are trained
and their latency measured.

Usage:
    python hybrid_eval.py --L1 4.10 --L2 4.35 --C1 0.82 --C2 0.31 --beta 1.0
"""

import argparse
import csv


def lambda_sweep(L1: float, L2: float, C1: float, C2: float, beta: float,
                  lambdas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    results = []
    for lam in lambdas:
        L = lam * L1 + (1 - lam) * L2
        C = lam * C1 + (1 - lam) * C2
        J = L + beta * C
        results.append({"lambda": lam, "L": L, "C": C, "J": J})
    return results


def select_best_lambda(results):
    return min(results, key=lambda r: r["J"])


def print_and_save(results, best, out_csv: str = "lambda_sweep_results.csv"):
    print(f"{'lambda':>8} | {'L(lambda)':>10} | {'C(lambda)':>10} | {'J(lambda)':>10}")
    print("-" * 48)
    for r in results:
        marker = "  <-- best" if r is best else ""
        print(f"{r['lambda']:>8.2f} | {r['L']:>10.4f} | {r['C']:>10.4f} | "
              f"{r['J']:>10.4f}{marker}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "L", "C", "J"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {out_csv}")
    print(f"Selected lambda* = {best['lambda']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L1", type=float, required=True,
                         help="Transformer validation loss")
    parser.add_argument("--L2", type=float, required=True,
                         help="Mamba validation loss")
    parser.add_argument("--C1", type=float, required=True,
                         help="Transformer per-token latency")
    parser.add_argument("--C2", type=float, required=True,
                         help="Mamba per-token latency")
    parser.add_argument("--beta", type=float, default=1.0,
                         help="Weighting of cost relative to loss in J(lambda)")
    args = parser.parse_args()

    results = lambda_sweep(args.L1, args.L2, args.C1, args.C2, args.beta)
    best = select_best_lambda(results)
    print_and_save(results, best)
