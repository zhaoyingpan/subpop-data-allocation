"""Analyze the n^3 training configurations used in the paper."""

import argparse
import itertools
from pathlib import Path
import numpy as np

from estimate import Estimate, compute_group_sizes


def direction_similarity(estimator, params):
    """Return the cosine similarity between the Grad and Bal directions."""
    gradient = estimator.get_derivative(params)
    target = np.full(4, 0.25) - np.asarray(compute_group_sizes(*params))
    jacobian = estimator._compute_jacobian(*params)
    balance, _, _, _ = np.linalg.lstsq(jacobian, target, rcond=None)

    gradient_norm = np.linalg.norm(gradient)
    balance_norm = np.linalg.norm(balance)
    if min(gradient_norm, balance_norm) <= estimator.zero_tolerance:
        return 0.0
    return np.dot(gradient, balance) / (gradient_norm * balance_norm)


def analyze_configurations(m1=2.0, m2=2.0, grid_size=7):
    """Estimate performance on an evenly spaced grid of configuration parameters.

    grid_size=7 samples each of alpha1, alpha0, and beta at 1/8, ..., 7/8, producing 7^3 = 343 configurations.
    we also used grid_size=49 with a reference result in ./theory/reference/grid49_m1_2_m2_2.npz

    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    values = np.linspace(
        1.0 / (grid_size + 1),
        grid_size / (grid_size + 1),
        grid_size,
    )
    all_params = np.asarray(list(itertools.product(values, repeat=3)))
    ratios = np.asarray([compute_group_sizes(*param) for param in all_params])

    estimator = Estimate(m1=m1, m2=m2)
    acc = np.asarray([estimator.estimate_acc(param) for param in all_params])
    all_sim = np.asarray(
        [direction_similarity(estimator, param) for param in all_params]
    )
    all_acc_sim = []
    for param in all_params:
        grad_acc, _ = estimator.estimate_trajectory(
            param, delta_p=0.01, num_steps=10
        )
        bal_acc, _ = estimator.baseline_trajectory(
            param, delta_p=0.01, traj_type="bal", num_steps=10
        )
        all_acc_sim.append(grad_acc[-1] - bal_acc[-1])

    return {
        "acc": acc,
        "ratios": ratios,
        "all_params": all_params,
        "all_sim": all_sim,
        "all_acc_sim": np.asarray(all_acc_sim),
    }


def print_summary(result, percentile):
    adjusted = result["acc"].mean(axis=1)
    worst_group = result["acc"].min(axis=1)
    threshold = np.percentile(adjusted, percentile)
    high_performance = adjusted >= threshold

    print(f"Configurations: {len(adjusted)}")
    print(
        f"Adjusted accuracy: min={adjusted.min():.4f}, "
        f"max={adjusted.max():.4f}"
    )
    print(
        f"Worst-group accuracy: min={worst_group.min():.4f}, "
        f"max={worst_group.max():.4f}"
    )
    print(
        f"Balanced adjusted accuracy: "
        f"{result['acc'][len(adjusted) // 2].mean():.4f}"
    )
    print(
        f"{percentile:g}th-percentile threshold: "
        f"{threshold:.4f} "
        f"({high_performance.sum()} configurations at or above threshold)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the theoretical performance of a configuration grid."
    )
    parser.add_argument("--m1", type=float, default=2.0)
    parser.add_argument("--m2", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .npz file for the analysis results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_configurations(
        m1=args.m1,
        m2=args.m2,
        grid_size=args.grid_size,
    )
    print_summary(result, args.percentile)

    if args.output is not None:
        if args.output.suffix != ".npz":
            raise ValueError("output must use the .npz extension")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **result)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
