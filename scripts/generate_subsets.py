#!/usr/bin/env python3
"""Generate the CelebA and CivilComments n=8 data subsets."""

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from theory.estimate import compute_group_sizes
from utils import get_subgroup_masks


DATASETS = {
    "celeba": {
        "metadata": Path("celeba/metadata_celeba.csv"),
        "num_samples": 2000,
    },
    "civilcomments": {
        "metadata": Path("civilcomments/metadata_civilcomments_coarse.csv"),
        "num_samples": 10000,
    },
}


def configuration_grid(values):
    """Return the configuration grid and its four-group allocation ratios."""
    all_params = np.asarray(list(itertools.product(values, repeat=3)))
    ratios = np.asarray([compute_group_sizes(*params) for params in all_params])
    return {
        "ratios": ratios,
        "all_params": all_params,
    }


def get_subset_df(metadata, group_counts, random_state=42):
    """Sample training groups and retain the original validation/test rows."""
    train_df = metadata[metadata["split"] == 0]
    masks = get_subgroup_masks(
        train_df["y"].to_numpy(), train_df["a"].to_numpy()
    )

    sampled_groups = []
    for mask, count in zip(masks, group_counts):
        count = int(count)
        group_df = train_df[mask]
        if count <= 0 or group_df.empty:
            continue
        if len(group_df) >= count:
            sampled = group_df.sample(n=count, random_state=random_state)
        else:
            sampled = group_df.sample(n=len(group_df), random_state=random_state)
            additional = group_df.sample(
                n=count - len(group_df),
                replace=True,
                random_state=random_state,
            )
            sampled = pd.concat([sampled, additional], ignore_index=True)
        sampled_groups.append(sampled)

    if sampled_groups:
        sampled_train = pd.concat(sampled_groups, ignore_index=True)
    else:
        sampled_train = pd.DataFrame(columns=metadata.columns)
    other_splits = metadata[metadata["split"] != 0]
    return pd.concat([sampled_train, other_splits], ignore_index=True)


def output_paths(data_root, output_dir, dataset, n, count):
    subset_dir = data_root / dataset / f"subsets_n{n}"
    csv_paths = [subset_dir / f"{index:02d}.csv" for index in range(count)]
    result_path = output_dir / f"{dataset}_subsets_n{n}.pkl"
    return subset_dir, csv_paths, result_path


def ensure_outputs_are_new(paths):
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = "\n".join(f"  {path}" for path in existing[:10])
        remainder = len(existing) - min(len(existing), 10)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileExistsError(
            "Refusing to overwrite existing CSV or pickle files:\n"
            f"{preview}{suffix}"
        )


def generate_dataset(dataset, spec, data_root, output_dir, n):
    metadata_path = data_root / spec["metadata"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    values = np.linspace(1.0 / n, (n - 1.0) / n, n - 1)
    result = configuration_grid(values)
    subset_dir, csv_paths, result_path = output_paths(
        data_root, output_dir, dataset, n, len(result["ratios"])
    )

    metadata = pd.read_csv(metadata_path)
    subset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ratio, csv_path in zip(result["ratios"], csv_paths):
        counts = np.round(spec["num_samples"] * ratio)
        subset = get_subset_df(metadata, counts)
        subset.to_csv(csv_path, index=False)

    result["metadata"] = [str(path) for path in csv_paths]
    result["param_dict"] = {
        "grad": {f"grad_{i}": [i] for i in range(len(csv_paths))},
        "bal": {f"bal_{i}": [i] for i in range(len(csv_paths))},
        "acc": {
            f"acc_{i}": params.tolist()
            for i, params in enumerate(result["all_params"])
        },
    }
    with result_path.open("wb") as file:
        pickle.dump(result, file)

    print(f"Generated {len(csv_paths)} subsets in {subset_dir}")
    print(f"Saved generation metadata to {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate n=8 configuration-grid subsets without overwriting outputs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--n", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n < 2:
        raise ValueError("n must be at least 2")

    plans = []
    for dataset in args.datasets:
        count = (args.n - 1) ** 3
        paths = output_paths(
            args.data_root, args.output_dir, dataset, args.n, count
        )
        plans.append((dataset, DATASETS[dataset], paths))

    # Check every selected dataset before writing anything, avoiding partial runs.
    ensure_outputs_are_new(
        [path for _, _, (_, csv_paths, result_path) in plans
         for path in [*csv_paths, result_path]]
    )
    for dataset, spec, _ in plans:
        generate_dataset(
            dataset, spec, args.data_root, args.output_dir, args.n
        )


if __name__ == "__main__":
    main()
