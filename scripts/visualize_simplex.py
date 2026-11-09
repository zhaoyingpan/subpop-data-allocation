"""Visualize theoretical estimates or empirical results on a 3-simplex."""

import argparse
import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from theory.estimate import compute_group_sizes


VERTICES = np.array([
    [1.0, 0.0, -1.0 / np.sqrt(2.0)],
    [-1.0, 0.0, -1.0 / np.sqrt(2.0)],
    [0.0, 1.0, 1.0 / np.sqrt(2.0)],
    [0.0, -1.0, 1.0 / np.sqrt(2.0)],
])
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
VERTEX_LABELS = (
    r"$r_{1,1}=1$",
    r"$r_{1,-1}=1$",
    r"$r_{-1,1}=1$",
    r"$r_{-1,-1}=1$",
)


class _ResultRecord:
    """Placeholder used to load saved AverageMeter records without training code."""


class _ResultUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "eval" and name == "AverageMeter":
            return _ResultRecord
        return super().find_class(module, name)


def project_to_3d(ratios):
    """Project four subgroup proportions onto a regular tetrahedron."""
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape[-1] != 4:
        raise ValueError("Expected subgroup ratios with a final dimension of four")
    return ratios @ VERTICES


def setup_simplex(title=None, angles=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for start, end in EDGES:
        ax.plot(
            VERTICES[[start, end], 0],
            VERTICES[[start, end], 1],
            VERTICES[[start, end], 2],
            "k--",
            alpha=0.5,
        )

    for vertex, label in zip(VERTICES, VERTEX_LABELS):
        ax.text(*(1.2 * vertex), label, ha="center", va="center")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(title)
    if angles is not None:
        if len(angles) != 2:
            raise ValueError("angles must contain elevation and azimuth")
        ax.view_init(elev=angles[0], azim=angles[1])
    return fig, ax


def plot_performance(
    ratios,
    performance,
    title,
    label,
    mask=None,
    marker_size=40,
    highlight=False,
    colorbar_limits=(0.0, 1.0),
    show_balanced=False,
    output=None,
    show=False,
    angles=None,
):
    """Plot theoretical or empirical performance with shared simplex styling."""
    ratios = np.asarray(ratios, dtype=float)
    performance = np.asarray(performance, dtype=float)
    if len(ratios) != len(performance):
        raise ValueError("ratios and performance must have the same length")
    if marker_size <= 0.0:
        raise ValueError("marker-size must be positive")
    if mask is None:
        mask = np.ones(len(performance), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(performance),) or not mask.any():
        raise ValueError("mask must select at least one configuration")

    points = project_to_3d(ratios[mask])
    fig, ax = setup_simplex(title, angles)
    if highlight:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="yellow",
            s=marker_size,
        )
    else:
        if len(colorbar_limits) != 2 or colorbar_limits[0] >= colorbar_limits[1]:
            raise ValueError("colorbar-limits must contain increasing lower and upper bounds")
        norm = colors.Normalize(vmin=colorbar_limits[0], vmax=colorbar_limits[1])
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=performance[mask],
            cmap="viridis",
            norm=norm,
            s=marker_size,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.55, label=label)

    if show_balanced:
        center = project_to_3d([[0.25, 0.25, 0.25, 0.25]])[0]
        ax.scatter(*center, c="darkblue", s=80, label="Balanced configuration")
        ax.legend(loc="lower center")

    _finish_figure(fig, output, show)
    return fig, ax


def plot_direction(
    params,
    direction,
    step_size=0.1,
    arrow_length=0.35,
    label="Direction",
    color="gray",
    linestyle="-",
    ax=None,
    output=None,
    show=False,
    angles=None,
):
    """Plot one theoretical parameter-space direction from one configuration."""
    params = np.asarray(params, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if params.shape != (3,) or direction.shape != (3,):
        raise ValueError("params and direction must each contain three values")
    if np.any((params <= 0.0) | (params >= 1.0)):
        raise ValueError("configuration parameters must be strictly between 0 and 1")
    if step_size <= 0.0:
        raise ValueError("step-size must be positive")
    if arrow_length <= 0.0:
        raise ValueError("arrow-length must be positive")

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    next_params = params + step_size * direction / direction_norm
    if np.any((next_params < 0.0) | (next_params > 1.0)):
        raise ValueError("direction leaves the valid configuration range; reduce step-size")

    start_ratio = np.asarray(compute_group_sizes(*params))
    end_ratio = np.asarray(compute_group_sizes(*next_params))
    start = project_to_3d(start_ratio)
    end = project_to_3d(end_ratio)
    projected_direction = end - start
    projected_norm = np.linalg.norm(projected_direction)
    if projected_norm == 0.0:
        raise ValueError("direction has zero length after simplex projection")
    displayed_direction = arrow_length * projected_direction / projected_norm

    if ax is None:
        fig, ax = setup_simplex("Estimated Direction", angles)
        ax.scatter(*start, c="royalblue", s=90, label="Initial configuration")
        center = project_to_3d([[0.25, 0.25, 0.25, 0.25]])[0]
        ax.scatter(*center, c="darkblue", s=80, label="3-simplex center")
    else:
        fig = ax.figure
    ax.quiver(
        *start,
        *displayed_direction,
        color=color,
        linewidth=2.5,
        arrow_length_ratio=0.25,
        linestyle=linestyle,
        label=label,
    )
    ax.legend(loc="lower center")
    _finish_figure(fig, output, show)
    return fig, ax


def load_results(path):
    """Load a trusted result file produced by ExperimentRunner."""
    with Path(path).open("rb") as file:
        result = _ResultUnpickler(file).load()
    if not isinstance(result, dict) or not result:
        raise ValueError("Expected a nonempty result dictionary")
    return result


def _adjusted_accuracy(record, task_name):
    accuracy = np.asarray(record.data["te_acc_grps"], dtype=float)
    if accuracy.ndim == 1:
        accuracy = accuracy.reshape(1, -1)
    if accuracy.ndim != 2 or accuracy.shape[1] != 4:
        raise ValueError(f"Unexpected test accuracy shape for {task_name}: {accuracy.shape}")
    completed = np.any(accuracy != 0.0, axis=1)
    if not completed.any():
        raise ValueError(f"No completed seed results found for {task_name}")
    return accuracy[completed].mean()


def _empirical_entries(result):
    """Return initial ratios, accuracies, and grid indices from an empirical run."""
    grid_size = round(len(result) ** (1.0 / 3.0))
    grid_params = None
    if grid_size ** 3 == len(result):
        values = np.linspace(1.0 / (grid_size + 1), grid_size / (grid_size + 1), grid_size)
        grid_params = list(itertools.product(values, repeat=3))

    ratios = []
    performance = []
    indices = []
    for task_name, record in result.items():
        allocation = record.info["data"]["params_dict"][task_name]
        allocation = np.asarray(allocation)
        if allocation.shape == (1,):
            if grid_params is None:
                raise ValueError("Indexed allocations require an n^3 empirical grid")
            index = int(allocation[0])
            if index < 0 or index >= len(grid_params):
                raise ValueError(f"Invalid configuration index for {task_name}: {index}")
            ratio = compute_group_sizes(*grid_params[index])
        elif allocation.shape == (3,):
            if np.any((allocation < 0.0) | (allocation > 1.0)):
                raise ValueError(f"Invalid configuration parameters for {task_name}")
            index = None
            ratio = compute_group_sizes(*allocation)
        else:
            raise ValueError(
                f"Expected a grid index or three configuration parameters for {task_name}"
            )
        ratios.append(ratio)
        performance.append(_adjusted_accuracy(record, task_name))
        indices.append(index)
    return np.asarray(ratios), np.asarray(performance), indices


def extract_plot_values(result_path, plot_type, comparison_path=None):
    """Extract simplex ratios and values from a theoretical NPZ or empirical pickle."""
    result_path = Path(result_path)
    if result_path.suffix == ".npz":
        if comparison_path is not None:
            raise ValueError("A comparison file is only used with empirical accuracy gaps")
        with np.load(result_path) as result:
            ratios = result["ratios"]
            if plot_type == "high-performance":
                values = result["acc"].mean(axis=1)
            elif plot_type == "direction-similarity":
                values = result["all_sim"]
            else:
                values = result["all_acc_sim"].mean(axis=1)
    elif result_path.suffix in (".pkl", ".pickle"):
        result = load_results(result_path)
        ratios, values, indices = _empirical_entries(result)
        if plot_type == "direction-similarity":
            raise ValueError("Direction similarity is only available in theoretical NPZ files")
        if plot_type == "accuracy-gap":
            if comparison_path is None:
                raise ValueError("Empirical accuracy gaps require --comparison")
            comparison = load_results(comparison_path)
            _, comparison_values, comparison_indices = _empirical_entries(comparison)
            if all(index is not None for index in indices + comparison_indices):
                comparison_by_index = dict(zip(comparison_indices, comparison_values))
                try:
                    comparison_values = np.asarray(
                        [comparison_by_index[index] for index in indices]
                    )
                except KeyError as error:
                    raise ValueError("The empirical runs use different configurations") from error
            elif len(values) != len(comparison_values):
                raise ValueError("The empirical runs contain different numbers of configurations")
            values = values - comparison_values
        elif comparison_path is not None:
            raise ValueError("--comparison is only used with accuracy-gap")
    else:
        raise ValueError("The result file must use the .npz, .pkl, or .pickle extension")

    ratios = np.asarray(ratios, dtype=float)
    values = np.asarray(values, dtype=float)
    if ratios.ndim != 2 or ratios.shape[1] != 4 or len(ratios) != len(values):
        raise ValueError("Expected matching N x 4 ratios and N visualization values")
    return ratios, values


def _finish_figure(fig, output, show):
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.08, top=0.92)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output}")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", type=Path, help="Theoretical .npz or empirical .pkl result."
    )
    parser.add_argument(
        "--plot",
        choices=("high-performance", "direction-similarity", "accuracy-gap"),
        required=True,
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Bal result pickle when the main result is a Grad result.",
    )
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ratios, values = extract_plot_values(
        args.result,
        args.plot,
        comparison_path=args.comparison,
    )

    if args.plot == "high-performance":
        if not 0.0 <= args.percentile <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        threshold = np.percentile(values, args.percentile)
        mask = values >= threshold
        print(f"{args.percentile:g}th-percentile threshold: {threshold:.4f}")
        plot_performance(
            ratios,
            values,
            title="High-performance Configurations",
            label="Adjusted accuracy",
            mask=mask,
            highlight=True,
            output=args.output,
            show=args.show,
        )
    else:
        if args.plot == "accuracy-gap":
            value_range = np.ptp(values)
            values = (
                np.zeros_like(values)
                if value_range == 0.0
                else (values - values.min()) / value_range
            )
            title = "Grad–Bal Accuracy Gap"
            label = "Normalized accuracy gap"
        else:
            title = "Grad–Bal Direction Similarity"
            label = "Cosine similarity"
        plot_performance(
            ratios,
            values,
            title=title,
            label=label,
            output=args.output,
            show=args.show,
        )


if __name__ == "__main__":
    main()
