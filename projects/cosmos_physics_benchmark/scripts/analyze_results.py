"""
Statistical analysis of trajectory extraction results across N runs.

Compares extracted Cosmos trajectories against physics ground truth,
computes per-run metrics, and generates aggregate statistics and plots.

Usage:
    python analyze_results.py \
        --trajectory_dir ../data/trajectories/free_fall \
        --scenario free_fall \
        --output_dir ../data/results/free_fall

Outputs:
    - per_run_metrics.csv: metrics for each individual run
    - aggregate_stats.json: mean, std, CI for all metrics across runs
    - trajectory_overlay.png: all trajectories + physics curve
    - gravity_histogram.png: distribution of estimated g values
    - error_vs_frame.png: per-frame error evolution
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from physics_ground_truth import (
    compare_trajectories,
    estimate_gravity,
    free_fall_trajectory,
    pendulum_trajectory,
    projectile_trajectory,
    ramp_trajectory,
)


def load_trajectory_csv(csv_path: str) -> dict:
    """Load a trajectory CSV into structured arrays.

    Returns:
        Dict with keys: frames, t_sec, x_norm, y_norm, detected (all numpy arrays).
    """
    frames, t_sec, x_norm, y_norm, detected = [], [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            t_sec.append(float(row["t_sec"]))
            if row["detected"] == "True" and row["y_norm"] not in ("", "None", None):
                x_norm.append(float(row["x_norm"]))
                y_norm.append(float(row["y_norm"]))
                detected.append(True)
            else:
                x_norm.append(float("nan"))
                y_norm.append(float("nan"))
                detected.append(False)

    return {
        "frames": np.array(frames),
        "t_sec": np.array(t_sec),
        "x_norm": np.array(x_norm),
        "y_norm": np.array(y_norm),
        "detected": np.array(detected),
    }


def get_ground_truth(scenario: str, n_frames: int = 93, fps: float = 16.0) -> np.ndarray:
    """Get the y-component ground truth for a given scenario."""
    if scenario == "free_fall":
        return free_fall_trajectory(n_frames, fps)
    elif scenario == "projectile":
        _, y = projectile_trajectory(n_frames, fps)
        return y
    elif scenario == "pendulum":
        _, y = pendulum_trajectory(n_frames, fps)
        return y
    elif scenario == "ramp":
        _, y = ramp_trajectory(n_frames, fps)
        return y
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def analyze_single_run(trajectory: dict, y_physics: np.ndarray, fps: float) -> dict:
    """Analyze a single run's trajectory against physics ground truth.

    Returns:
        Dict with all per-run metrics.
    """
    y_cosmos = trajectory["y_norm"]

    # Gravity estimation from quadratic fit
    grav = estimate_gravity(y_cosmos, fps=fps)

    # Comparison against physics ground truth
    comp = compare_trajectories(y_cosmos, y_physics)

    detection_rate = np.sum(trajectory["detected"]) / len(trajectory["detected"])

    return {
        **grav,
        "comparison_nrmse": comp.get("nrmse"),
        "comparison_mae": comp.get("mae"),
        "comparison_max_error": comp.get("max_error"),
        "comparison_r_squared": comp.get("r_squared"),
        "detection_rate": float(detection_rate),
    }


def compute_aggregate_stats(per_run_metrics: list[dict]) -> dict:
    """Compute aggregate statistics across all runs."""
    keys_to_aggregate = [
        "a_coeff",
        "r_squared",
        "nrmse",
        "mae",
        "comparison_nrmse",
        "comparison_mae",
        "comparison_max_error",
        "comparison_r_squared",
        "detection_rate",
    ]

    stats = {}
    for key in keys_to_aggregate:
        values = [m[key] for m in per_run_metrics if m.get(key) is not None]
        if values:
            arr = np.array(values)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "ci_95_lower": float(np.percentile(arr, 2.5)),
                "ci_95_upper": float(np.percentile(arr, 97.5)),
                "n_runs": len(values),
            }

    return stats


def plot_trajectory_overlay(
    trajectories: list[dict],
    y_physics: np.ndarray,
    output_path: str,
    fps: float = 16.0,
):
    """Plot all Cosmos trajectories overlaid with the physics ground truth."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual Cosmos trajectories (light, transparent)
    for traj in trajectories:
        valid = traj["detected"]
        ax.plot(
            traj["frames"][valid],
            traj["y_norm"][valid],
            color="steelblue",
            alpha=0.15,
            linewidth=0.8,
        )

    # Compute and plot mean trajectory
    max_frames = max(len(t["y_norm"]) for t in trajectories)
    all_y = np.full((len(trajectories), max_frames), np.nan)
    for i, traj in enumerate(trajectories):
        n = len(traj["y_norm"])
        all_y[i, :n] = traj["y_norm"]

    mean_y = np.nanmean(all_y, axis=0)
    std_y = np.nanstd(all_y, axis=0)
    frames = np.arange(max_frames)

    ax.plot(frames, mean_y, color="blue", linewidth=2, label="Cosmos mean")
    ax.fill_between(
        frames,
        mean_y - std_y,
        mean_y + std_y,
        color="blue",
        alpha=0.2,
        label="Cosmos +/- 1 std",
    )

    # Plot physics ground truth
    physics_frames = np.arange(len(y_physics))
    ax.plot(
        physics_frames,
        y_physics,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Physics (analytical)",
    )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("y position (normalized, 0=top, 1=bottom)")
    ax.set_title(f"Cosmos vs Physics: Trajectory Comparison (N={len(trajectories)} runs)")
    ax.legend()
    ax.set_xlim(0, max_frames)
    ax.set_ylim(-0.05, 1.05)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved trajectory overlay: {output_path}")


def plot_gravity_histogram(
    per_run_metrics: list[dict],
    output_path: str,
    true_g: float = 9.81,
):
    """Plot histogram of estimated gravity values across runs."""
    # Use the a_coeff directly since we don't know absolute scale
    a_coeffs = [m["a_coeff"] for m in per_run_metrics if m.get("a_coeff") is not None]
    if not a_coeffs:
        print("No valid a_coeff values for histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(a_coeffs, bins=min(20, len(a_coeffs)), color="steelblue", edgecolor="black", alpha=0.7)
    mean_a = np.mean(a_coeffs)
    ax.axvline(mean_a, color="blue", linestyle="-", linewidth=2, label=f"Mean: {mean_a:.6f}")

    ax.set_xlabel("Quadratic coefficient (a) in y = a*n^2 + b*n + c")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Estimated Acceleration (N={len(a_coeffs)} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved gravity histogram: {output_path}")


def plot_error_vs_frame(
    trajectories: list[dict],
    y_physics: np.ndarray,
    output_path: str,
):
    """Plot per-frame absolute error evolution."""
    max_frames = min(len(y_physics), max(len(t["y_norm"]) for t in trajectories))
    all_errors = np.full((len(trajectories), max_frames), np.nan)

    for i, traj in enumerate(trajectories):
        n = min(len(traj["y_norm"]), max_frames)
        errors = np.abs(traj["y_norm"][:n] - y_physics[:n])
        all_errors[i, :n] = errors

    mean_error = np.nanmean(all_errors, axis=0)
    std_error = np.nanstd(all_errors, axis=0)
    frames = np.arange(max_frames)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frames, mean_error, color="red", linewidth=2, label="Mean |error|")
    ax.fill_between(
        frames,
        np.maximum(mean_error - std_error, 0),
        mean_error + std_error,
        color="red",
        alpha=0.2,
        label="+/- 1 std",
    )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Absolute error (normalized)")
    ax.set_title("Per-Frame Error: Cosmos vs Physics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved error plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory results across N runs"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Directory containing trajectory CSVs",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["free_fall", "projectile", "pendulum", "ramp"],
        help="Physics scenario for ground truth comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis results and plots",
    )
    parser.add_argument("--fps", type=float, default=16.0, help="Video FPS")
    parser.add_argument("--n_frames", type=int, default=93, help="Frames per video")
    args = parser.parse_args()

    # Find all trajectory CSVs
    traj_dir = Path(args.trajectory_dir)
    csv_files = sorted(traj_dir.rglob("*.csv"))
    # Exclude summary files
    csv_files = [f for f in csv_files if f.name != "extraction_summary.json"]
    csv_files = [f for f in csv_files if f.suffix == ".csv"]

    if not csv_files:
        print(f"No CSV files found in {args.trajectory_dir}")
        return

    print(f"Found {len(csv_files)} trajectory files")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all trajectories
    trajectories = []
    for csv_path in csv_files:
        traj = load_trajectory_csv(str(csv_path))
        trajectories.append(traj)

    # Generate ground truth
    y_physics = get_ground_truth(args.scenario, args.n_frames, args.fps)

    # Analyze each run
    per_run_metrics = []
    for i, (csv_path, traj) in enumerate(zip(csv_files, trajectories)):
        metrics = analyze_single_run(traj, y_physics, args.fps)
        metrics["file"] = str(csv_path.relative_to(traj_dir))
        per_run_metrics.append(metrics)

    # Save per-run metrics
    metrics_path = os.path.join(args.output_dir, "per_run_metrics.csv")
    if per_run_metrics:
        fieldnames = list(per_run_metrics[0].keys())
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_run_metrics)
        print(f"Saved per-run metrics: {metrics_path}")

    # Compute aggregate statistics
    agg_stats = compute_aggregate_stats(per_run_metrics)
    agg_stats["scenario"] = args.scenario
    agg_stats["n_runs"] = len(per_run_metrics)
    agg_stats["fps"] = args.fps

    stats_path = os.path.join(args.output_dir, "aggregate_stats.json")
    with open(stats_path, "w") as f:
        json.dump(agg_stats, f, indent=2)
    print(f"Saved aggregate stats: {stats_path}")

    # Generate plots
    plot_trajectory_overlay(
        trajectories, y_physics, os.path.join(args.output_dir, "trajectory_overlay.png"), args.fps
    )
    plot_gravity_histogram(per_run_metrics, os.path.join(args.output_dir, "gravity_histogram.png"))
    plot_error_vs_frame(
        trajectories, y_physics, os.path.join(args.output_dir, "error_vs_frame.png")
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY: {args.scenario} ({len(per_run_metrics)} runs)")
    print("=" * 60)

    for key in ["a_coeff", "r_squared", "comparison_nrmse", "comparison_mae", "detection_rate"]:
        if key in agg_stats:
            s = agg_stats[key]
            print(f"  {key:>25s}: {s['mean']:.4f} +/- {s['std']:.4f}  (range: {s['min']:.4f} - {s['max']:.4f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
