"""
Analytical physics solutions for benchmark scenarios.

Generates ground-truth trajectories in normalized coordinates for comparison
against Cosmos-generated trajectories. All functions return arrays indexed by
frame number with positions in [0, 1] normalized space.

Usage (as library):
    from physics_ground_truth import free_fall_trajectory
    y_physics = free_fall_trajectory(n_frames=93, fps=16.0)

Usage (CLI -- generate ground truth CSVs):
    python physics_ground_truth.py \
        --scenario free_fall \
        --n_frames 93 \
        --fps 16 \
        --output ground_truth_free_fall.csv
"""

import argparse
import csv
import sys

import numpy as np


def free_fall_trajectory(
    n_frames: int = 93,
    fps: float = 16.0,
    g: float = 9.81,
    y_start: float = 0.05,
    drop_height_m: float = 50.0,
) -> np.ndarray:
    """Generate normalized free-fall trajectory.

    A ball released from rest falling under gravity.
    y(t) = y_start + (0.5 * g * t^2) / drop_height_m * (1 - y_start)

    Args:
        n_frames: Number of frames.
        fps: Frames per second.
        g: Gravitational acceleration (m/s^2).
        y_start: Starting y position (normalized, 0=top).
        drop_height_m: Physical drop height in meters (sets the scale).

    Returns:
        Array of shape (n_frames,) with y positions in [0, 1].
    """
    t = np.arange(n_frames) / fps
    y_m = 0.5 * g * t**2  # displacement in meters
    # Normalize: map [0, drop_height_m] to [y_start, 1.0]
    y_norm = y_start + (y_m / drop_height_m) * (1.0 - y_start)
    # Clip to [0, 1] (ball hits ground)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    return y_norm


def projectile_trajectory(
    n_frames: int = 93,
    fps: float = 16.0,
    g: float = 9.81,
    v0: float = 15.0,
    angle_deg: float = 45.0,
    x_start: float = 0.1,
    y_start: float = 0.7,
    scale_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized projectile trajectory.

    Ball launched at angle from lower-left.
    x(t) = v0 * cos(a) * t
    y(t) = v0 * sin(a) * t - 0.5 * g * t^2

    Args:
        n_frames: Number of frames.
        fps: Frames per second.
        g: Gravitational acceleration.
        v0: Initial velocity (m/s).
        angle_deg: Launch angle in degrees.
        x_start: Starting x position (normalized).
        y_start: Starting y position (normalized, 0=top).
        scale_m: Physical scale in meters (frame width/height).

    Returns:
        Tuple of (x_norm, y_norm) arrays, each shape (n_frames,).
    """
    t = np.arange(n_frames) / fps
    angle_rad = np.radians(angle_deg)

    x_m = v0 * np.cos(angle_rad) * t
    y_m = v0 * np.sin(angle_rad) * t - 0.5 * g * t**2

    x_norm = x_start + x_m / scale_m
    # y is inverted: positive y_m (upward) decreases y_norm
    y_norm = y_start - y_m / scale_m

    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    return x_norm, y_norm


def pendulum_trajectory(
    n_frames: int = 93,
    fps: float = 16.0,
    g: float = 9.81,
    length_m: float = 2.0,
    theta0_deg: float = 30.0,
    pivot_x: float = 0.5,
    pivot_y: float = 0.05,
    visual_length: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized pendulum trajectory (small-angle approximation).

    theta(t) = theta0 * cos(sqrt(g/L) * t)

    Args:
        n_frames: Number of frames.
        fps: Frames per second.
        g: Gravitational acceleration.
        length_m: Pendulum length in meters.
        theta0_deg: Initial angular displacement.
        pivot_x: Pivot x position (normalized).
        pivot_y: Pivot y position (normalized).
        visual_length: Visual length of pendulum (normalized).

    Returns:
        Tuple of (x_norm, y_norm) arrays.
    """
    t = np.arange(n_frames) / fps
    theta0 = np.radians(theta0_deg)
    omega = np.sqrt(g / length_m)

    theta = theta0 * np.cos(omega * t)

    x_norm = pivot_x + visual_length * np.sin(theta)
    y_norm = pivot_y + visual_length * np.cos(theta)

    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    return x_norm, y_norm


def ramp_trajectory(
    n_frames: int = 93,
    fps: float = 16.0,
    g: float = 9.80665,
    angle_deg: float = 45.0,
    ramp_start_x: float = 0.05,
    ramp_start_y: float = 0.05,
    ramp_length_norm: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized trajectory for a solid sphere rolling without slipping down a ramp.

    s(t) = 0.5 * (5/7) * g * sin(angle) * t^2  (distance along ramp)

    For a solid sphere (I = 2/5 mR^2) rolling without slipping, the
    effective linear acceleration along the incline is (5/7) * g * sin(angle).

    Args:
        n_frames: Number of frames.
        fps: Frames per second.
        g: Gravitational acceleration (m/s^2).
        angle_deg: Ramp angle in degrees.
        ramp_start_x: Top of ramp x position (normalized).
        ramp_start_y: Top of ramp y position (normalized).
        ramp_length_norm: Visual length of ramp (normalized).

    Returns:
        Tuple of (x_norm, y_norm) arrays.
    """
    t = np.arange(n_frames) / fps
    angle_rad = np.radians(angle_deg)
    # Solid sphere rolling without slipping: a = (5/7) * g * sin(angle)
    a_eff = (5.0 / 7.0) * g * np.sin(angle_rad)

    # Distance along ramp (normalized)
    # Choose scale so the ball traverses the ramp in about 3 seconds
    ramp_length_m = 0.5 * a_eff * 3.0**2
    s_norm = 0.5 * a_eff * t**2 / ramp_length_m * ramp_length_norm
    s_norm = np.clip(s_norm, 0.0, ramp_length_norm)

    x_norm = ramp_start_x + s_norm * np.cos(angle_rad)
    y_norm = ramp_start_y + s_norm * np.sin(angle_rad)

    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    return x_norm, y_norm


# ---------------------------------------------------------------------------
# Trajectory comparison utilities
# ---------------------------------------------------------------------------


def estimate_gravity(
    y_observed: np.ndarray,
    fps: float = 16.0,
    fit_method: str = "polyfit",
) -> dict:
    """Estimate effective gravity from an observed free-fall trajectory.

    Fits y = a * n^2 + b * n + c to the frame-indexed trajectory,
    then derives g_est = 2 * a * fps^2 (after scale normalization).

    Args:
        y_observed: Array of y positions (normalized, 0=top, 1=bottom).
        fps: Frames per second.
        fit_method: 'polyfit' for numpy polyfit.

    Returns:
        Dict with: g_est (if scale known), a_coeff, b_coeff, c_coeff,
                   r_squared, residuals.
    """
    n = np.arange(len(y_observed))
    valid = ~np.isnan(y_observed)
    n_valid = n[valid]
    y_valid = y_observed[valid]

    if len(y_valid) < 3:
        return {"error": "Too few valid points for quadratic fit"}

    # Fit quadratic: y = a*n^2 + b*n + c
    coeffs = np.polyfit(n_valid, y_valid, 2)
    a, b, c = coeffs

    # Compute R^2
    y_pred = np.polyval(coeffs, n_valid)
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # The effective acceleration in normalized_units / frame^2
    a_eff = 2 * a

    # NRMSE
    nrmse = np.sqrt(np.mean((y_valid - y_pred) ** 2)) / (np.max(y_valid) - np.min(y_valid)) if np.max(y_valid) > np.min(y_valid) else float("inf")

    return {
        "a_coeff": float(a),
        "b_coeff": float(b),
        "c_coeff": float(c),
        "a_eff_per_frame2": float(a_eff),
        "r_squared": float(r_squared),
        "nrmse": float(nrmse),
        "mae": float(np.mean(np.abs(y_valid - y_pred))),
        "max_error": float(np.max(np.abs(y_valid - y_pred))),
        "n_points": int(len(y_valid)),
    }


def compare_trajectories(
    y_cosmos: np.ndarray,
    y_physics: np.ndarray,
) -> dict:
    """Compare a Cosmos trajectory against physics ground truth.

    Args:
        y_cosmos: Observed trajectory from Cosmos (normalized).
        y_physics: Analytical physics trajectory (normalized).

    Returns:
        Dict with comparison metrics.
    """
    # Align lengths
    min_len = min(len(y_cosmos), len(y_physics))
    yc = y_cosmos[:min_len]
    yp = y_physics[:min_len]

    # Handle NaN in cosmos trajectory
    valid = ~np.isnan(yc)
    yc_valid = yc[valid]
    yp_valid = yp[valid]

    if len(yc_valid) == 0:
        return {"error": "No valid data points"}

    errors = yc_valid - yp_valid
    abs_errors = np.abs(errors)

    y_range = np.max(yp_valid) - np.min(yp_valid)
    nrmse = np.sqrt(np.mean(errors**2)) / y_range if y_range > 0 else float("inf")

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((yp_valid - np.mean(yp_valid)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "nrmse": float(nrmse),
        "mae": float(np.mean(abs_errors)),
        "max_error": float(np.max(abs_errors)),
        "r_squared": float(r_squared),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "n_valid_points": int(len(yc_valid)),
        "n_total_points": int(min_len),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate physics ground truth trajectories"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["free_fall", "projectile", "pendulum", "ramp"],
        help="Physics scenario",
    )
    parser.add_argument("--n_frames", type=int, default=93, help="Number of frames")
    parser.add_argument("--fps", type=float, default=16.0, help="Frames per second")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    frames = np.arange(args.n_frames)
    t_sec = frames / args.fps

    if args.scenario == "free_fall":
        y = free_fall_trajectory(args.n_frames, args.fps)
        x = np.full_like(y, 0.5)
    elif args.scenario == "projectile":
        x, y = projectile_trajectory(args.n_frames, args.fps)
    elif args.scenario == "pendulum":
        x, y = pendulum_trajectory(args.n_frames, args.fps)
    elif args.scenario == "ramp":
        x, y = ramp_trajectory(args.n_frames, args.fps)
    else:
        print(f"Unknown scenario: {args.scenario}")
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_sec", "x_norm", "y_norm"])
        for i in range(args.n_frames):
            writer.writerow([i, round(t_sec[i], 6), round(x[i], 6), round(y[i], 6)])

    print(f"Saved {args.scenario} ground truth ({args.n_frames} frames) to {args.output}")


if __name__ == "__main__":
    main()
