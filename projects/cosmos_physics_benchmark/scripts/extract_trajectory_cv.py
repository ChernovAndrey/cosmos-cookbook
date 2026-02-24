"""
Extract object trajectories from generated videos using classical computer vision.

Primary extraction method: HSV color thresholding + contour detection + centroid
tracking. This is deterministic, pixel-precise, and requires no GPU.

For each video, outputs a CSV with columns: [frame, t_sec, x_norm, y_norm]
where x_norm and y_norm are in [0, 1] (0=top/left, 1=bottom/right).

Usage:
    python extract_trajectory_cv.py \
        --video_dir ../data/generated_videos/free_fall \
        --output_dir ../data/trajectories/free_fall

    # Custom HSV range for non-red objects:
    python extract_trajectory_cv.py \
        --video_dir ../data/generated_videos/free_fall \
        --output_dir ../data/trajectories/free_fall \
        --hsv_lower 0 100 100 \
        --hsv_upper 10 255 255

Hardware: CPU only (no GPU needed)
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


# Default HSV ranges for red ball detection.
# Red wraps around in HSV space, so we use two ranges and combine them.
DEFAULT_HSV_LOWER_1 = (0, 80, 80)
DEFAULT_HSV_UPPER_1 = (10, 255, 255)
DEFAULT_HSV_LOWER_2 = (170, 80, 80)
DEFAULT_HSV_UPPER_2 = (180, 255, 255)

# Minimum contour area (fraction of frame area) to filter noise
MIN_CONTOUR_AREA_FRAC = 0.0005


def extract_trajectory_from_video(
    video_path: str,
    hsv_ranges: list[tuple[tuple, tuple]],
    min_contour_area_frac: float = MIN_CONTOUR_AREA_FRAC,
    fps: float = 16.0,
) -> list[dict]:
    """Extract ball trajectory from a video using color thresholding.

    Args:
        video_path: Path to the input video file.
        hsv_ranges: List of (lower, upper) HSV tuples for color detection.
        min_contour_area_frac: Minimum contour area as fraction of frame area.
        fps: Frames per second of the video (default: 16 for Cosmos Predict 2.5).

    Returns:
        List of dicts with keys: frame, t_sec, x_norm, y_norm, area_px, detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    min_contour_area = frame_area * min_contour_area_frac

    trajectory = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Combine masks from all HSV ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = False
        cx_norm, cy_norm, area_px = 0.0, 0.0, 0.0

        if contours:
            # Take the largest contour that meets minimum area threshold
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    cx_norm = cx / frame_width  # 0=left, 1=right
                    cy_norm = cy / frame_height  # 0=top, 1=bottom
                    area_px = cv2.contourArea(largest)
                    detected = True

        trajectory.append(
            {
                "frame": frame_idx,
                "t_sec": frame_idx / fps,
                "x_norm": round(cx_norm, 6) if detected else None,
                "y_norm": round(cy_norm, 6) if detected else None,
                "area_px": round(area_px, 1) if detected else None,
                "detected": detected,
            }
        )
        frame_idx += 1

    cap.release()
    return trajectory


def save_trajectory_csv(trajectory: list[dict], output_path: str):
    """Save trajectory data to CSV."""
    fieldnames = ["frame", "t_sec", "x_norm", "y_norm", "area_px", "detected"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory)


def compute_detection_stats(trajectory: list[dict]) -> dict:
    """Compute detection quality statistics."""
    total_frames = len(trajectory)
    detected_frames = sum(1 for t in trajectory if t["detected"])
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0

    # Find first and last detected frame
    detected_indices = [t["frame"] for t in trajectory if t["detected"]]
    first_detected = detected_indices[0] if detected_indices else None
    last_detected = detected_indices[-1] if detected_indices else None

    # Check for gaps in detection
    gaps = []
    if len(detected_indices) > 1:
        for i in range(1, len(detected_indices)):
            gap = detected_indices[i] - detected_indices[i - 1]
            if gap > 1:
                gaps.append(
                    {
                        "start": detected_indices[i - 1],
                        "end": detected_indices[i],
                        "length": gap - 1,
                    }
                )

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": round(detection_rate, 4),
        "first_detected": first_detected,
        "last_detected": last_detected,
        "detection_gaps": gaps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract ball trajectories from generated videos using OpenCV"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing generated videos (searches recursively for .mp4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trajectory CSVs",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Video frame rate (default: 16 for Cosmos Predict 2.5)",
    )
    parser.add_argument(
        "--hsv_lower",
        type=int,
        nargs=3,
        default=None,
        help="Custom HSV lower bound (H S V). If not set, uses red detection.",
    )
    parser.add_argument(
        "--hsv_upper",
        type=int,
        nargs=3,
        default=None,
        help="Custom HSV upper bound (H S V). If not set, uses red detection.",
    )
    parser.add_argument(
        "--min_area_frac",
        type=float,
        default=MIN_CONTOUR_AREA_FRAC,
        help=f"Minimum contour area as fraction of frame (default: {MIN_CONTOUR_AREA_FRAC})",
    )
    args = parser.parse_args()

    # Set up HSV ranges
    if args.hsv_lower is not None and args.hsv_upper is not None:
        hsv_ranges = [(tuple(args.hsv_lower), tuple(args.hsv_upper))]
    else:
        # Default: red detection (two ranges because red wraps in HSV)
        hsv_ranges = [
            (DEFAULT_HSV_LOWER_1, DEFAULT_HSV_UPPER_1),
            (DEFAULT_HSV_LOWER_2, DEFAULT_HSV_UPPER_2),
        ]

    # Find all video files
    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.rglob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} videos in {args.video_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = {}

    for i, video_path in enumerate(video_files):
        # Preserve directory structure relative to video_dir
        rel_path = video_path.relative_to(video_dir)
        csv_name = rel_path.with_suffix(".csv")
        output_path = Path(args.output_dir) / csv_name

        os.makedirs(output_path.parent, exist_ok=True)

        print(f"[{i + 1}/{len(video_files)}] Processing: {rel_path}")

        trajectory = extract_trajectory_from_video(
            str(video_path),
            hsv_ranges=hsv_ranges,
            min_contour_area_frac=args.min_area_frac,
            fps=args.fps,
        )

        save_trajectory_csv(trajectory, str(output_path))

        stats = compute_detection_stats(trajectory)
        all_stats[str(rel_path)] = stats

        det_rate = stats["detection_rate"]
        status = "OK" if det_rate > 0.5 else "LOW"
        print(
            f"  -> {stats['detected_frames']}/{stats['total_frames']} frames "
            f"detected ({det_rate:.1%}) [{status}]"
        )

    # Save summary statistics
    summary_path = os.path.join(args.output_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved extraction summary to {summary_path}")

    # Print overall summary
    rates = [s["detection_rate"] for s in all_stats.values()]
    print(f"\nOverall: {len(video_files)} videos processed")
    print(f"  Mean detection rate: {np.mean(rates):.1%}")
    print(f"  Min detection rate:  {np.min(rates):.1%}")
    print(f"  Videos with >50% detection: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")


if __name__ == "__main__":
    main()
