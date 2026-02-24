"""
Fallback trajectory extraction using Cosmos Reason 2-2B as a VLM.

For complex backgrounds where classical CV (extract_trajectory_cv.py) fails,
this script uses Cosmos Reason 2 to detect the ball position in each frame
via structured JSON output.

Usage:
    python extract_trajectory_vlm.py \
        --video_dir ../data/generated_videos/free_fall \
        --output_dir ../data/trajectories/free_fall_vlm \
        --model_path <path_to_cosmos_reason2_2b>

Hardware: 24GB GPU (Cosmos Reason 2-2B)
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


# System prompt for structured position extraction
SYSTEM_PROMPT = (
    "You are a precise visual measurement tool. For each frame, identify the "
    "position of the specified object and report its center coordinates as "
    "normalized fractions of the frame dimensions."
)

# Per-frame extraction prompt
FRAME_PROMPT = (
    "In this frame, identify the red ball. Report its center position as JSON: "
    '{"x": <fraction_from_left>, "y": <fraction_from_top>} '
    "where 0.0 is the top/left edge and 1.0 is the bottom/right edge. "
    "If no red ball is visible, respond with: "
    '{"x": null, "y": null}'
)


def load_reason_model(model_path: str, device: str = "cuda"):
    """Load Cosmos Reason 2-2B model.

    Adapt this function to the actual Cosmos Reason 2 inference API.
    See: https://github.com/nvidia-cosmos/cosmos-reason1
    """
    # NOTE: Update with actual API. Example pattern:
    #   from cosmos_reason import CosmosReasonPipeline
    #   pipeline = CosmosReasonPipeline.from_pretrained(model_path)
    #   pipeline = pipeline.to(device)
    #   return pipeline

    raise NotImplementedError(
        f"Update load_reason_model() with the actual Cosmos Reason 2 API.\n"
        f"Model path: {model_path}\n"
        f"See: https://github.com/nvidia-cosmos/cosmos-reason1"
    )


def extract_position_from_frame(pipeline, frame: np.ndarray) -> dict:
    """Use Cosmos Reason 2 to extract ball position from a single frame.

    Args:
        pipeline: Loaded Cosmos Reason 2 model.
        frame: BGR image as numpy array.

    Returns:
        Dict with keys: x (float or None), y (float or None).
    """
    # NOTE: Adapt to actual Cosmos Reason 2 inference API.
    #
    # Example pattern:
    #   # Convert BGR to RGB
    #   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #   response = pipeline(
    #       system_prompt=SYSTEM_PROMPT,
    #       user_prompt=FRAME_PROMPT,
    #       image=frame_rgb,
    #       max_new_tokens=64,
    #       temperature=0.1,
    #   )
    #   # Parse JSON from response
    #   result = json.loads(response.text)
    #   return {"x": result.get("x"), "y": result.get("y")}

    raise NotImplementedError(
        "Update extract_position_from_frame() with actual Cosmos Reason 2 API."
    )


def extract_trajectory_from_video(
    video_path: str,
    pipeline,
    fps: float = 16.0,
    sample_every: int = 1,
) -> list[dict]:
    """Extract ball trajectory from a video using VLM per-frame analysis.

    Args:
        video_path: Path to video file.
        pipeline: Loaded Cosmos Reason 2 model.
        fps: Video frame rate.
        sample_every: Process every Nth frame (1 = all frames).

    Returns:
        List of dicts with keys: frame, t_sec, x_norm, y_norm, detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    trajectory = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            position = extract_position_from_frame(pipeline, frame)
            detected = position["x"] is not None and position["y"] is not None
            trajectory.append(
                {
                    "frame": frame_idx,
                    "t_sec": round(frame_idx / fps, 6),
                    "x_norm": round(position["x"], 6) if detected else None,
                    "y_norm": round(position["y"], 6) if detected else None,
                    "area_px": None,  # VLM doesn't report area
                    "detected": detected,
                }
            )
        frame_idx += 1

    cap.release()
    return trajectory


def save_trajectory_csv(trajectory: list[dict], output_path: str):
    """Save trajectory data to CSV (same format as CV extraction)."""
    fieldnames = ["frame", "t_sec", "x_norm", "y_norm", "area_px", "detected"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory)


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectories using Cosmos Reason 2-2B (VLM fallback)"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing generated videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trajectory CSVs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/cosmos-reason2-2b",
        help="Path or HF ID for Cosmos Reason 2-2B",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Video frame rate (default: 16)",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.rglob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} videos")
    print(f"Loading Cosmos Reason 2-2B from {args.model_path}...")
    pipeline = load_reason_model(args.model_path, device=args.device)
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, video_path in enumerate(video_files):
        rel_path = video_path.relative_to(video_dir)
        csv_name = rel_path.with_suffix(".csv")
        output_path = Path(args.output_dir) / csv_name
        os.makedirs(output_path.parent, exist_ok=True)

        print(f"[{i + 1}/{len(video_files)}] Processing: {rel_path}")

        trajectory = extract_trajectory_from_video(
            str(video_path),
            pipeline,
            fps=args.fps,
            sample_every=args.sample_every,
        )

        save_trajectory_csv(trajectory, str(output_path))
        detected = sum(1 for t in trajectory if t["detected"])
        print(f"  -> {detected}/{len(trajectory)} frames detected")

    print(f"\nDone. Trajectories saved to {args.output_dir}")


if __name__ == "__main__":
    main()
