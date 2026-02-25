"""
Generate physics scenario videos using Cosmos Predict 2.5.

This script generates N videos for a given physics scenario by prompting
Cosmos Predict 2.5 (2B or 14B) with carefully crafted text prompts. Each
run uses a different random seed to produce statistical variation.

Inference uses the cosmos-predict2.5 CLI (torchrun examples/inference.py)
from the cloned repo. Set --cosmos_dir to point to your local clone.

Usage:
    # Default (2B model, ~25GB VRAM):
    python generate_scenarios.py \
        --scenario free_fall \
        --n_runs 30 \
        --output_dir ../data/generated_videos/free_fall \
        --cosmos_dir /path/to/cosmos-predict2.5

    # 14B model (~55-65GB VRAM, A100 80GB required):
    python generate_scenarios.py \
        --scenario free_fall \
        --model 14b \
        --n_runs 30 \
        --output_dir ../data/generated_videos/free_fall_14b \
        --cosmos_dir /path/to/cosmos-predict2.5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "2b": {
        "hf_id": "2B/post-trained",
        "params": "2B",
        "vram_gb": "~25",
        "min_gpu": "A100 40GB",
    },
    "2b-distill": {
        "hf_id": "2B/distilled",
        "params": "2B-distilled",
        "vram_gb": "~25",
        "min_gpu": "A100 40GB",
    },
    "14b": {
        "hf_id": "14B/post-trained",
        "params": "14B",
        "vram_gb": "~55-65",
        "min_gpu": "A100 80GB",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates for each physics scenario
# ---------------------------------------------------------------------------
# Each scenario has multiple prompt variations for sensitivity analysis.
# The prompts are designed for:
#   - Simple, trackable objects (bright colored ball against plain background)
#   - Stationary camera to simplify trajectory extraction
#   - Physics-relevant motion within the 5.8s generation window

SCENARIO_PROMPTS = {
    "free_fall": {
        "description": "Ball dropped from height under gravity",
        "physics": "y = 0.5 * g * t^2",
        "prompts": [
            (
                "A high-definition video captures a physics experiment in a "
                "bright, cleanly lit indoor laboratory with plain white walls "
                "and even overhead fluorescent lighting that eliminates harsh "
                "shadows. A single solid red rubber ball, roughly the size of "
                "a tennis ball with a smooth matte surface, is positioned at "
                "the very top of the frame and is released from rest in the "
                "first frame. There are no hands, people, or other objects "
                "in the scene — only the red ball against the white "
                "background. The camera is completely stationary on a tripod, "
                "framing the full vertical drop distance from ceiling to "
                "floor. The ball has an initial velocity of zero and falls "
                "under constant gravitational acceleration of 9.8 metres per "
                "second squared. It immediately begins to fall straight down, "
                "picking up speed as it descends. As the video progresses "
                "the ball accelerates noticeably, moving slowly at first "
                "near the top of the frame and much faster as it approaches "
                "the bottom, tracing a perfectly vertical path through the "
                "centre of the frame. The white background remains static "
                "throughout, providing high contrast against the vivid red "
                "ball. The motion is smooth and continuous, consistent with "
                "real-world gravitational free fall, and the ball remains "
                "sharply in focus for the entire duration of its descent."
            ),
            (
                "A real-world physics demonstration filmed with a stationary "
                "high-speed camera in a well-lit studio with a uniform light "
                "grey backdrop. A glossy red billiard-sized ball is visible at "
                "the top centre of the frame, suspended momentarily before "
                "being released. Once released the ball drops vertically "
                "downward in a straight line, accelerating smoothly under "
                "Earth's gravity. In the first moments the ball drifts slowly "
                "downward, but it gains speed rapidly, falling faster and "
                "faster as it descends toward the bottom of the frame. The "
                "camera does not pan, tilt, or zoom — the framing is fixed "
                "throughout the entire clip. Soft diffused studio lighting "
                "illuminates the ball evenly, creating a subtle highlight on "
                "its upper surface and a small shadow that moves with it. The "
                "backdrop is completely featureless, keeping full visual "
                "attention on the falling ball. The overall scene resembles a "
                "textbook illustration of free-fall motion brought to life."
            ),
            (
                "A clean educational video showing gravitational free fall. "
                "The scene opens in a minimalist white room with bright even "
                "lighting. A single bright red sphere, about the size of a "
                "cricket ball, appears at the top of the frame held in place. "
                "The ball is then released and falls straight down through the "
                "centre of the frame, accelerating under gravity. It starts "
                "slowly and picks up speed continuously as it drops, moving "
                "noticeably faster near the bottom of the frame than at the "
                "top. The camera is locked off on a tripod and does not move "
                "at all. The background is a plain white wall with no "
                "distractions. The red ball stands out sharply against the "
                "white background throughout its descent."
            ),
        ],
    },
    "projectile": {
        "description": "Ball thrown at an angle (parabolic trajectory)",
        "physics": "x = v0*cos(a)*t, y = v0*sin(a)*t - 0.5*g*t^2",
        "prompts": [
            (
                "A bright red ball is thrown from the left side of the frame "
                "at an upward angle. It follows a smooth parabolic arc across "
                "the frame against a plain sky background. The camera is "
                "stationary, capturing the full arc from launch to descent."
            ),
            (
                "A red sphere is launched diagonally upward from the lower "
                "left corner. It rises, reaches a peak, and falls back down "
                "in a symmetric parabolic path. Plain blue sky background, "
                "static camera, no other objects in the scene."
            ),
        ],
    },
    "pendulum": {
        "description": "Ball swinging like a pendulum (periodic motion)",
        "physics": "theta(t) = theta0 * cos(sqrt(g/L) * t)",
        "prompts": [
            (
                "A bright red ball hangs from a thin string attached to the "
                "top of the frame. The ball is released from the right side "
                "and swings back and forth like a pendulum. Plain light "
                "background, stationary camera, smooth periodic motion."
            ),
            (
                "A red sphere suspended by a thin wire swings left and right "
                "in a regular pendulum motion. The camera is fixed, showing "
                "the full swing arc against a uniform white background. The "
                "motion is smooth and periodic."
            ),
        ],
    },
    "ramp": {
        "description": "Solid sphere rolling without slipping down a 45-degree ramp",
        "physics": "s = 0.5 * (5/7) * g * sin(45) * t^2, g = 9.80665 m/s^2, a_eff = 4.95 m/s^2",
        "prompts": [
            (
                "A two-dimensional side-view video of a controlled physics "
                "experiment recorded by a stationary camera mounted rigidly on "
                "a tripod. The camera is positioned perpendicular to the plane "
                "of motion, looking directly at the scene from the side so that "
                "all motion occurs strictly within the flat plane of the image. "
                "There is no perspective distortion, no camera shake, and no "
                "movement of the camera at any time. "
                "The setting is a brightly and evenly lit laboratory with a "
                "uniform matte white wall as the background. The lighting is "
                "diffuse and shadow-soft, with no flicker or brightness "
                "variation throughout the video. "
                "A long, straight, smooth wooden ramp with a light beige surface "
                "is fixed securely at exactly forty-five degrees relative to the "
                "horizontal. The ramp extends diagonally from the upper left "
                "corner of the frame down to the lower right corner. The ramp "
                "remains completely stationary throughout the video. "
                "A small solid red rubber ball, approximately the size of a "
                "tennis ball, is positioned at the very top edge of the ramp. "
                "The ball is a uniform solid sphere with no markings except a "
                "small visible surface texture to make rotation perceptible. "
                "There are no people, hands, tools, supports, or additional "
                "objects in the scene. Only the ramp and the red ball are "
                "present against the white wall. "
                "At time t equals zero seconds, the ball is released from "
                "complete rest with zero initial linear velocity and zero "
                "initial angular velocity. No external push is applied. "
                "The ball rolls down the incline without slipping. Static "
                "friction at the contact point provides the necessary torque to "
                "produce angular acceleration. The ball maintains continuous "
                "contact with the ramp surface at all times. "
                "Gravitational acceleration is 9.80665 metres per second "
                "squared. Because the object is a solid sphere rolling without "
                "slipping, its linear acceleration along the incline is five "
                "sevenths times g times sine of forty-five degrees, which "
                "equals approximately 4.95 metres per second squared. "
                "The motion exhibits uniform linear acceleration down the "
                "slope. The ball starts moving slowly and increases speed "
                "smoothly and continuously as it descends. Its center follows "
                "a perfectly straight diagonal path parallel to the ramp "
                "surface from the upper left toward the lower right. "
                "The camera remains completely fixed. There is no zoom, pan, "
                "tilt, focus shift, or reframing. The lighting, ramp, and "
                "background remain unchanged and motionless for the entire "
                "duration. "
                "The red ball is the only moving object in the frame and "
                "visually contrasts strongly against the light beige ramp and "
                "white background. "
                "By the final frame, the ball reaches the bottom end of the "
                "ramp while still rolling without slipping."
            ),
        ],
    },
}


def load_model(
    cosmos_dir: str,
    model: str = "2b",
    checkpoint_path: str | None = None,
) -> dict:
    """Validate cosmos-predict2.5 installation and return inference config.

    Rather than loading the model into this process, we delegate inference
    to the cosmos-predict2.5 CLI (``torchrun examples/inference.py``).
    This function validates the installation and returns a config dict
    used by :func:`generate_video`.

    Args:
        cosmos_dir: Path to the cloned cosmos-predict2.5 repository.
        model: Model size key (``"2b"`` or ``"14b"``).
        checkpoint_path: Optional path to a LoRA/fine-tuned checkpoint
            (.pt file). If None, the base model is used via HuggingFace.

    Returns:
        Config dict with validated paths for the inference CLI.
    """
    model_cfg = MODEL_CONFIGS[model]

    cosmos_dir = Path(cosmos_dir).resolve()
    inference_script = cosmos_dir / "examples" / "inference.py"

    if not inference_script.exists():
        print(
            f"Error: inference script not found at {inference_script}\n"
            f"Clone the repo and install:\n"
            f"  git clone git@github.com:nvidia-cosmos/cosmos-predict2.5.git\n"
            f"  cd cosmos-predict2.5 && uv sync --extra=cu128",
            file=sys.stderr,
        )
        sys.exit(1)

    venv_python = cosmos_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print(
            f"Error: cosmos-predict2.5 venv not found at {venv_python}\n"
            f"Run: cd {cosmos_dir} && uv sync --extra=cu128",
            file=sys.stderr,
        )
        sys.exit(1)

    config = {
        "cosmos_dir": str(cosmos_dir),
        "inference_script": str(inference_script),
        "venv_python": str(venv_python),
        "model": model,
        "hf_id": model_cfg["hf_id"],
    }
    if checkpoint_path:
        config["checkpoint_path"] = str(Path(checkpoint_path).resolve())

    print(
        f"Cosmos Predict 2.5-{model_cfg['params']} validated at {cosmos_dir} "
        f"(VRAM: {model_cfg['vram_gb']}GB, min GPU: {model_cfg['min_gpu']})"
    )
    return config


def generate_video(
    config: dict,
    prompt: str,
    seed: int,
    output_path: str,
    num_frames: int = 93,
    guidance: float = 7.0,
    nproc: int = 1,
):
    """Generate a single video via the cosmos-predict2.5 inference CLI.

    Writes a temporary JSON input file, calls ``torchrun examples/inference.py``,
    and moves the resulting mp4 to *output_path*.

    Args:
        config: Config dict returned by :func:`load_model`.
        prompt: Text prompt describing the physics scenario.
        seed: Random seed for reproducibility.
        output_path: Path to save the generated .mp4 video.
        num_frames: Number of frames to generate (default 93 = 5.8s at 16fps).
        guidance: Classifier-free guidance scale.
        nproc: Number of GPUs for torchrun (default 1).
    """
    output_path = Path(output_path)
    name = output_path.stem

    input_data = {
        "inference_type": "text2world",
        "name": name,
        "prompt": prompt,
        "seed": seed,
        "guidance": guidance,
        "num_output_frames": num_frames,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(tmpdir, "input.json")
        infer_output_dir = os.path.join(tmpdir, "output")

        with open(input_json, "w") as f:
            json.dump(input_data, f, indent=2)

        cmd = [
            config["venv_python"],
            "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            config["inference_script"],
            "-i", input_json,
            "-o", infer_output_dir,
            "--model", config["hf_id"],
            "--disable-guardrails",
        ]
        if "checkpoint_path" in config:
            cmd.extend(["--checkpoint-path", config["checkpoint_path"]])

        result = subprocess.run(
            cmd,
            cwd=config["cosmos_dir"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Inference stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"cosmos-predict2.5 inference failed (exit {result.returncode})"
            )

        generated = list(Path(infer_output_dir).rglob("*.mp4"))
        if not generated:
            raise RuntimeError(
                f"No mp4 produced. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        os.makedirs(output_path.parent, exist_ok=True)
        shutil.move(str(generated[0]), str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate physics scenario videos with Cosmos Predict 2.5"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=list(SCENARIO_PROMPTS.keys()),
        help="Physics scenario to generate",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=30,
        help="Number of videos to generate per prompt (default: 30)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated videos",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="2b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size: 2b (~25GB VRAM) or 14b (~55-65GB VRAM). Default: 2b",
    )
    parser.add_argument(
        "--cosmos_dir",
        type=str,
        default=os.environ.get("COSMOS_PREDICT_DIR", ""),
        help="Path to cloned cosmos-predict2.5 repo "
        "(or set COSMOS_PREDICT_DIR env var)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a LoRA/fine-tuned .pt checkpoint. "
        "If omitted, the base HuggingFace model is used.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Use only this prompt index (for sensitivity analysis). "
        "If None, generates for all prompts.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=42,
        help="Starting random seed (incremented per run)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (default: 7.0)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs for torchrun (default: 1)",
    )
    args = parser.parse_args()

    if not args.cosmos_dir:
        print(
            "Error: --cosmos_dir is required (or set COSMOS_PREDICT_DIR env var).\n"
            "This should point to your cloned cosmos-predict2.5 repository.",
            file=sys.stderr,
        )
        sys.exit(1)

    scenario = SCENARIO_PROMPTS[args.scenario]
    prompts = scenario["prompts"]

    if args.prompt_index is not None:
        if args.prompt_index >= len(prompts):
            print(
                f"Error: prompt_index {args.prompt_index} out of range "
                f"(scenario has {len(prompts)} prompts)"
            )
            sys.exit(1)
        prompts = [prompts[args.prompt_index]]
        prompt_indices = [args.prompt_index]
    else:
        prompt_indices = list(range(len(prompts)))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment metadata
    model_cfg = MODEL_CONFIGS[args.model]
    metadata = {
        "scenario": args.scenario,
        "description": scenario["description"],
        "physics": scenario["physics"],
        "model": args.model,
        "model_hf_id": model_cfg["hf_id"],
        "n_runs": args.n_runs,
        "seed_start": args.seed_start,
        "cosmos_dir": args.cosmos_dir,
        "checkpoint_path": args.checkpoint_path,
        "guidance": args.guidance,
        "prompts_used": {i: prompts[j] for j, i in enumerate(prompt_indices)},
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Validate cosmos-predict2.5 installation
    config = load_model(
        args.cosmos_dir, model=args.model, checkpoint_path=args.checkpoint_path,
    )

    # Generate videos
    total = len(prompts) * args.n_runs
    count = 0
    for prompt_idx, prompt in zip(prompt_indices, prompts):
        prompt_dir = os.path.join(args.output_dir, f"prompt_{prompt_idx:02d}")
        os.makedirs(prompt_dir, exist_ok=True)

        for run_idx in range(args.n_runs):
            seed = args.seed_start + run_idx
            output_path = os.path.join(prompt_dir, f"run_{run_idx:03d}_seed{seed}.mp4")

            if os.path.exists(output_path):
                print(f"[{count + 1}/{total}] Skipping (exists): {output_path}")
                count += 1
                continue

            print(
                f"[{count + 1}/{total}] Generating: prompt={prompt_idx}, "
                f"run={run_idx}, seed={seed}"
            )
            generate_video(
                config, prompt, seed, output_path,
                guidance=args.guidance, nproc=args.nproc,
            )
            print(f"  Saved: {output_path}")
            count += 1

    print(f"\nDone. Generated {count} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
