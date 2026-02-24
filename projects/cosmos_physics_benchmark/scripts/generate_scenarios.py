"""
Generate physics scenario videos using Cosmos Predict 2.5-2B.

This script generates N videos for a given physics scenario by prompting
Cosmos Predict 2.5-2B with carefully crafted text prompts. Each run uses
a different random seed to produce statistical variation.

Inference uses the cosmos-predict2.5 CLI (torchrun examples/inference.py)
from the cloned repo. Set --cosmos_dir to point to your local clone.

Usage:
    python generate_scenarios.py \
        --scenario free_fall \
        --n_runs 30 \
        --output_dir ../data/generated_videos/free_fall \
        --cosmos_dir /path/to/cosmos-predict2.5

Hardware: A100 80GB (~25GB VRAM for 2B model)
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
                "A bright red ball is released from the top of the frame and "
                "falls straight down against a plain light blue sky background. "
                "The camera is completely stationary, showing the full vertical "
                "trajectory. The ball accelerates as it falls due to gravity. "
                "Simple, clean scene with no other objects."
            ),
            (
                "A solid red sphere drops from a high position and falls "
                "vertically downward against a clear, uniform sky. Static "
                "camera captures the entire descent. The ball speeds up as it "
                "falls, following a natural gravitational trajectory."
            ),
            (
                "Against a plain white background, a bright red ball is "
                "released from rest at the top of the frame. It falls "
                "straight down, accelerating under gravity. The camera does "
                "not move. The scene is minimal with only the falling ball "
                "visible."
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
        "description": "Ball rolling down an inclined ramp",
        "physics": "s = 0.5 * g * sin(angle) * t^2",
        "prompts": [
            (
                "A bright red ball sits at the top of a smooth inclined ramp "
                "and begins rolling down. The ramp is angled at roughly 30 "
                "degrees. Plain background, stationary camera showing the "
                "full ramp. The ball accelerates as it rolls downward."
            ),
        ],
    },
}


def load_model(cosmos_dir: str, checkpoint_path: str | None = None) -> dict:
    """Validate cosmos-predict2.5 installation and return inference config.

    Rather than loading the model into this process, we delegate inference
    to the cosmos-predict2.5 CLI (``torchrun examples/inference.py``).
    This function validates the installation and returns a config dict
    used by :func:`generate_video`.

    Args:
        cosmos_dir: Path to the cloned cosmos-predict2.5 repository.
        checkpoint_path: Optional path to a LoRA/fine-tuned checkpoint
            (.pt file). If None, the base model is used via HuggingFace.

    Returns:
        Config dict with validated paths for the inference CLI.
    """
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
    }
    if checkpoint_path:
        config["checkpoint_path"] = str(Path(checkpoint_path).resolve())

    print(f"Cosmos Predict 2.5 validated at {cosmos_dir}")
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

    input_data = [
        {
            "inference_type": "text2world",
            "name": name,
            "prompt": prompt,
            "seed": seed,
            "guidance": guidance,
            "num_output_frames": num_frames,
        }
    ]

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
    metadata = {
        "scenario": args.scenario,
        "description": scenario["description"],
        "physics": scenario["physics"],
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
    config = load_model(args.cosmos_dir, checkpoint_path=args.checkpoint_path)

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
