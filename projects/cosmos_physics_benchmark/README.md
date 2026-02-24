# Cosmos Physics Benchmark -- Pipeline Runbook

A quantitative benchmark measuring physics understanding in NVIDIA Cosmos Predict 2.5. We generate physics scenarios (free fall, projectile, pendulum, ramp), extract object trajectories with classical CV, and compare them against analytical physics solutions. The key output: *"Cosmos estimates gravity as X.XX +/- Y.YY m/s² (real: 9.81)"*.

## Pipeline Overview

```
Stage 1: GENERATE (GPU)          Stage 2: EXTRACT (CPU)         Stage 3: ANALYZE (CPU)
─────────────────────────        ─────────────────────────      ─────────────────────────
 Text prompts (per scenario)      Generated .mp4 videos          Trajectory CSVs
         │                                │                              │
         ▼                                ▼                              ▼
 Cosmos Predict 2.5-2B            OpenCV HSV threshold +         Curve fit (numpy polyfit)
 (torchrun, A100 80GB)            contour → centroid tracking    + physics ground truth
         │                                │                              │
         ▼                                ▼                              ▼
 N×.mp4 per scenario              CSV per video:                 per_run_metrics.csv
 (93 frames, 16fps, 5.8s)        [frame, t_sec, x_norm, y_norm] aggregate_stats.json
                                                                 trajectory_overlay.png
                                                                 gravity_histogram.png
```

## Environment Setup

### Prerequisites

- **Python 3.10** (required by PyTorch + Cosmos)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **GPU machine** (A100 80GB recommended) for Stage 1 only; Stages 2-3 are CPU-only

### 1. Clone and install cosmos-predict2.5

This is the inference engine. It has its own environment.

```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
uv sync --extra=cu128
```

Accept the [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) and log in:

```bash
uv tool install -U "huggingface_hub[cli]"
hf auth login
```

### 2. Install this project's dependencies

```bash
cd projects/cosmos_physics_benchmark
uv sync                     # installs core deps + gpu group (torch, torchvision)
# uv sync --group notebook  # add Jupyter if running notebooks
```

### 3. Set the cosmos-predict2.5 path

```bash
export COSMOS_PREDICT_DIR=/absolute/path/to/cosmos-predict2.5
```

Or pass `--cosmos_dir` to `generate_scenarios.py` each time.

## Run the Full Pipeline (GPU Machine)

All commands below assume you are in `projects/cosmos_physics_benchmark/`.

### Stage 1: Generate Videos

```bash
uv run python scripts/generate_scenarios.py \
    --scenario free_fall \
    --n_runs 30 \
    --output_dir data/generated_videos/free_fall \
    --cosmos_dir $COSMOS_PREDICT_DIR
```

This produces `data/generated_videos/free_fall/prompt_00/run_000_seed42.mp4` etc. Already-generated videos are skipped (safe to re-run).

**Options:**
- `--prompt_index 0` -- use only the first prompt variant (faster, less variety)
- `--seed_start 100` -- start from a different seed range
- `--guidance 7.0` -- classifier-free guidance scale
- `--nproc 1` -- number of GPUs for torchrun

### Stage 2: Extract Trajectories

```bash
uv run python scripts/extract_trajectory_cv.py \
    --video_dir data/generated_videos/free_fall \
    --output_dir data/trajectories/free_fall
```

Produces one CSV per video with columns `[frame, t_sec, x_norm, y_norm]`.

**Options:**
- `--hsv_lower 0 100 100 --hsv_upper 10 255 255` -- custom HSV range for non-red objects

### Stage 3: Analyze Results

```bash
uv run python scripts/analyze_results.py \
    --trajectory_dir data/trajectories/free_fall \
    --scenario free_fall \
    --output_dir data/results/free_fall
```

## Run Extraction + Analysis Only (CPU Machine)

If you already have generated videos (e.g. copied from a GPU machine), skip Stage 1:

```bash
# Copy videos to data/generated_videos/free_fall/ first, then:
uv sync                      # no GPU needed, torch won't be used

uv run python scripts/extract_trajectory_cv.py \
    --video_dir data/generated_videos/free_fall \
    --output_dir data/trajectories/free_fall

uv run python scripts/analyze_results.py \
    --trajectory_dir data/trajectories/free_fall \
    --scenario free_fall \
    --output_dir data/results/free_fall
```

## What Each Script Does

| Script | Stage | GPU? | Description |
|--------|-------|------|-------------|
| `generate_scenarios.py` | 1 | Yes (A100) | Calls cosmos-predict2.5 inference CLI to generate N videos per scenario. Manages prompts, seeds, and output directories. |
| `extract_trajectory_cv.py` | 2 | No | HSV color threshold + contour detection + centroid tracking. Outputs per-video CSVs. |
| `extract_trajectory_vlm.py` | 2 | Optional | Fallback extractor using Cosmos Reason 2-2B for complex backgrounds where CV fails. |
| `physics_ground_truth.py` | 3 | No | Analytical physics solutions (free fall, projectile, pendulum, ramp). Used as a library and as a CLI to generate reference CSVs. |
| `analyze_results.py` | 3 | No | Curve fitting, error metrics (NRMSE, R², MAE), gravity estimation, aggregate statistics, and plots. |

## Output Files

```
data/
  generated_videos/<scenario>/
    metadata.json                          # experiment config (prompts, seeds, etc.)
    prompt_00/
      run_000_seed42.mp4                   # 93 frames, 16fps, 1280x704
      run_001_seed43.mp4
      ...
  trajectories/<scenario>/
    prompt_00__run_000_seed42.csv          # [frame, t_sec, x_norm, y_norm]
    ...
  results/<scenario>/
    per_run_metrics.csv                    # g_est, NRMSE, R², MAE per video
    aggregate_stats.json                   # mean, std, 95% CI across all runs
    trajectory_overlay.png                 # all trajectories + physics curve
    gravity_histogram.png                  # distribution of estimated g
    error_vs_frame.png                     # per-frame error evolution
```

## Scenarios

| Scenario | Physics | Key Metric |
|----------|---------|------------|
| `free_fall` | y = 0.5 * g * t² | Estimated g vs 9.81 m/s² |
| `projectile` | Parabolic 2D motion | Trajectory shape (R²) |
| `pendulum` | theta(t) = theta₀ * cos(sqrt(g/L) * t) | Period accuracy |
| `ramp` | s = 0.5 * g * sin(angle) * t² | Acceleration along incline |

Start with `free_fall` (the core experiment), then expand to others if results are promising.

## Interactive Notebook

For an end-to-end walkthrough of the free-fall experiment:

```bash
uv sync --group notebook
uv run jupyter notebook notebooks/01_free_fall.ipynb
```
