# Cosmos Models Overview

## 1. The Cosmos Ecosystem

**NVIDIA Cosmos** is a platform of state-of-the-art generative World Foundation Models (WFMs) for Physical AI. A WFM is a digital twin of the physical world -- a model that generates the future state of the world based on past observations and current perturbations (actions, text prompts, control signals). Cosmos positions WFMs as a core tool for Physical AI builders, enabling policy evaluation, policy initialization, synthetic data generation, and model-predictive control -- all in silicon, before deploying to real hardware.

The platform is organized around five pillars: **Cosmos Curator** (data processing), **Cosmos Predict** (world generation), **Cosmos Transfer** (controlled video translation), **Cosmos Reason** (vision-language understanding), and **Cosmos RL** (distributed training infrastructure). Together with **Guardrails** (Pre-Guard + Post-Guard), these components cover the complete Physical AI lifecycle from data curation through inference.

**v1 (January 2025)** introduced the initial platform: a custom video tokenizer, diffusion-based and autoregressive WFMs (Predict 1), a 7B ControlNet (Transfer 1), and the Curator pipeline processing 20M hours of video. **v2 (October 2025)** brought Cosmos Predict 2.5 and Transfer 2.5, switching to flow-matching, replacing the custom tokenizer with WAN2.1 VAE, using Cosmos Reason 1 as the text encoder, and scaling the curation pipeline to 35M hours (200M curated clips). The v2 models achieve comparable quality to models 2-4x larger while enabling unified Text2World, Image2World, and Video2World in a single architecture.

---

## 2. Cosmos Tokenizer (v1)

The Cosmos Tokenizer is a family of visual tokenizers that compress images and videos into compact latent representations for efficient WFM training and inference.

### Architecture

- **Encoder-decoder** design operating in **wavelet space** (3D Haar wavelet transform as the first stage)
- **Temporally causal**: token computation for current frames does not depend on future frames, making it compatible with Physical AI's causal setting and enabling joint image-video training
- Post-wavelet stages use residual blocks with spatio-temporal factorized 3D convolutions and causal self-attention
- Supports both **continuous** tokens (vanilla autoencoder, latent dim 16) and **discrete** tokens (Finite-Scalar-Quantization / FSQ with levels (8,8,8,5,5,5), vocab size 64,000, latent dim 6)
- Trained on multiple aspect ratios (1:1, 3:4, 4:3, 9:16, 16:9) and variable-length videos

### Compression Rates

| Variant | Type | Compression (T x H x W) | Used By |
|---------|------|--------------------------|---------|
| CV4x8x8 | Continuous Video | 4 x 8 x 8 | -- |
| **CV8x8x8-720p** | Continuous Video | 8 x 8 x 8 | **Predict 1 Diffusion** |
| CV8x16x16 | Continuous Video | 8 x 16 x 16 | -- |
| DV4x8x8 | Discrete Video | 4 x 8 x 8 | -- |
| DV8x8x8 | Discrete Video | 8 x 8 x 8 | -- |
| **DV8x16x16-720p** | Discrete Video | 8 x 16 x 16 | **Predict 1 Autoregressive** |

### Performance

- SOTA on DAVIS and TokenBench benchmarks for both continuous and discrete variants
- 2x-12x faster than prior tokenizers (e.g., 62.7ms vs 242ms for FLUX-Tokenizer at 8x8 image compression)
- Runs up to 8s at 1080p and 10s at 720p on a single A100 80GB GPU

### v2 Note

Cosmos Predict 2.5 **replaces** the custom Cosmos Tokenizer with the **WAN2.1 VAE**, a causal variational autoencoder with **4 x 8 x 8** compression. This provides a different quality-compression tradeoff optimized for the flow-matching architecture. A 1x2x2 patchification is applied on top, yielding 24 latent frames for 93 pixel frames at 16fps (~5.8s videos).

---

## 3. Cosmos Predict

Cosmos Predict is the core world generation model family. It takes text prompts, images, or videos as input and generates future video frames.

### 3.1 Predict 1 -- Diffusion (v1)

A latent diffusion model operating in the continuous token space of Cosmos-Tokenize1-CV8x8x8-720p.

**Architecture:**
- **DiT** (Diffusion Transformer) backbone with self-attention, cross-attention, and feed-forward layers
- **EDM** (Elucidated Diffusion Model) formulation with uncertainty-based weighting
- **T5-XXL** text encoder for cross-attention conditioning
- **3D RoPE** (FPS-aware) + learnable absolute positional embedding
- **AdaLN-LoRA**: adaptive layer normalization with low-rank projections (36% parameter reduction for 7B)

**Models:**

| Model | Layers | Dim | Params | Mode |
|-------|--------|-----|--------|------|
| Cosmos-Predict1-7B-Text2World | 28 | 4,096 | 7B | Text -> Video |
| Cosmos-Predict1-7B-Video2World | 28 | 4,096 | 7B | Text + Video -> Video |
| Cosmos-Predict1-14B-Text2World | 36 | 5,120 | 14B | Text -> Video |
| Cosmos-Predict1-14B-Video2World | 36 | 5,120 | 14B | Text + Video -> Video |

**Output:** 720p (1280x704), 121 frames. Includes a prompt upsampler (Cosmos-UpsamplePrompt1-12B-Text2World, fine-tuned Mistral-NeMo-12B).

**Training:** Progressive (512p 57f -> 720p 121f -> high-quality fine-tune), joint image-video, trained on 10,000 H100 GPUs over 3 months.

### 3.2 Predict 1 -- Autoregressive (v1)

A GPT-style next-token prediction model operating on discrete FSQ tokens from Cosmos-Tokenize1-DV8x16x16-720p.

**Architecture:**
- Llama3-style transformer decoder, trained from scratch for video prediction
- T5-XXL cross-attention for text conditioning (added in Stage 2)
- 3D RoPE + 3D absolute positional embedding (sinusoidal)
- QK-Normalization + z-loss for training stability
- Medusa speculative decoding heads for 2x-3x inference speedup

**Models:**

| Model | Layers | Dim | Params | Mode |
|-------|--------|-----|--------|------|
| Cosmos-Predict1-4B | 16 | 4,096 | 4B | Video -> Video |
| Cosmos-Predict1-5B-Video2World | 16 | 4,096 | 5B | Text + Video -> Video |
| Cosmos-Predict1-12B | 40 | 5,120 | 12B | Video -> Video |
| Cosmos-Predict1-13B-Video2World | 40 | 5,120 | 13B | Text + Video -> Video |

**Output:** 640x1024 resolution, 33 frames. Includes a **diffusion decoder** (Cosmos-Predict1-7B-Decoder) to upscale discrete tokens from DV8x16x16 to CV8x8x8 space for higher visual quality.

**Training:** Multi-stage (17f -> 34f -> text-conditioned 34f), with a cooling-down phase on high-quality data.

### 3.3 Predict 2.5 -- Flow-Matching DiT (v2)

The latest generation, unifying Text2World, Image2World, and Video2World in a single model.

**Key changes from Predict 1:**
- **Flow matching** replaces EDM (velocity prediction instead of denoising score matching)
- **WAN2.1 VAE** (4x8x8) replaces Cosmos Tokenizer
- **Cosmos Reason 1** replaces T5-XXL as text encoder (multi-layer feature concatenation -> 1024-dim projection, richer text grounding)
- Absolute positional embeddings removed; **3D RoPE only** (better resolution/length generalization)
- Unified architecture handles T2W, I2W, V2W via frame-replacement conditioning

**Models:**

| Model | Layers | Dim | Heads | Params |
|-------|--------|-----|-------|--------|
| Cosmos-Predict2.5-2B | 32 | 2,048 | 16 | 2B |
| Cosmos-Predict2.5-14B | 36 | 5,120 | 40 | 14B |

**Output:** 720p (1280x704), **93 frames at 16fps** (~5.8 seconds). 24 latent frames after VAE + patchification.

**Training pipeline:**
1. **Progressive pre-training**: Text2Image (256p) -> T2I+V2W (256p, 93f) -> (480p, 93f) -> (720p, 93f) -> T2I+V2W+T2W (720p)
2. **Domain-specific SFT**: separate fine-tuning on 5 domains (object permanence, high motion, complex scenes, driving, robotic manipulation) + 4K cooldown
3. **Model merging**: model soup across SFT checkpoints to combine domain strengths
4. **GRPO reinforcement learning**: using VideoAlign reward model (text alignment, motion quality, visual quality)
5. **4-step timestep distillation**: rCM (hybrid forward-reverse consistency + distribution matching), producing near-teacher quality in 4 steps

**Specialized variants:**

| Variant | Capability | Input |
|---------|-----------|-------|
| Cosmos-Predict2.5-2B/auto/multiview | 7-camera driving view | text + image or video |
| Cosmos-Predict2.5-2B/robot/action-cond | Action-conditioned robotics | action + image |
| Cosmos-Predict2.5-2B/robot/multiview-agibot | AgiBot 3-camera views | text + image |
| Cosmos-Predict2.5-14B/robot/gr00tdream-gr1 | GR00T GR1 humanoid | text + image or video |

**Benchmarks:** On PAI-Bench, Predict2.5-2B post-trained achieves 0.768 (T2W) and 0.810 (I2W) overall scores, comparable to Wan2.2-5B and Wan2.1-14B despite being 60-86% smaller.

---

## 4. Cosmos Transfer

Cosmos Transfer provides control-net style conditioning for guided video generation -- translating simulation outputs, control signals, or spatial conditions into photorealistic video.

### 4.1 Transfer 1 (v1)

- **7B ControlNet** built on top of Cosmos-Predict1-7B diffusion backbone
- **7+ control modalities**: depth, segmentation, edge, blur, HDMap, LiDAR, and more
- **MultiControlNet**: combine multiple control signals simultaneously
- **4K upscaler** for super-resolution
- 4 control blocks inserted sequentially at the start of the main branch
- Used for Sim2Real conversion, weather augmentation, warehouse simulation, and robotics

### 4.2 Transfer 2.5 (v2)

- Built on **Cosmos-Predict2.5-2B** backbone (3.5x smaller than Transfer 1's 7B base)
- **4 primary control modalities**: edge, blur, segmentation, depth
- Control blocks distributed **evenly** (1 per 7 DiT blocks) instead of front-loaded
- Each control branch trained independently for 100K iterations

**Performance:** Outperforms Transfer1-7B on PAIBench-Transfer across all metrics despite being 3.5x smaller. Significantly less error accumulation in long video generation (measured by RNDS -- Relative Normalized Dover Score).

**Specialized variants:**

| Variant | Capability | Control Input |
|---------|-----------|---------------|
| Cosmos-Transfer2.5-2B/general | ControlNet | edge, blur, segmentation, depth |
| Cosmos-Transfer2.5-2B/auto/multiview | Driving, multiview ControlNet | world scenario map |
| Cosmos-Transfer2.5-2B/robot/multiview | Robotic, 3-camera view | text + third-person video |
| Cosmos-Transfer2.5-2B/robot/multiview-agibot | Robotic, AgiBot 3-camera | text + head-view video |

---

## 5. Cosmos Reason

Cosmos Reason is a vision-language model (VLM) family designed for physically grounded reasoning about video content.

### 5.1 Reason 1 (v1)

- **7B VLM** for spatial and temporal understanding
- Chain-of-thought reasoning capabilities
- SFT-based training on Physical AI reasoning tasks
- Supports image and video inputs
- **Dual role in v2**: serves as both a standalone reasoning model AND the text encoder for Cosmos Predict 2.5 (replacing T5-XXL)

**Cookbook recipes:** spatial AI warehouse understanding, AV video captioning/VQA, temporal localization, wafer map classification, intelligent transportation, physical plausibility prediction.

### 5.2 Reason 2

- **2B and 8B** parameter variants
- Training: **SFT + reinforcement learning** with **VideoAlign** reward model
- Structured output capabilities (JSON extraction for per-frame analysis)
- Enhanced spatial/temporal grounding over Reason 1

**Cookbook recipes:** worker safety detection, video search & summarization, egocentric social reasoning, 3D AV grounding, physical plausibility prediction v2, intelligent transportation v2.

---

## 6. Cosmos Curator

A GPU-accelerated video data curation pipeline built on AnyScale **Ray** for processing massive video datasets.

### v1 Pipeline (5 stages)

1. **Splitting**: shot boundary detection (TransNetV2) + GPU-based transcoding (h264_nvenc)
2. **Filtering**: motion filtering, visual quality (DOVER), overlay text detection, video type classification
3. **Annotation**: VLM-based captioning (VILA 13B, FP8-quantized TensorRT-LLM, 1.96 clips/s at batch 16)
4. **Deduplication**: semantic dedup via InternVideo2 embeddings + GPU k-means (k=10,000)
5. **Sharding**: by resolution, aspect ratio, and length for curriculum training

**Scale:** 20M hours raw video -> ~100M pre-training clips + ~10M fine-tuning clips. 30% removed by dedup.

### v2 Pipeline (7 stages, expanded)

1. Shot-aware video splitting
2. GPU-based transcoding
3. Video cropping (remove black borders)
4. **Multi-stage filtering**: aesthetic scoring -> motion filter -> OCR detection -> perceptual quality (DOVER) -> semantic artifacts (VTSS) -> VLM filter -> content type classifier
5. Video captioning (Qwen2.5-VL-7B, multi-length: short/medium/long)
6. Semantic deduplication (online, incremental)
7. Sharding (26-type taxonomy, multi-axis: content type, resolution, aspect ratio, length)

**Scale:** 35M hours raw video -> 6B+ segments -> **200M curated clips** (4% survival rate vs 30% in v1). Petabyte-scale infrastructure with dynamic CPU/GPU auto-scaling.

---

## 7. Cosmos RL

A distributed training framework supporting the complete post-training pipeline.

**Capabilities:**
- **Supervised fine-tuning (SFT)** with domain-specific data
- **Reinforcement learning** with GRPO (Group Relative Policy Optimization)
- **Elastic rollout service**: dynamically scaled reward computation with VideoAlign and other reward models
- **FP8/FP4 precision** support for efficient training
- **FSDP2** (per-parameter sharding) as primary distributed training framework
- **Flexible context parallelism** (Ulysses-style) for long video sequences
- **Selective activation checkpointing** (SAC) for memory optimization
- Asynchronous reward computation via CUDA IPC, Redis-backed result store

**Training efficiency (v2):** On 4,096 H100 GPUs at 720p/93 frames: 36.49% MFU for 2B model, 33.08% MFU for 14B model.

---

## 8. Guardrails

A two-stage safety system for protecting WFM deployment:

- **Pre-Guard**: blocks harmful inputs before they reach the model (filters prompts, images, and video inputs)
- **Post-Guard**: blocks harmful outputs after generation (filters generated videos)

Both guards work together to ensure safe usage of Cosmos WFMs in production Physical AI applications.

---

## 9. Quick Reference Table

| Model Family | Latest Version | Params | Architecture | Input | Output | Resolution | Frames |
|---|---|---|---|---|---|---|---|
| **Predict (Diffusion)** | 2.5 | 2B / 14B | Flow-matching DiT | text + image/video | video | 720p | 93 @ 16fps |
| **Predict (AR)** | 1 | 4B-13B | GPT decoder + FSQ | text + video | video | 640x1024 | 33 |
| **Transfer** | 2.5 | ~2B | ControlNet on Predict2.5 | control signals + text | video | 720p | 93 @ 16fps |
| **Reason** | 2 | 2B / 8B | VLM | image/video + text | text/JSON | -- | -- |
| **Tokenizer** | 1 (v2 uses WAN VAE) | 77-105M | Encoder-decoder (wavelet) | video/image | tokens | up to 1080p | up to 121 |
| **Curator** | 2 | -- | Ray pipeline (7 stages) | raw video | curated dataset | -- | -- |
| **RL** | -- | -- | Distributed framework | -- | trained model | -- | -- |

---

## 10. v1 -> v2 Evolution Table

| Component | v1 (Jan 2025) | v2 (Oct 2025) | Key Change |
|---|---|---|---|
| **Paper** | "Cosmos World Foundation Model Platform for Physical AI" | "World Simulation with Video Foundation Models for Physical AI" | Expanded scope |
| **Predict architecture** | EDM diffusion OR autoregressive (separate families) | Flow-matching DiT (unified T2W/I2W/V2W) | Single unified model |
| **Predict sizes** | 7B/14B (diffusion), 4B-13B (AR) | 2B/14B | Smaller, more efficient |
| **Tokenizer** | Custom Cosmos Tokenizer (CV8x8x8 / DV8x16x16) | WAN2.1 VAE (4x8x8) | External VAE, simpler |
| **Text encoder** | T5-XXL | Cosmos Reason 1 (multi-layer features) | Richer text grounding |
| **Positional encoding** | 3D RoPE + learnable absolute | 3D RoPE only | Better generalization |
| **Post-training** | SFT only | Domain SFT -> model merging -> GRPO RL -> distillation | Full RL pipeline |
| **Transfer** | 7B ControlNet, 7+ modalities | ~2B ControlNet (on Predict2.5-2B), 4 modalities | 3.5x smaller, better quality |
| **Reason** | Reason 1 (7B, SFT only) | Reason 2 (2B/8B, SFT + RL) | RL-aligned, smaller options |
| **Curator scale** | 20M hrs raw -> ~100M clips (30% dedup removal) | 35M hrs raw -> 200M clips (4% survival rate) | 10x stricter filtering |
| **Curator pipeline** | 5 stages | 7 stages (+ cropping, multi-stage filtering) | VLM filter, content classifier |
| **Training infra** | FSDP + CP (ring attention) | FSDP2 + CP (Ulysses) + SAC | Per-parameter sharding |
| **Output specs** | 720p, 121 frames (diffusion) / 33 frames (AR) | 720p, 93 frames @ 16fps (~5.8s) | Standardized output |
| **Multiview** | Single-view camera control | 7-camera driving, 3-camera robotics | Native multi-view |
| **Action conditioning** | Action-conditioned via cross-attention or timestamp | Action embedder MLP + timestamp embedding | Improved action integration |
| **Distillation** | Not available | 4-step rCM (near-teacher quality) | Fast inference |

---

## Further Reading

- **v1 paper**: `papers/cosmos_v1.pdf` -- Full details on Tokenizer (S4), Data Curation (S3), Diffusion WFM (S5.1), Autoregressive WFM (S5.2), Post-training (S6), Guardrails (S7)
- **v2 paper**: `papers/cosmos_v2.pdf` -- Predict 2.5 architecture (S3), Training pipeline (S4), Benchmarks (S5), Transfer 2.5 and applications (S6)
- **Cookbook recipes**: See the [Recipes](recipes/) section for step-by-step workflows
- **GitHub repositories**: [cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), [cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1), [cosmos-curate](https://github.com/nvidia-cosmos/cosmos-curate), [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl)
