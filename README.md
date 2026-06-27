# Style Transfer Benchmark

Benchmark suite comparing **four diffusion models (SDXL and FLUX.1)** across **four adapter variations** for style transfer.

## Models

| Alias | Model | Steps | Guidance |
|-------|-------|-------|----------|
| **AA** | SDXL Lightning 4-step | 4 | 0.0 |
| **BB** | SDXL Turbo | 1 | 0.0 |
| **CC** | Base SDXL | 30 | 7.5 |
| **DD** | FLUX.1-dev | 28 | 3.5 |

SDXL models use **ControlNet Canny (SDXL)** and **IP-Adapter (SDXL)**.
FLUX.1-dev uses **InstantX ControlNet Canny (FLUX)** and **InstantX IP-Adapter (FLUX)**.

## Adapter Variations

1. **No adapter** — prompt-only (no ControlNet, no IP-Adapter)
2. **Only IP-Adapter** — style reference via IP-Adapter, no ControlNet
3. **Only ControlNet** — structural guidance via Canny edges, no IP-Adapter
4. **ControlNet + IP-Adapter** — both adapters active

## Project Structure

```
benchmark-1/
├── config.yaml               # All parameters (models, paths, inference settings)
├── benchmark_results.py      # 4 variations × 3 models — full content×style grid
├── benchmark_paper.py        # Paper-style grids (1:1 pairs, model comparison)
├── benchmark_numbers.py      # Metrics placeholder (FID, CLIP, LPIPS, SSIM)
├── benchmark_full.py         # Orchestrator — runs all + timing report
├── preprocess_image.py       # Center-crop images to square
├── requirements.txt          # Python dependencies
├── images/
│   ├── content/              # Content images (content1–5)
│   ├── style/                # Style images (style1–5)
│   └── style_prompts.csv     # Text prompts per style
├── reference/                # Original reference scripts & table layout
└── results/                  # Generated outputs (auto-created)
    ├── benchmark_results/    # Per-model, per-variation grids
    ├── benchmark_paper/      # Paper-style comparison grids
    ├── inference_times.txt   # Timing summary
    └── metrics_placeholder.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess images (optional)

Center-crop all images to square:

```bash
python preprocess_image.py
```

### 3. Run the full benchmark

```bash
python benchmark_full.py
```

This runs all three sub-benchmarks and writes results to `./results/`.

### 4. Run individual benchmarks

```bash
# Full grid (all content × style combinations)
python benchmark_results.py

# Paper-style comparison grids (1:1 pairs)
python benchmark_paper.py

# Metrics placeholder
python benchmark_numbers.py
```

## Configuration

All parameters are in [`config.yaml`](config.yaml):

- **Paths**: content/style directories, CSV, output directory
- **Image settings**: resolution (1024), grid cell size (256)
- **Inference**: seed, prompts, Canny thresholds, adapter scales
- **Models**: base repo, UNet checkpoint, scheduler, steps, guidance

## Output

| File | Description |
|------|-------------|
| `results/benchmark_results/<model>/<variation>/results_grid.jpg` | Content × style grid |
| `results/benchmark_paper/<variation>/paper_grid.jpg` | Paper-style model comparison |
| `results/inference_times.txt` | Average inference time per model per variation |
| `results/metrics_placeholder.txt` | Template for FID/CLIP/LPIPS/SSIM |
