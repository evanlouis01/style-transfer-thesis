import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

# ── Lazy-loaded heavy modules ─────────────────────────────────────────────────
# We import these inside functions to keep startup fast and allow --help
# to work without GPU libraries installed.


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load the benchmark YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_SLUG = {
    "AA": "sdxl-lightning",
    "BB": "sdxl-turbo",
    "CC": "sdxl-base",
    "DD": "flux1-dev",
}


def find_image(directory: str, prefix: str) -> Optional[str]:
    """Find an image file matching *prefix* in *directory* (any extension)."""
    pattern = os.path.join(directory, f"{prefix}.*")
    for m in glob.glob(pattern):
        if m.lower().endswith((".png", ".jpg", ".jpeg")):
            return m
    return None


# ── Image I/O helpers ─────────────────────────────────────────────────────────

def _load_image_tensor(path: str, size: int = 256) -> torch.Tensor:
    """Load an image as a float32 [1, 3, H, W] tensor normalised to [0, 1]."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LPIPS  (Learned Perceptual Image Patch Similarity)
# ═══════════════════════════════════════════════════════════════════════════════

_lpips_model = None


def _get_lpips_model(device: str = "cpu"):
    """Return a cached LPIPS model (AlexNet backbone)."""
    global _lpips_model
    if _lpips_model is None or str(next(_lpips_model.parameters()).device) != device:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    return _lpips_model


def compute_lpips_score(
    img1_path: str,
    img2_path: str,
    device: str = "cpu",
) -> float:
    """
    Compute LPIPS perceptual distance between two images.
    Lower = more similar.
    """
    model = _get_lpips_model(device)
    t1 = _load_image_tensor(img1_path).to(device) * 2.0 - 1.0
    t2 = _load_image_tensor(img2_path).to(device) * 2.0 - 1.0

    with torch.no_grad():
        score = model(t1, t2)
    return score.item()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DINOv2 Distance (Semantic / Style similarity)
# ═══════════════════════════════════════════════════════════════════════════════

_dinov2_processor = None
_dinov2_model = None

def _get_dinov2(device: str = "cpu"):
    global _dinov2_processor, _dinov2_model
    if _dinov2_model is None or _dinov2_model.device.type != device:
        from transformers import AutoImageProcessor, AutoModel
        _dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    return _dinov2_processor, _dinov2_model

def compute_dinov2_score(img1_path: str, img2_path: str, device: str = "cpu") -> float:
    """
    Compute Cosine Distance between DINOv2 CLS tokens of two images.
    Lower = more similar.
    """
    processor, model = _get_dinov2(device)
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    inputs1 = processor(images=img1, return_tensors="pt").to(device)
    inputs2 = processor(images=img2, return_tensors="pt").to(device)
    
    with torch.no_grad():
        feat1 = model(**inputs1).last_hidden_state[:, 0, :]  # CLS token
        feat2 = model(**inputs2).last_hidden_state[:, 0, :]
    
    # Cosine similarity -> distance (1 - sim)
    sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    return 1.0 - sim


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KID (Kernel Inception Distance via clean-fid)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kid(real_dir: str, gen_dir: str, batch_size: int = 8) -> float:
    """Compute KID using clean-fid. Lower = closer distributions."""
    from cleanfid import fid as cleanfid
    score = cleanfid.compute_kid(real_dir, gen_dir, batch_size=batch_size, num_workers=0)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PSNR & SSIM
# ═══════════════════════════════════════════════════════════════════════════════

_psnr_metric = None
_ssim_metric = None

def compute_psnr_score(img1_path: str, img2_path: str, device: str = "cpu") -> float:
    """Compute PSNR. Higher = better content preservation."""
    global _psnr_metric
    if _psnr_metric is None or _psnr_metric.device.type != device:
        from torchmetrics.image import PeakSignalNoiseRatio
        _psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    t1 = _load_image_tensor(img1_path).to(device)
    t2 = _load_image_tensor(img2_path).to(device)
    with torch.no_grad():
        score = _psnr_metric(t1, t2)
    return score.item()

def compute_ssim_score(img1_path: str, img2_path: str, device: str = "cpu") -> float:
    """Compute SSIM. Higher = better structural similarity."""
    global _ssim_metric
    if _ssim_metric is None or _ssim_metric.device.type != device:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        _ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    t1 = _load_image_tensor(img1_path).to(device)
    t2 = _load_image_tensor(img2_path).to(device)
    with torch.no_grad():
        score = _ssim_metric(t1, t2)
    return score.item()


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_NAMES = ["lpips", "kid", "dinov2", "psnr", "ssim"]


def _collect_image_pairs(
    cfg: dict,
    model_filter: Optional[str] = None,
    variation_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Build a list of dicts describing every (model, variation, content, style)
    combination that has a generated image on disk.
    """
    paths = cfg["paths"]
    content_dir = paths["content_dir"]
    style_dir = paths["style_dir"]
    results_dir = paths["results_dir"]

    content_files = sorted(
        f for f in os.listdir(content_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    style_files = sorted(
        f for f in os.listdir(style_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    num_content = len(content_files)
    num_style = len(style_files)

    models = cfg["models"]
    variations = cfg["variations"]

    pairs: List[Dict] = []
    for model_key in models:
        if model_filter:
            if isinstance(model_filter, str) and model_key != model_filter:
                continue
            if isinstance(model_filter, (list, set, tuple)) and model_key not in model_filter:
                continue
        slug = MODEL_SLUG.get(model_key, model_key.lower())

        for var in variations:
            if variation_filter and var != variation_filter:
                continue

            gen_dir = os.path.join(results_dir, "benchmark_results", slug, var)
            if not os.path.isdir(gen_dir):
                gen_dir = os.path.join(results_dir, "benchmark_results", model_key, var)
            if not os.path.isdir(gen_dir):
                print(f"  [SKIP] No results directory: {gen_dir}")
                continue

            for c in range(1, num_content + 1):
                content_path = find_image(content_dir, f"content{c}")
                if content_path is None:
                    continue
                for s in range(1, num_style + 1):
                    gen_path = os.path.join(gen_dir, f"content{c}_style{s}.png")
                    if not os.path.exists(gen_path):
                        continue
                    style_path = find_image(style_dir, f"style{s}")
                    if style_path is None:
                        continue
                    pairs.append({
                        "model": model_key,
                        "variation": var,
                        "content_idx": c,
                        "style_idx": s,
                        "content_path": content_path,
                        "style_path": style_path,
                        "generated_path": gen_path,
                        "gen_dir": gen_dir,
                    })
    return pairs


def run(
    config_path: str = "config.yaml",
    device: str = "cpu",
    metric_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    variation_filter: Optional[str] = None,
) -> None:
    """
    Run benchmark metrics and write results to disk.
    """
    cfg = load_config(config_path)
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    pairs = _collect_image_pairs(cfg, model_filter, variation_filter)
    if not pairs:
        print("No generated images found. Run benchmark_results.py first.")
        return

    print(f"Found {len(pairs)} image pairs to evaluate.\n")

    # ── Determine which metrics to compute ────────────────────────────────
    do_all = metric_filter is None
    do_lpips = do_all or metric_filter == "lpips"
    do_kid = do_all or metric_filter == "kid"
    do_dinov2 = do_all or metric_filter == "dinov2"
    do_psnr = do_all or metric_filter == "psnr"
    do_ssim = do_all or metric_filter == "ssim"
    # ── Per-pair metrics ──────────────────────────────────────────────────
    csv_rows: List[Dict] = []

    for i, pair in enumerate(pairs):
        tag = (
            f"[{pair['model']}/{pair['variation']}] "
            f"content{pair['content_idx']}_style{pair['style_idx']}"
        )
        print(f"  ({i + 1}/{len(pairs)}) {tag}")

        row = {
            "model": pair["model"],
            "variation": pair["variation"],
            "content": f"content{pair['content_idx']}",
            "style": f"style{pair['style_idx']}",
            "lpips": "",
            "dinov2": "",
            "psnr": "",
            "ssim": "",
        }

        if do_lpips:
            row["lpips"] = f"{compute_lpips_score(pair['content_path'], pair['generated_path'], device):.6f}"

        if do_dinov2:
            row["dinov2"] = f"{compute_dinov2_score(pair['style_path'], pair['generated_path'], device):.6f}"

        if do_psnr:
            row["psnr"] = f"{compute_psnr_score(pair['content_path'], pair['generated_path'], device):.6f}"

        if do_ssim:
            row["ssim"] = f"{compute_ssim_score(pair['content_path'], pair['generated_path'], device):.6f}"

        csv_rows.append(row)

    # ── KID (per model × variation, needs directories) ────────────────────
    kid_scores: Dict[Tuple[str, str], float] = {}
    if do_kid:
        print("\nComputing KID scores (per model × variation)...")
        content_dir = cfg["paths"]["content_dir"]
        seen = set()
        for pair in pairs:
            key = (pair["model"], pair["variation"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  KID: {key[0]}/{key[1]}")
            try:
                score = compute_kid(content_dir, pair["gen_dir"])
                kid_scores[key] = score
                print(f"       → {score:.6f}")
            except Exception as e:
                print(f"       → ERROR: {e}")
                kid_scores[key] = float("nan")

    # ── Write CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "metrics.csv")
    fieldnames = ["model", "variation", "content", "style", "lpips", "dinov2", "psnr", "ssim"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nPer-pair metrics written → {csv_path}")

    # ── Build summary report ──────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STYLE TRANSFER BENCHMARK — METRICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Group rows by (model, variation)
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in csv_rows:
        groups[(row["model"], row["variation"])].append(row)

    for model_key in cfg["models"]:
        if model_filter:
            if isinstance(model_filter, str) and model_key != model_filter:
                continue
            if isinstance(model_filter, (list, set, tuple)) and model_key not in model_filter:
                continue
        model_name = cfg["models"][model_key]["name"]
        report_lines.append(f"Model: {model_key} ({model_name})")
        report_lines.append("-" * 70)
        header = (
            f"  {'Variation':<25}"
            f"  {'LPIPS':>10}"
            f"  {'DINOv2':>10}"
            f"  {'PSNR':>10}"
            f"  {'SSIM':>10}"
            f"  {'KID':>10}"
        )
        report_lines.append(header)
        report_lines.append(
            f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}"
        )

        for var in cfg["variations"]:
            if variation_filter and var != variation_filter:
                continue
            rows = groups.get((model_key, var), [])
            if not rows:
                report_lines.append(f"  {var:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
                continue

            def _avg(key):
                vals = [float(r[key]) for r in rows if r.get(key)]
                return sum(vals) / len(vals) if vals else None

            avg_l = _avg("lpips")
            avg_d = _avg("dinov2")
            avg_p = _avg("psnr")
            avg_s = _avg("ssim")
            kid_val = kid_scores.get((model_key, var))

            parts = [f"  {var:<25}"]
            parts.append(f"  {avg_l:10.6f}" if avg_l is not None else f"  {'N/A':>10}")
            parts.append(f"  {avg_d:10.6f}" if avg_d is not None else f"  {'N/A':>10}")
            parts.append(f"  {avg_p:10.6f}" if avg_p is not None else f"  {'N/A':>10}")
            parts.append(f"  {avg_s:10.6f}" if avg_s is not None else f"  {'N/A':>10}")
            parts.append(f"  {kid_val:10.4f}" if kid_val is not None else f"  {'N/A':>10}")
            report_lines.append("".join(parts))

        report_lines.append("")

    report_path = os.path.join(results_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))



# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute style-transfer benchmark metrics from pre-generated images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_numbers.py                            # all metrics, all models
  python benchmark_numbers.py --model AA                 # only SDXL Lightning
  python benchmark_numbers.py --variation no_adapter     # only one variation
  python benchmark_numbers.py --metric kid               # only KID
  python benchmark_numbers.py --device cuda              # use GPU
        """,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation (default: cpu)",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        default=None,
        help="Compute only this metric",
    )
    parser.add_argument(
        "--model",
        default=None,
        nargs="+",
        help="Filter by model key(s) (e.g. AA BB CC)",
    )
    parser.add_argument(
        "--variation",
        default=None,
        help="Filter by variation (e.g. no_adapter, only_ip_adapter)",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        device=args.device,
        metric_filter=args.metric,
        model_filter=args.model,
        variation_filter=args.variation,
    )


if __name__ == "__main__":
    main()
