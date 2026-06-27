"""
benchmark_paper.py
==================
Produces paper-style comparison grids matching the layout of result_table.png.

For each adapter variation, one grid is generated comparing AA vs BB vs CC vs DD:
  Columns: Prompt | Content Image | Style Image | AA | BB | CC | DD
  Rows:    content1-style1, content2-style2, ... content5-style5  (1:1 pairs)

Outputs saved to ./results/benchmark_paper/<variation>/paper_grid.jpg
"""

import os
import sys
import glob
import csv
import time
import yaml
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import variables
import gc

# Set HF token from variables before loading diffusers/huggingface_hub
os.environ["HF_TOKEN"] = variables.HF_TOKEN

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxImg2ImgPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.utils import load_image


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_image(directory, prefix):
    pattern = os.path.join(directory, f"{prefix}.*")
    matches = glob.glob(pattern)
    for m in matches:
        if m.lower().endswith((".png", ".jpg", ".jpeg")):
            return m
    return None


def load_style_prompts(csv_path):
    prompts = {}
    if not os.path.exists(csv_path):
        return prompts
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prefix = row["image_name"].split(".")[0]
            prompts[prefix] = row["style_prompt"]
    return prompts


def make_canny(image, low, high):
    arr = np.array(image)
    edges = cv2.Canny(arr, low, high)[:, :, None]
    return Image.fromarray(np.concatenate([edges, edges, edges], axis=2))


def text_to_image(text, width, height, font_size=18):
    """Render text onto a white rectangle."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = f"{line} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > width - 16:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    y = (height - len(lines) * (font_size + 4)) // 2
    for l in lines:
        bbox = draw.textbbox((0, 0), l, font=font)
        x = (width - bbox[2]) // 2
        draw.text((x, y), l, fill="black", font=font)
        y += font_size + 4
    return img


# ── Pipeline builders ────────────────────────────────────────────────────────

SCHEDULERS = {
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
}


def get_param_count(pipe, ip_model=None):
    count = 0
    for name, component in pipe.components.items():
        if hasattr(component, "parameters"):
            count += sum(p.numel() for p in component.parameters())
    if ip_model is not None:
        count += sum(p.numel() for p in ip_model.image_proj_model.parameters())
        count += sum(p.numel() for p in ip_model.image_encoder.parameters())
    return count / 1e9


def build_sdxl_pipelines(model_cfg):
    """Build SDXL ControlNet + Img2Img pipelines."""
    controlnet = ControlNetModel.from_pretrained(
        model_cfg["controlnet_repo"], torch_dtype=torch.float16
    )
    pipe_cn = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_cfg["base_repo"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    if model_cfg.get("unet_repo") and model_cfg.get("unet_ckpt"):
        pipe_cn.unet.load_state_dict(
            load_file(hf_hub_download(model_cfg["unet_repo"], model_cfg["unet_ckpt"]), device="cuda")
        )

    sched_cls = SCHEDULERS[model_cfg["scheduler"]]
    pipe_cn.scheduler = sched_cls.from_config(
        pipe_cn.scheduler.config, **model_cfg.get("scheduler_kwargs", {})
    )

    pipe_base = StableDiffusionXLImg2ImgPipeline(
        vae=pipe_cn.vae,
        text_encoder=pipe_cn.text_encoder,
        text_encoder_2=pipe_cn.text_encoder_2,
        tokenizer=pipe_cn.tokenizer,
        tokenizer_2=pipe_cn.tokenizer_2,
        unet=pipe_cn.unet,
        scheduler=pipe_cn.scheduler,
    )
    return pipe_cn, pipe_base


def build_flux_pipelines(model_cfg):
    """Build FLUX ControlNet + Img2Img pipelines."""
    controlnet = FluxControlNetModel.from_pretrained(
        model_cfg["controlnet_repo"], torch_dtype=torch.bfloat16
    )
    pipe_cn = FluxControlNetPipeline.from_pretrained(
        model_cfg["base_repo"],
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe_img2img = FluxImg2ImgPipeline(
        transformer=pipe_cn.transformer,
        scheduler=pipe_cn.scheduler,
        vae=pipe_cn.vae,
        text_encoder=pipe_cn.text_encoder,
        text_encoder_2=pipe_cn.text_encoder_2,
        tokenizer=pipe_cn.tokenizer,
        tokenizer_2=pipe_cn.tokenizer_2,
    )
    return pipe_cn, pipe_img2img


def build_flux_ip_model(model_cfg, pipe):
    """Build the InstantX FLUX IP-Adapter model."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flux_ipa"))
    from flux_ipa.infer_flux_ipa_siglip import IPAdapter

    ip_ckpt = hf_hub_download(
        model_cfg["ip_adapter_repo"],
        model_cfg["ip_adapter_weight"],
    )
    num_tokens = model_cfg.get("ip_adapter_num_tokens", 128)

    print(f"  Building FLUX IP-Adapter (tokens={num_tokens})...")
    ip_model = IPAdapter(
        pipe, model_cfg["image_encoder_path"], ip_ckpt,
        device="cuda", num_tokens=num_tokens,
    )
    return ip_model


# ── Inference for one (model, variation, pair) ────────────────────────────────

def run_single_inference_sdxl(
    pipe, content_img, style_path, prompt, model_cfg, inf_cfg, use_cn, use_ip, res
):
    """Run a single SDXL inference and return (result_image, elapsed_seconds)."""
    kwargs = {
        "prompt": prompt,
        "negative_prompt": inf_cfg["negative_prompt"],
        "num_inference_steps": model_cfg["num_inference_steps"],
        "guidance_scale": model_cfg["guidance_scale"],
        "generator": torch.Generator().manual_seed(inf_cfg["seed"]),
    }
    if use_cn:
        kwargs["image"] = make_canny(content_img, inf_cfg["canny_low"], inf_cfg["canny_high"])
        kwargs["controlnet_conditioning_scale"] = inf_cfg["controlnet_conditioning_scale"]
    else:
        kwargs["image"] = content_img
        kwargs["strength"] = inf_cfg["strength"]
    if use_ip and style_path:
        kwargs["ip_adapter_image"] = load_image(style_path).resize((res, res))

    t0 = time.time()
    images = pipe(**kwargs).images
    return images[0], time.time() - t0


def run_single_inference_flux(
    pipe_cn, pipe_img2img, ip_model, content_img, style_path,
    prompt, model_cfg, inf_cfg, use_cn, use_ip, res
):
    """Run a single FLUX inference and return (result_image, elapsed_seconds)."""
    control_img = make_canny(content_img, inf_cfg["canny_low"], inf_cfg["canny_high"]) if use_cn else None
    style_img = load_image(style_path).resize((res, res)) if style_path else None

    t0 = time.time()

    if use_ip:
        ip_scale = inf_cfg["ip_adapter_scale"]
        if use_cn:
            # controlnet_ip_adapter
            ip_model.set_scale(ip_scale)
            image_prompt_embeds = ip_model.get_image_embeds(pil_image=style_img)
            images = pipe_cn(
                prompt=prompt,
                control_image=control_img,
                controlnet_conditioning_scale=inf_cfg["controlnet_conditioning_scale"],
                num_inference_steps=model_cfg["num_inference_steps"],
                guidance_scale=model_cfg["guidance_scale"],
                generator=torch.Generator("cuda").manual_seed(inf_cfg["seed"]),
                width=res, height=res,
                joint_attention_kwargs={"image_emb": image_prompt_embeds},
            ).images
        else:
            # only_ip_adapter
            images = ip_model.generate(
                pil_image=style_img,
                prompt=prompt,
                scale=ip_scale,
                num_inference_steps=model_cfg["num_inference_steps"],
                guidance_scale=model_cfg["guidance_scale"],
                seed=inf_cfg["seed"],
                pipe=pipe_img2img,
                image=content_img,
                strength=inf_cfg["strength"],
            )
    else:
        if use_cn:
            # only_controlnet
            kwargs = {
                "prompt": prompt,
                "control_image": control_img,
                "controlnet_conditioning_scale": inf_cfg["controlnet_conditioning_scale"],
                "num_inference_steps": model_cfg["num_inference_steps"],
                "guidance_scale": model_cfg["guidance_scale"],
                "generator": torch.Generator("cuda").manual_seed(inf_cfg["seed"]),
                "width": res, "height": res,
            }
            if ip_model is not None:
                ip_model.set_scale(0.0)
                kwargs["joint_attention_kwargs"] = {"image_emb": ip_model.get_image_embeds(pil_image=content_img)}
            images = pipe_cn(**kwargs).images
        else:
            # no_adapter
            kwargs = {
                "prompt": prompt,
                "image": content_img,
                "strength": inf_cfg["strength"],
                "num_inference_steps": model_cfg["num_inference_steps"],
                "guidance_scale": model_cfg["guidance_scale"],
                "generator": torch.Generator("cuda").manual_seed(inf_cfg["seed"]),
            }
            if ip_model is not None:
                ip_model.set_scale(0.0)
                kwargs["joint_attention_kwargs"] = {"image_emb": ip_model.get_image_embeds(pil_image=content_img)}
            images = pipe_img2img(**kwargs).images

    return images[0], time.time() - t0


# ── Grid Builder ──────────────────────────────────────────────────────────────

def build_paper_grid(
    rows_data, model_keys, model_names, cell_size, output_path
):
    """
    rows_data: list of dicts, one per content-style pair:
      { "prompt": str, "content_img": PIL, "style_img": PIL,
        "results": {model_key: PIL} }
    model_names: list of display names matching model_keys order.
    """
    num_rows = len(rows_data)
    num_cols = 3 + len(model_keys)
    header_h = cell_size // 4

    grid_w = num_cols * cell_size
    grid_h = header_h + num_rows * cell_size
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    # Header row — use display names instead of keys
    headers = ["Prompt", "Content\nImage", "Style\nImage"] + model_names
    for i, h in enumerate(headers):
        grid.paste(
            text_to_image(h, cell_size, header_h, font_size=16),
            (i * cell_size, 0),
        )

    # Data rows
    for r, data in enumerate(rows_data):
        y = header_h + r * cell_size
        # Prompt
        grid.paste(text_to_image(data["prompt"], cell_size, cell_size, font_size=14), (0, y))
        # Content
        if data["content_img"]:
            grid.paste(data["content_img"].resize((cell_size, cell_size)), (cell_size, y))
        # Style
        if data["style_img"]:
            grid.paste(data["style_img"].resize((cell_size, cell_size)), (2 * cell_size, y))
        # Results per model
        for j, mk in enumerate(model_keys):
            if mk in data["results"] and data["results"][mk] is not None:
                grid.paste(data["results"][mk].resize((cell_size, cell_size)), ((3 + j) * cell_size, y))

    grid.save(output_path)
    print(f"  Paper grid saved → {output_path}")


# ── Main Runner ───────────────────────────────────────────────────────────────

def _resolve_image(results_dir, model_key, var, idx):
    """Check caches for an existing result image. Returns path or None.

    Priority:
      1. benchmark_paper/<var>_<model>_content<i>_style<i>.png   (own cache)
      2. benchmark_results/<model>/<var>/content<i>_style<i>.png  (results cache)
    """
    paper_path = os.path.join(
        results_dir, "benchmark_paper", f"{var}_{model_key}_content{idx}_style{idx}.png"
    )
    if os.path.exists(paper_path):
        return paper_path

    results_path = os.path.join(
        results_dir, "benchmark_results", model_key, var, f"content{idx}_style{idx}.png"
    )
    if os.path.exists(results_path):
        return results_path

    return None


def run(config_path="config.yaml", model_filter=None):
    """Run benchmark_paper pipeline. Returns dict of timing data."""
    cfg = load_config(config_path)
    paths = cfg["paths"]
    img_cfg = cfg["image"]
    inf_cfg = cfg["inference"]
    res = img_cfg["resolution"]
    cell = img_cfg["cell_size"]

    content_dir = paths["content_dir"]
    style_dir = paths["style_dir"]
    style_prompts = load_style_prompts(paths["style_prompts_csv"])

    num_pairs = min(
        len([f for f in os.listdir(content_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]),
        len([f for f in os.listdir(style_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]),
    )

    variations = cfg["variations"]
    model_keys = list(cfg["models"].keys())
    all_timings = {}  # {model_key: {variation: [times]}}

    # Storage: collected[var][idx] = {model_key: PIL.Image}
    collected = {var: {} for var in variations}

    # ── Process one model at a time to avoid CUDA OOM ──
    for mk, mcfg in cfg["models"].items():
        if model_filter and mk not in model_filter:
            continue
        all_timings[mk] = {}
        model_type = mcfg.get("type", "sdxl")

        # 1. Identify which (variation, pair) combos still need inference
        missing = []  # list of (var, idx)
        for var in variations:
            for idx in range(1, num_pairs + 1):
                cached = _resolve_image(paths["results_dir"], mk, var, idx)
                if cached:
                    collected[var].setdefault(idx, {})[mk] = load_image(cached)
                    print(f"  [{var}/{mk}] Cached content{idx}_style{idx}")
                else:
                    missing.append((var, idx))

        if not missing:
            print(f"\n  Model {mk} — all pairs cached, skipping pipeline load.")
            continue

        # 2. Load pipeline only if needed
        print(f"\n{'='*60}")
        print(f"MODEL: {mk} — {mcfg['name']}")
        print(f"{'='*60}")

        if model_type == "flux":
            pipe_cn, pipe_img2img = build_flux_pipelines(mcfg)
            ip_model = None  # lazy-loaded
        else:
            pipe_cn, pipe_base = build_sdxl_pipelines(mcfg)
            pipe_img2img = pipe_base

        # Group missing pairs by variation (so IP-Adapter load/unload happens once per var)
        from collections import defaultdict
        missing_by_var = defaultdict(list)
        for var, idx in missing:
            missing_by_var[var].append(idx)

        for var, indices in missing_by_var.items():
            use_cn = "controlnet" in var
            use_ip = "ip_adapter" in var

            if model_type == "flux":
                # Build IP-Adapter if needed (lazy)
                if use_ip and ip_model is None:
                    ip_model = build_flux_ip_model(mcfg, pipe_cn)
            else:
                pipe = pipe_cn if use_cn else pipe_base
                # Load / unload IP-Adapter for SDXL
                if use_ip:
                    print(f"  Loading IP-Adapter for {mk}...")
                    pipe.load_ip_adapter(
                        mcfg["ip_adapter_repo"],
                        subfolder=mcfg["ip_adapter_subfolder"],
                        weight_name=mcfg["ip_adapter_weight"],
                    )
                    pipe.set_ip_adapter_scale(inf_cfg["ip_adapter_scale"])
                else:
                    try:
                        pipe.unload_ip_adapter()
                    except Exception:
                        pass

            out_dir = os.path.join(paths["results_dir"], "benchmark_paper")
            os.makedirs(out_dir, exist_ok=True)
            
            pipe_active = pipe_cn if use_cn else (pipe_base if model_type == "sdxl" else pipe_img2img)
            active_ip_model = ip_model if use_ip and model_type == "flux" else None
            
            torch.cuda.reset_peak_memory_stats()
            params_b = get_param_count(pipe_active, active_ip_model)
            
            if var not in all_timings[mk]:
                all_timings[mk][var] = {"times": [], "peak_vram_gb": 0.0, "params_b": 0.0}

            for idx in indices:
                cp = find_image(content_dir, f"content{idx}")
                sp = find_image(style_dir, f"style{idx}")
                if not cp:
                    continue
                content_img = load_image(cp).resize((res, res))
                prompt = style_prompts.get(f"style{idx}", "")

                print(f"  [{var}/{mk}] Generating content{idx}_style{idx}...")

                if model_type == "flux":
                    result, elapsed = run_single_inference_flux(
                        pipe_cn, pipe_img2img, ip_model,
                        content_img, sp, prompt, mcfg, inf_cfg,
                        use_cn, use_ip, res,
                    )
                else:
                    result, elapsed = run_single_inference_sdxl(
                        pipe, content_img, sp, prompt, mcfg, inf_cfg,
                        use_cn, use_ip, res,
                    )

                out_path = os.path.join(out_dir, f"{var}_{mk}_content{idx}_style{idx}.png")
                result.save(out_path)
                collected[var].setdefault(idx, {})[mk] = result
                all_timings[mk][var]["times"].append(elapsed)
                gc.collect()
                torch.cuda.empty_cache()
            
            all_timings[mk][var]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            all_timings[mk][var]["params_b"] = params_b

        # 3. Free GPU before loading next model
        if model_type == "flux":
            del pipe_cn, pipe_img2img
            if ip_model is not None:
                del ip_model
        else:
            del pipe_cn, pipe_base
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Freed GPU memory for {mk}.")

    # ── Build paper grids (all models done) ──
    out_dir = os.path.join(paths["results_dir"], "benchmark_paper")
    os.makedirs(out_dir, exist_ok=True)
    model_names = [cfg["models"][mk]["name"] for mk in model_keys]

    for var in variations:
        rows_data = []
        for idx in range(1, num_pairs + 1):
            cp = find_image(content_dir, f"content{idx}")
            sp = find_image(style_dir, f"style{idx}")
            if not cp:
                continue
            content_img = load_image(cp).resize((res, res))
            style_img = load_image(sp).resize((res, res)) if sp else None
            prompt = style_prompts.get(f"style{idx}", "")

            rows_data.append({
                "prompt": prompt,
                "content_img": content_img,
                "style_img": style_img,
                "results": collected[var].get(idx, {}),
            })

        grid_path = os.path.join(out_dir, f"paper_grid_{var}.jpg")
        build_paper_grid(rows_data, model_keys, model_names, cell, grid_path)

    return all_timings


if __name__ == "__main__":
    timings = run()
    print("\n" + "=" * 60)
    print("BENCHMARK PAPER — INFERENCE TIMES")
    print("=" * 60)
    for model, vars_data in timings.items():
        for var, metrics in vars_data.items():
            t_list = metrics["times"]
            avg = sum(t_list) / len(t_list) if t_list else 0
            vram = metrics["peak_vram_gb"]
            params = metrics["params_b"]
            print(f"  {model}/{var}: avg={avg:.4f}s ({len(t_list)} samples) | VRAM: {vram:.2f}GB | Params: {params:.2f}B")
