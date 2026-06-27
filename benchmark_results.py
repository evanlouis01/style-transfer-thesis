"""
benchmark_results.py
====================
Runs 4 adapter variations × 4 models (AA, BB, CC, DD) and produces result grids.

Grid layout per (model, variation): 
  - Row 0 header = style images (or text prompts when IP-Adapter is off)
  - Col 0 header = content images
  - Cells = generated results (content_i × style_j)

All outputs are saved to ./results/benchmark_results/<model>/<variation>/
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
    """Finds an image with the given prefix, ignoring extension."""
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
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def text_to_image(text, size, font_size=20):
    """Render text onto a white square image."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    # word-wrap
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = f"{line} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > size - 20:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    y = (size - len(lines) * (font_size + 4)) // 2
    for l in lines:
        bbox = draw.textbbox((0, 0), l, font=font)
        x = (size - bbox[2]) // 2
        draw.text((x, y), l, fill="black", font=font)
        y += font_size + 4
    return img


# ── Pipeline Builders ─────────────────────────────────────────────────────────

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
    """Build ControlNet + img2img pipelines for an SDXL model config."""
    print(f"  Loading ControlNet from {model_cfg['controlnet_repo']}...")
    controlnet = ControlNetModel.from_pretrained(
        model_cfg["controlnet_repo"], torch_dtype=torch.float16
    )

    print(f"  Loading base pipeline from {model_cfg['base_repo']}...")
    pipe_cn = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_cfg["base_repo"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Swap UNet if needed (e.g. Lightning)
    if model_cfg.get("unet_repo") and model_cfg.get("unet_ckpt"):
        print(f"  Swapping UNet from {model_cfg['unet_repo']}...")
        pipe_cn.unet.load_state_dict(
            load_file(hf_hub_download(model_cfg["unet_repo"], model_cfg["unet_ckpt"]), device="cuda")
        )

    # Scheduler
    sched_cls = SCHEDULERS[model_cfg["scheduler"]]
    sched_kwargs = model_cfg.get("scheduler_kwargs", {})
    pipe_cn.scheduler = sched_cls.from_config(pipe_cn.scheduler.config, **sched_kwargs)

    # Base (img2img) pipeline sharing components
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
    """Build FLUX ControlNet pipeline and Img2Img pipeline.

    Returns (pipe_cn, pipe_img2img, ip_model_loader).
    ip_model_loader is a callable that returns an IPAdapter instance
    (deferred so IP-Adapter is only loaded when needed).
    """
    print(f"  Loading FLUX ControlNet from {model_cfg['controlnet_repo']}...")
    controlnet = FluxControlNetModel.from_pretrained(
        model_cfg["controlnet_repo"], torch_dtype=torch.bfloat16
    )

    print(f"  Loading FLUX base pipeline from {model_cfg['base_repo']}...")
    pipe_cn = FluxControlNetPipeline.from_pretrained(
        model_cfg["base_repo"],
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Img2Img pipeline (for no_adapter / only_ip_adapter without ControlNet)
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
    """Build the InstantX FLUX IP-Adapter model.

    Uses the custom InstantX files from the flux_ipa/ directory.
    The IP-Adapter modifies the transformer's attention processors
    in-place, so this must be called with the pipeline that will be used.
    """
    # Add flux_ipa to path so InstantX files can find each other
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flux_ipa"))
    from flux_ipa.infer_flux_ipa_siglip import IPAdapter
    from flux_ipa.transformer_flux import FluxTransformer2DModel as FluxTransformerIPA

    image_encoder_path = model_cfg["image_encoder_path"]
    num_tokens = model_cfg.get("ip_adapter_num_tokens", 128)

    # Download IP-Adapter weights
    ip_ckpt = hf_hub_download(
        model_cfg["ip_adapter_repo"],
        model_cfg["ip_adapter_weight"],
    )

    print(f"  Building FLUX IP-Adapter (tokens={num_tokens})...")
    ip_model = IPAdapter(
        pipe, image_encoder_path, ip_ckpt,
        device="cuda", num_tokens=num_tokens,
    )
    return ip_model


# ── Grid Builder ──────────────────────────────────────────────────────────────

def create_results_grid(
    content_dir, style_dir, results_dir, style_prompts,
    num_content, num_style, cell_size, use_ip_adapter, output_path
):
    cols = num_style + 1
    rows = num_content + 1
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")

    # Top row: style images or text prompts
    for s in range(1, num_style + 1):
        if use_ip_adapter:
            sp = find_image(style_dir, f"style{s}")
            if sp:
                grid.paste(load_image(sp).resize((cell_size, cell_size)), (s * cell_size, 0))
        else:
            prompt = style_prompts.get(f"style{s}", f"style{s}")
            grid.paste(text_to_image(prompt, cell_size), (s * cell_size, 0))

    # Left column + cells
    for c in range(1, num_content + 1):
        cp = find_image(content_dir, f"content{c}")
        if cp:
            grid.paste(load_image(cp).resize((cell_size, cell_size)), (0, c * cell_size))
        for s in range(1, num_style + 1):
            rp = os.path.join(results_dir, f"content{c}_style{s}.png")
            if os.path.exists(rp):
                grid.paste(load_image(rp).resize((cell_size, cell_size)), (s * cell_size, c * cell_size))

    grid.save(output_path)
    print(f"  Grid saved → {output_path}")


# ── SDXL Inference Runner ─────────────────────────────────────────────────────

def run_sdxl_model(model_key, model_cfg, cfg, content_dir, style_dir,
                   style_prompts, num_content, num_style, res, cell):
    """Run all 4 variations for an SDXL model. Returns {variation: [times]}."""
    paths = cfg["paths"]
    inf_cfg = cfg["inference"]
    variations = cfg["variations"]

    pipe_cn, pipe_base = build_sdxl_pipelines(model_cfg)
    timings = {}

    for var in variations:
        print(f"\n  --- Variation: {var} ---")
        use_cn = "controlnet" in var
        use_ip = "ip_adapter" in var
        pipe = pipe_cn if use_cn else pipe_base

        # Load / unload IP-Adapter
        if use_ip:
            print("  Loading IP-Adapter...")
            pipe.load_ip_adapter(
                model_cfg["ip_adapter_repo"],
                subfolder=model_cfg["ip_adapter_subfolder"],
                weight_name=model_cfg["ip_adapter_weight"],
            )
            pipe.set_ip_adapter_scale(inf_cfg["ip_adapter_scale"])
        else:
            try:
                pipe.unload_ip_adapter()
            except Exception:
                pass

        out_dir = os.path.join(paths["results_dir"], "benchmark_results", model_key, var)
        os.makedirs(out_dir, exist_ok=True)
        times = []

        torch.cuda.reset_peak_memory_stats()
        params_b = get_param_count(pipe)

        for c in range(1, num_content + 1):
            cp = find_image(content_dir, f"content{c}")
            if not cp:
                continue
            content_img = load_image(cp).resize((res, res))
            control_img = make_canny(content_img, inf_cfg["canny_low"], inf_cfg["canny_high"]) if use_cn else None

            for s in range(1, num_style + 1):
                out_path = os.path.join(out_dir, f"content{c}_style{s}.png")
                if os.path.exists(out_path):
                    print(f"    [{var}] Exists, skipping content{c}_style{s}")
                    continue

                prompt = style_prompts.get(f"style{s}", "")
                kwargs = {
                    "prompt": prompt,
                    "negative_prompt": inf_cfg["negative_prompt"],
                    "num_inference_steps": model_cfg["num_inference_steps"],
                    "guidance_scale": model_cfg["guidance_scale"],
                    "generator": torch.Generator().manual_seed(inf_cfg["seed"]),
                }
                if use_cn:
                    kwargs["image"] = control_img
                    kwargs["controlnet_conditioning_scale"] = inf_cfg["controlnet_conditioning_scale"]
                else:
                    kwargs["image"] = content_img
                    kwargs["strength"] = inf_cfg["strength"]
                if use_ip:
                    sp = find_image(style_dir, f"style{s}")
                    if sp:
                        kwargs["ip_adapter_image"] = load_image(sp).resize((res, res))

                print(f"    [{var}] Generating content{c}_style{s}...")
                t0 = time.time()
                images = pipe(**kwargs).images
                t1 = time.time()
                times.append(t1 - t0)
                images[0].save(out_path)
                gc.collect()
                torch.cuda.empty_cache()

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        timings[var] = {
            "times": times,
            "peak_vram_gb": peak_vram_gb,
            "params_b": params_b
        }

        # Build grid
        grid_path = os.path.join(out_dir, "results_grid.jpg")
        create_results_grid(
            content_dir, style_dir, out_dir, style_prompts,
            num_content, num_style, cell, use_ip, grid_path,
        )

    # Free GPU memory
    del pipe_cn, pipe_base
    gc.collect()
    torch.cuda.empty_cache()

    return timings


# ── FLUX Inference Runner ─────────────────────────────────────────────────────

def run_flux_model(model_key, model_cfg, cfg, content_dir, style_dir,
                   style_prompts, num_content, num_style, res, cell):
    """Run all 4 variations for a FLUX model. Returns {variation: [times]}.

    FLUX architecture differences:
    - Uses FluxControlNetPipeline + FluxImg2ImgPipeline instead of SDXL variants
    - IP-Adapter uses InstantX custom IPAdapter class (not diffusers native)
    - Uses bfloat16 instead of float16
    - Text-to-image pipeline (no negative_prompt support)
    """
    paths = cfg["paths"]
    inf_cfg = cfg["inference"]
    variations = cfg["variations"]

    pipe_cn, pipe_img2img = build_flux_pipelines(model_cfg)
    timings = {}

    # Track whether IP-Adapter has been loaded (modifies transformer in-place)
    ip_model = None

    for var in variations:
        print(f"\n  --- Variation: {var} ---")
        use_cn = "controlnet" in var
        use_ip = "ip_adapter" in var

        # Build IP-Adapter model if needed (lazy, one-time)
        if use_ip and ip_model is None:
            # IP-Adapter needs the ControlNet pipeline's transformer
            ip_model = build_flux_ip_model(model_cfg, pipe_cn)

        out_dir = os.path.join(paths["results_dir"], "benchmark_results", model_key, var)
        os.makedirs(out_dir, exist_ok=True)
        times = []

        pipe_active = pipe_cn if use_cn else pipe_img2img
        active_ip_model = ip_model if use_ip else None
        
        torch.cuda.reset_peak_memory_stats()
        params_b = get_param_count(pipe_active, active_ip_model)

        for c in range(1, num_content + 1):
            cp = find_image(content_dir, f"content{c}")
            if not cp:
                continue
            content_img = load_image(cp).resize((res, res))
            control_img = make_canny(content_img, inf_cfg["canny_low"], inf_cfg["canny_high"]) if use_cn else None

            for s in range(1, num_style + 1):
                out_path = os.path.join(out_dir, f"content{c}_style{s}.png")
                if os.path.exists(out_path):
                    print(f"    [{var}] Exists, skipping content{c}_style{s}")
                    continue

                prompt = style_prompts.get(f"style{s}", "")

                print(f"    [{var}] Generating content{c}_style{s}...")
                t0 = time.time()

                if use_ip:
                    # ── IP-Adapter path (uses InstantX ip_model.generate) ──
                    sp = find_image(style_dir, f"style{s}")
                    style_img = load_image(sp).resize((res, res)) if sp else None

                    ip_scale = inf_cfg["ip_adapter_scale"]

                    if use_cn:
                        # controlnet_ip_adapter: ControlNet + IP-Adapter
                        # IP-Adapter scale controls style influence
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
                        # only_ip_adapter: Text-to-image with style reference
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
                        # only_controlnet: ControlNet only, no IP-Adapter
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
                        # no_adapter: Img2Img with prompt only
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

                t1 = time.time()
                times.append(t1 - t0)
                images[0].save(out_path)
                gc.collect()
                torch.cuda.empty_cache()

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        timings[var] = {
            "times": times,
            "peak_vram_gb": peak_vram_gb,
            "params_b": params_b
        }

        # Build grid
        grid_path = os.path.join(out_dir, "results_grid.jpg")
        create_results_grid(
            content_dir, style_dir, out_dir, style_prompts,
            num_content, num_style, cell, use_ip, grid_path,
        )

    # Free GPU memory
    del pipe_cn, pipe_img2img, ip_model
    gc.collect()
    torch.cuda.empty_cache()

    return timings


# ── Main Runner ───────────────────────────────────────────────────────────────

def run(config_path="config.yaml", model_filter=None):
    """Run benchmark_results pipeline. Returns dict of timing data."""
    cfg = load_config(config_path)
    paths = cfg["paths"]
    img_cfg = cfg["image"]

    content_dir = paths["content_dir"]
    style_dir = paths["style_dir"]
    style_prompts = load_style_prompts(paths["style_prompts_csv"])

    num_content = len([f for f in os.listdir(content_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    num_style = len([f for f in os.listdir(style_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    res = img_cfg["resolution"]
    cell = img_cfg["cell_size"]

    all_timings = {}  # {model_key: {variation: [times]}}

    for model_key, model_cfg in cfg["models"].items():
        if model_filter and model_key not in model_filter:
            continue
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key} — {model_cfg['name']}")
        print(f"{'='*60}")

        model_type = model_cfg.get("type", "sdxl")

        if model_type == "flux":
            timings = run_flux_model(
                model_key, model_cfg, cfg, content_dir, style_dir,
                style_prompts, num_content, num_style, res, cell,
            )
        else:
            timings = run_sdxl_model(
                model_key, model_cfg, cfg, content_dir, style_dir,
                style_prompts, num_content, num_style, res, cell,
            )

        all_timings[model_key] = timings

    return all_timings


if __name__ == "__main__":
    timings = run()
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — INFERENCE TIMES")
    print("=" * 60)
    for model, vars_data in timings.items():
        for var, metrics in vars_data.items():
            t_list = metrics["times"]
            avg = sum(t_list) / len(t_list) if t_list else 0
            vram = metrics["peak_vram_gb"]
            params = metrics["params_b"]
            print(f"  {model}/{var}: avg={avg:.4f}s ({len(t_list)} samples) | VRAM: {vram:.2f}GB | Params: {params:.2f}B")
