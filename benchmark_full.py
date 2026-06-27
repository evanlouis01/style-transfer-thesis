import os
import yaml
import benchmark_results
import benchmark_paper
import benchmark_numbers


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_timings(t1, t2):
    """Merge two {model: {variation: {"times": [], "peak_vram_gb": 0.0, "params_b": 0.0}}} dicts."""
    merged = {}
    for d in (t1, t2):
        for mk, vars_data in d.items():
            if mk not in merged:
                merged[mk] = {}
            for var, metrics in vars_data.items():
                if var not in merged[mk]:
                    merged[mk][var] = {"times": [], "peak_vram_gb": 0.0, "params_b": 0.0}
                if isinstance(metrics, dict):
                    merged[mk][var]["times"].extend(metrics.get("times", []))
                    merged[mk][var]["peak_vram_gb"] = max(merged[mk][var]["peak_vram_gb"], metrics.get("peak_vram_gb", 0))
                    merged[mk][var]["params_b"] = max(merged[mk][var]["params_b"], metrics.get("params_b", 0))
                else:
                    merged[mk][var]["times"].extend(metrics)
    return merged


def write_timing_report(timings, cfg, output_path):
    """Write a human-readable timing report to a text file."""
    res = cfg["image"]["resolution"]
    lines = []
    lines.append("=" * 100)
    lines.append(f"STYLE TRANSFER BENCHMARK — INFERENCE TIMES (Resolution: {res}x{res})")
    lines.append("=" * 100)
    lines.append("")

    for mk in cfg["models"]:
        model_name = cfg["models"][mk]["name"]
        model_times = timings.get(mk, {})
        lines.append(f"Model: {mk} ({model_name})")
        lines.append("-" * 100)

        all_model_times = []
        for var in cfg["variations"]:
            metrics = model_times.get(var, {})
            t_list = metrics.get("times", []) if isinstance(metrics, dict) else metrics
            
            if t_list:
                all_model_times.extend(t_list)
                avg = sum(t_list) / len(t_list)
                vram = metrics.get("peak_vram_gb", 0) if isinstance(metrics, dict) else 0
                params = metrics.get("params_b", 0) if isinstance(metrics, dict) else 0
                
                lines.append(f"  {var:<25} avg={avg:.4f}s ({len(t_list):>2} inferences) | VRAM: {vram:>5.2f} GB | Params: {params:>5.2f} B")
            else:
                lines.append(f"  {var:<25} N/A (cached or skipped)")

        if all_model_times:
            overall = sum(all_model_times) / len(all_model_times)
            lines.append(f"  {'OVERALL':<25} avg={overall:.4f}s ({len(all_model_times):>2} total)")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nTiming report written → {output_path}")
    print("\n".join(lines))


import argparse

def main():
    cfg = load_config("config.yaml")
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run full benchmark suite.")
    for mk in cfg.get("models", {}):
        parser.add_argument(f"--{mk}", type=str, choices=["yes", "no", "y", "n", "true", "false"], default=None, help=f"Run model {mk}")
    args, unknown = parser.parse_known_args()

    model_filter = []
    has_filter = False
    for mk in cfg.get("models", {}):
        val = getattr(args, mk, None)
        if val is not None:
            has_filter = True
            if val.lower() in ["yes", "y", "true"]:
                model_filter.append(mk)
    if not has_filter:
        model_filter = None

    # 1. Run benchmark_results
    print("\n" + "=" * 70)
    print("PHASE 1: benchmark_results")
    print("=" * 70)
    timings_results = benchmark_results.run("config.yaml", model_filter=model_filter)

    # 2. Run benchmark_paper
    print("\n" + "=" * 70)
    print("PHASE 2: benchmark_paper")
    print("=" * 70)
    timings_paper = benchmark_paper.run("config.yaml", model_filter=model_filter)

    # 3. Run benchmark_numbers (placeholder)
    print("\n" + "=" * 70)
    print("PHASE 3: benchmark_numbers")
    print("=" * 70)
    # benchmark_numbers uses a string for single model filter, but we can run it multiple times if there's a list
    # Update: benchmark_numbers now accepts a list for model_filter so we can run it once!
    benchmark_numbers.run("config.yaml", model_filter=model_filter)

    # 4. Aggregate and report
    # Note: we need to reconstruct the final cfg with only the filtered models for the report
    report_cfg = dict(cfg)
    if model_filter is not None:
        report_cfg["models"] = {k: v for k, v in cfg["models"].items() if k in model_filter}

    all_timings = merge_timings(timings_results, timings_paper)
    timing_path = os.path.join(results_dir, "inference_times.txt")
    write_timing_report(all_timings, report_cfg, timing_path)


if __name__ == "__main__":
    main()
