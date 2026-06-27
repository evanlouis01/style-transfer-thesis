import os
import glob
import csv
import yaml
from PIL import Image, ImageDraw, ImageFont

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

def find_image(directory, prefix):
    pattern = os.path.join(directory, f"{prefix}.*")
    matches = glob.glob(pattern)
    for m in matches:
        if m.lower().endswith((".png", ".jpg", ".jpeg")):
            return m
    return None

def text_to_image(text, width, height, font_size=40, font_path=None, bg_color="white"):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
        
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = f"{line} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > width - 32:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
        
    y = (height - len(lines) * (font_size * 1.2)) // 2
    for l in lines:
        bbox = draw.textbbox((0, 0), l, font=font)
        x = (width - bbox[2]) // 2
        draw.text((x, y), l, fill="black", font=font)
        y += int(font_size * 1.2)
    return img

def main():
    cfg = load_config()
    paths = cfg["paths"]
    
    content_dir = paths["content_dir"]
    style_dir = paths["style_dir"]
    style_prompts = load_style_prompts(paths["style_prompts_csv"])
    
    # We use all 4 models and all 4 variations
    # The models order: SDXL Lightning, SDXL Turbo, Base SDXL, FLUX.1
    # Keys from config: AA, BB, CC, DD
    model_keys = ["AA", "BB", "CC", "DD"]
    model_names = [cfg["models"][mk]["name"].upper() for mk in model_keys]
    
    # Variations mapping for display
    variation_keys = ["controlnet_ip_adapter", "no_adapter", "only_controlnet", "only_ip_adapter"]
    variation_names = ["ControlNet & IP-Adapter", "No Adapter", "ControlNet Only", "IP-Adapter Only"]
    
    # Load sample image to determine native resolution for cell size
    # Fallback to 1024x1024 if not found
    cell_w, cell_h = 1024, 1024
    sample_cp = find_image(content_dir, "content1")
    if sample_cp:
        with Image.open(sample_cp) as img:
            cell_w, cell_h = img.size
            
    header1_h = cell_h // 4
    header2_h = cell_h // 4
    
    num_pairs = 5
    num_cols = 3 + len(model_keys) * len(variation_keys)
    num_rows = 2 + num_pairs
    
    grid_w = num_cols * cell_w
    grid_h = header1_h + header2_h + (num_pairs * cell_h)
    
    print(f"Grid dimensions: {grid_w}x{grid_h} pixels (Cell size: {cell_w}x{cell_h}).")
    
    # The canvas is white by default
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    
    # Calculate dynamic font sizes
    fs_h1 = max(int(cell_w * 0.12), 14)
    fs_h2_col = max(int(cell_w * 0.10), 12)
    fs_h2_var = max(int(cell_w * 0.08), 10)
    fs_prompt = max(int(cell_w * 0.08), 10)
    
    # --- Draw Headers ---
    bg_color_h1 = "#D3D3D3"
    bg_color_h2 = "#EBEBEB"
    
    # Header 1: Empty left area
    empty_h1 = Image.new("RGB", (3 * cell_w, header1_h), bg_color_h1)
    grid.paste(empty_h1, (0, 0))
    
    # Header 1: Models
    for i, mk in enumerate(model_keys):
        start_col = 3 + i * len(variation_keys)
        name_img = text_to_image(model_names[i], cell_w * len(variation_keys), header1_h, font_size=fs_h1, bg_color=bg_color_h1)
        grid.paste(name_img, (start_col * cell_w, 0))
    
    # Header 2: Columns (Prompt, Content, Style)
    headers2_first = ["Prompt", "Content", "Style"]
    for i, h in enumerate(headers2_first):
        grid.paste(text_to_image(h, cell_w, header2_h, font_size=fs_h2_col, bg_color=bg_color_h2), (i * cell_w, header1_h))
        
    # Header 2: Variations
    for i in range(len(model_keys)):
        for j, vname in enumerate(variation_names):
            col_idx = 3 + i * len(variation_keys) + j
            grid.paste(text_to_image(vname, cell_w, header2_h, font_size=fs_h2_var, bg_color=bg_color_h2), (col_idx * cell_w, header1_h))
            
    # --- Draw Separating Lines ---
    draw = ImageDraw.Draw(grid)
    
    # Horizontal line between Model header and Variation header
    draw.line([(3 * cell_w, header1_h), (grid_w, header1_h)], fill="black", width=3)
    # Horizontal line between Headers and Data
    draw.line([(0, header1_h + header2_h), (grid_w, header1_h + header2_h)], fill="black", width=3)
    
    # Vertical line separating Inputs from Results
    draw.line([(3 * cell_w, 0), (3 * cell_w, grid_h)], fill="black", width=5)
    
    # Vertical lines separating Models
    for i in range(1, len(model_keys)):
        x = (3 + i * len(variation_keys)) * cell_w
        draw.line([(x, 0), (x, grid_h)], fill="black", width=5)
        
    # Vertical lines separating Variations
    for i in range(len(model_keys)):
        for j in range(1, len(variation_keys)):
            x = (3 + i * len(variation_keys) + j) * cell_w
            draw.line([(x, header1_h), (x, grid_h)], fill="#808080", width=2)
            
    # --- Draw Data Rows ---
    for row_idx in range(num_pairs):
        pair_num = row_idx + 1
        y = header1_h + header2_h + (row_idx * cell_h)
        
        # 1. Prompt
        prompt = style_prompts.get(f"style{pair_num}", f"Prompt {pair_num}")
        grid.paste(text_to_image(prompt, cell_w, cell_h, font_size=fs_prompt, bg_color="white"), (0, y))
        
        # 2. Content Image
        cp = find_image(content_dir, f"content{pair_num}")
        if cp:
            with Image.open(cp) as img:
                c_img = img.resize((cell_w, cell_h))
                grid.paste(c_img, (cell_w, y))
            
        # 3. Style Image
        sp = find_image(style_dir, f"style{pair_num}")
        if sp:
            with Image.open(sp) as img:
                s_img = img.resize((cell_w, cell_h))
                grid.paste(s_img, (2 * cell_w, y))
            
        # 4. Results
        MODEL_SLUG = {
            "AA": "sdxl-lightning",
            "BB": "sdxl-turbo",
            "CC": "sdxl-base",
            "DD": "flux1-dev",
        }
        for i, mk in enumerate(model_keys):
            slug = MODEL_SLUG.get(mk, mk.lower())
            for j, var in enumerate(variation_keys):
                col_idx = 3 + i * len(variation_keys) + j
                
                # Check benchmark_paper first, then benchmark_results
                paper_res_path = os.path.join(paths["results_dir"], "benchmark_paper", f"{var}_{mk}_content{pair_num}_style{pair_num}.png")
                results_res_path = os.path.join(paths["results_dir"], "benchmark_results", slug, var, f"content{pair_num}_style{pair_num}.png")
                
                res_path = None
                if os.path.exists(paper_res_path):
                    res_path = paper_res_path
                elif os.path.exists(results_res_path):
                    res_path = results_res_path
                
                if res_path:
                    try:
                        with Image.open(res_path) as img:
                            r_img = img.resize((cell_w, cell_h))
                            grid.paste(r_img, (col_idx * cell_w, y))
                    except Exception as e:
                        print(f"Failed to load {res_path}: {e}")
                else:
                    print(f"Missing image: {paper_res_path} and {results_res_path}")

    # --- Save ---
    output_path = os.path.join(paths["results_dir"], "combined_paper_grid.jpg")
    print(f"Saving large grid to {output_path} ... (This may take a minute due to large resolution)")
    
    # Save with PIL limit increased if needed
    Image.MAX_IMAGE_PIXELS = None
    grid.save(output_path, quality=85)
    print("Done!")

if __name__ == "__main__":
    main()
