import os
import glob
from PIL import Image

def center_crop_to_square(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if it isn't (e.g. RGBA png)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            if width == height:
                return # Already a square
            
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(image_path)
            print(f"Cropped {os.path.basename(image_path)} from {width}x{height} to {min_dim}x{min_dim}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    benchmark_dir = "images"
    subdirs = ["content", "style"]
    
    for subdir in subdirs:
        dir_path = os.path.join(benchmark_dir, subdir)
        if not os.path.isdir(dir_path):
            print(f"Directory {dir_path} not found.")
            continue
            
        print(f"Processing images in {dir_path}...")
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                center_crop_to_square(file_path)

if __name__ == "__main__":
    main()
