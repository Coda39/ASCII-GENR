import argparse
import cv2
import time
from multiprocessing import cpu_count
from pipeline import create_ascii_art_shader_style
from renderer import print_ascii_art, render_ascii_to_image, ansi_color_code, ansi_reset

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate ASCII art")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--color", action="store_true", help="Enable colored ASCII output")
    parser.add_argument("--tile-size", type=int, default=8, help="ASCII character tile size")
    parser.add_argument("--edge-threshold", type=int, default=12, help="Min edge pixels for detection")
    parser.add_argument("--no-debug", action="store_true", help="Skip debug visualizations")
    parser.add_argument("--render", action="store_true", help="Generate rasterized image output")
    parser.add_argument("--font-size", type=int, default=20, help="Font size for rendered image")
    args = parser.parse_args()

    # Load input image
    original_image = cv2.imread(args.image_path)
    if original_image is None:
        print(f"Error: Could not load image at {args.image_path}")
        return

    # Downscale logic
    max_dim = 1920
    h, w = original_image.shape[:2]
    if w > max_dim or h > max_dim:
        scale = min(max_dim / w, max_dim / h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Downscaling to {new_w}x{new_h}")
        original_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"Processing on {cpu_count()} cores...")

    # Run Pipeline
    ascii_array, color_array, _, _, template_library = create_ascii_art_shader_style(
        original_image,
        tile_size=args.tile_size,
        edge_threshold=args.edge_threshold,
        parallel=True,
        verbose=True,
        extract_colors=args.color,
        debug_mode=None if args.no_debug else "tiles" # Example default debug
    )

    # Render Image
    if args.render:
        render_ascii_to_image(ascii_array, template_library, color_array)

    # Save Debug Images
    if not args.no_debug:
        modes = ["dog", "sobel", "directions", "tiles"]
        for mode in modes:
            _, _, dbg, _, _ = create_ascii_art_shader_style(
                original_image, tile_size=args.tile_size, edge_threshold=args.edge_threshold,
                debug_mode=mode, parallel=True
            )
            if dbg is not None:
                cv2.imwrite(f"output/debug_{mode}.png", dbg)
                print(f"Saved debug_{mode}.png")

if __name__ == "__main__":
    main()
