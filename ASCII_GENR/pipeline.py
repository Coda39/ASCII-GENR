import time
import cv2
import numpy as np
from filters import difference_of_gaussians, sobel_edge_detection_shader_style, quantize_edge_direction
from tiling import tile_based_edge_consensus, extract_tile_colors
from template_match import matchTemplates
from renderer import render_ascii_to_image

def create_ascii_art_shader_style(
    image, tile_size=16, sigma=2.0, sigma_scale=1.6, tau=1.0,
    dog_threshold=0.04, edge_threshold=8, exposure=1.0, attenuation=1.0,
    no_edges=False, no_fill=False, debug_mode=None, parallel=True,
    verbose=False, extract_colors=False
):
    timings = {}
    start_total = time.time()

    # 1. Convert image to gray_scale image (luminace)
    t0 = time.time()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray_normalized = gray.astype(np.float32) / 255.0

    # pad image to be divisible by tile_size
    img_height, img_width = gray_normalized.shape # Correct: (rows, cols) -> (height, width)

    # Calculate padding needed to make dimensions divisible
    pad_right = img_width % tile_size
    pad_bottom = img_height % tile_size

    # Pad on the bottom and right, not top and left
    gray_normalized = np.pad(gray_normalized, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)

    timings["luminance_extraction"] = time.time() - t0

    # 2. DoG
    t0 = time.time()
    dog_edges = difference_of_gaussians(gray_normalized, sigma, sigma_scale, tau, dog_threshold)
    timings["difference_of_gaussians"] = time.time() - t0

    # 3. Sobel
    t0 = time.time()
    magnitude, theta, mask = sobel_edge_detection_shader_style(dog_edges)
    timings["sobel_edge_detection"] = time.time() - t0

    # 4. Quantize Directions
    t0 = time.time()
    direction_map = quantize_edge_direction(theta, mask)
    timings["quantize_directions"] = time.time() - t0

    # 5. Tile Consensus
    t0 = time.time()
    tile_direction_map = tile_based_edge_consensus(
        direction_map, tile_size, edge_threshold, parallel
    )
    timings["tile_consensus"] = time.time() - t0

    # 6. Downscale Luminance
    t0 = time.time()
    height, width = gray.shape
    tile_rows = height // tile_size
    tile_cols = width // tile_size
    luminance_downscaled = cv2.resize(gray_normalized, (tile_cols, tile_rows), interpolation=cv2.INTER_AREA)
    timings["luminance_downscale"] = time.time() - t0

    # 7. Generate ASCII
    t0 = time.time()
    tile_directions_sampled = tile_direction_map[::tile_size, ::tile_size][:tile_rows, :tile_cols]

    edge_chars = {0: "|", 1: "-", 2: "/", 3: "\\"}

    adjusted_lum = np.clip(np.abs(luminance_downscaled * exposure) ** attenuation, 0, 1)

    edge_ascii_array = np.full((tile_rows, tile_cols), " ", dtype=object)

    if not no_edges:
        for direction, char in edge_chars.items():
            mask = tile_directions_sampled == direction
            edge_ascii_array[mask] = char

    if not no_fill:
        no_edge_mask = tile_directions_sampled == -1
        matched_ascii_array, template_library = matchTemplates(gray_normalized, no_edge_mask, patch_size=(tile_size, tile_size))

    render_ascii_to_image(edge_ascii_array, template_library, color_array=None, output_path="output/ascii_edge_rendered.png")
    render_ascii_to_image(matched_ascii_array, template_library, color_array=None, output_path="output/ascii_matched_rendered.png")

    ascii_array = np.where(no_edge_mask, matched_ascii_array, edge_ascii_array)

    ascii_array[ascii_array == None] = " "
    timings["ascii_generation"] = time.time() - t0

    # 8. Extract Colors
    color_array = None
    if extract_colors:
        t0 = time.time()
        color_array = extract_tile_colors(image, tile_size)
        timings["color_extraction"] = time.time() - t0

    # Debug Logic
    debug_image = None
    if debug_mode:
        t0 = time.time()
        if debug_mode == "dog":
            debug_image = (dog_edges * 255).astype(np.uint8)
        elif debug_mode == "sobel":
            debug_image = magnitude
        elif debug_mode == "directions":
            debug_image = np.zeros((height, width, 3), dtype=np.uint8)
            dir_map_cropped = direction_map[:height, :width]
            colors = {0: [0,0,255], 1: [0,255,0], 2: [0,255,255], 3: [255,255,0]}
            for k,v in colors.items(): debug_image[dir_map_cropped == k] = v
        elif debug_mode == "tiles":
            tile_h, tile_w = tile_direction_map.shape
            debug_image = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            colors = {0: [255,0,0], 1: [0,255,0], 2: [0,255,255], 3: [255,255,0]}
            for k,v in colors.items(): debug_image[tile_direction_map == k] = v
        timings["debug_visualization"] = time.time() - t0

    timings["total"] = time.time() - start_total
    if verbose:
        print_timings(timings)

    return ascii_array, color_array, debug_image, timings, template_library

def print_timings(timings):
    print("\n=== Performance Timing ===")
    for k, v in timings.items():
        if k != "total": print(f"{k.replace('_', ' ').capitalize()}: {v * 1000:7.2f} ms")
    print(f"{'=' * 26}")
    print(f"TOTAL: {timings['total'] * 1000:7.2f} ms")
