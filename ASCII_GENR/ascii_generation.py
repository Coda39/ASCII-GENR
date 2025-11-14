import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial


def pixilate_image(image_path, block_size=8, contrast_factor=1.2):
    """
    Args:
        image_path: Path to image
        block_size: Downscaling factor
        contrast_factor: Enhancement factor (unused currently)
    """
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    original_height, original_width = original_image.shape[:2]

    downscaled_width = original_width // block_size
    downscaled_height = original_height // block_size

    downscaled_image = cv2.resize(
        original_image,
        (downscaled_width, downscaled_height),
        interpolation=cv2.INTER_AREA,
    )

    downscaled_bgr = downscaled_image.astype(np.float32)

    # Use correct luminance formula (matching shader: 0.299*R + 0.587*G + 0.114*B)
    # OpenCV uses BGR, so: 0.114*B + 0.587*G + 0.299*R
    luminance = (
        0.114 * downscaled_bgr[:, :, 0]
        + 0.587 * downscaled_bgr[:, :, 1]
        + 0.299 * downscaled_bgr[:, :, 2]
    )

    enhanced_luminance = cv2.equalizeHist(luminance.astype(np.uint8))

    # Return the downscaled grayscale image (don't upscale back to original!)
    # This is what gets converted to ASCII characters
    return original_image, downscaled_image, enhanced_luminance


def brightness_to_ascii(brightness_array, ascii_chars):
    max_index = len(ascii_chars) - 1
    # Normalize brightness (0-255) to character indices
    char_indices = ((brightness_array / 255.0) * max_index).astype(int)
    char_indices = np.clip(char_indices, 0, max_index)

    # Vectorized mapping: use fancy indexing instead of nested loops
    ascii_array = np.array(
        [[ascii_chars[idx] for idx in row] for row in char_indices], dtype=object
    )

    return ascii_array


def print_ascii_art(ascii_array):
    """Print ASCII art directly to terminal"""
    height, width = ascii_array.shape
    print(f"\nASCII Art ({height}x{width} characters):\n")

    for row in range(height):
        print("".join(ascii_array[row, :]))

    print(f"\n")


def difference_of_gaussians(
    image, sigma=2.0, sigma_scale=1.6, tau=1.0, threshold=0.005
):
    """
    Apply Difference of Gaussians (DoG) filter.
    This implements the exact same algorithm.

    Args:
        image: Grayscale input image (normalized 0-1)
        sigma: Base standard deviation for first Gaussian
        sigma_scale: Scale factor (k) for second Gaussian
        tau: Weight for second Gaussian in subtraction
        threshold: Minimum DoG value to keep as edge

    Returns:
        Binary edge map (0 or 1)
    """
    # Ensure image is grayscale and normalized to 0-1
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Apply two Gaussian blurs with different sigmas (matching shader)
    sigma2 = sigma * sigma_scale
    gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma)
    gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)

    # Compute DoG: D = G(σ) - τ × G(σk)
    dog = gaussian1 - tau * gaussian2

    # Threshold: binary edge detection (matching shader Pass 4 line 130-132)
    edges = (dog >= threshold).astype(np.uint8)

    return edges


def sobel_edge_detection_shader_style(image):
    """
    Apply Sobel edge detection matching the Unity shader's separable approach.
    Implements Pass 5-6 from ascii.shader with proper angle computation.

    Args:
        image: Grayscale input image (0-255 or 0-1)

    Returns:
        magnitude: Edge strength at each pixel (0-255)
        theta: Gradient direction in radians (-π to π, matching shader)
        mask: Valid edge mask (1 where edges exist, 0 otherwise)
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize to 0-1 if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Calculate gradients using Sobel operators (matching shader kernels)
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize magnitude to 0-255
    magnitude_normalized = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Calculate angle: arctan2(Gy, Gx) in radians [-π, π]
    theta = np.arctan2(gradient_y, gradient_x)

    # --- BUG WAS HERE ---
    # The old mask only checked for NaN. It was TRUE for (0,0) gradients.
    # mask = (~np.isnan(theta)).astype(np.uint8)

    # --- FIX IS HERE ---
    # A pixel is only a valid edge if its magnitude is non-zero.
    # This correctly creates a mask that is 0 for all flat, non-edge areas.
    mask = ((magnitude > 0) & (~np.isnan(theta))).astype(np.uint8)

    return magnitude_normalized, theta, mask


def quantize_edge_direction(theta, mask):
    """
    Quantize edge angles to 4 discrete directions, strictly respecting the mask.

    Args:
        theta: Angle array in radians [-π, π] (can contain NaNs for non-edges)
        mask: Valid edge mask (1 where edges exist, 0 otherwise)

    Returns:
        direction: Array of direction indices (0-3 for edges, -1 for no edge)
    """
    height, width = theta.shape
    direction_map = np.full(
        (height, width), -1, dtype=np.int8
    )  # Renamed to avoid confusion with the return value

    # Identify pixels where Sobel detected a valid edge (mask > 0)
    # This is the ONLY set of pixels we should consider for angle quantization
    valid_edge_mask = mask > 0

    if not np.any(valid_edge_mask):
        return direction_map  # No valid edges, return map of all -1

    # Extract theta values only for valid edge pixels
    valid_thetas = theta[valid_edge_mask]

    # Calculate absolute normalized theta ONLY for valid edge pixels
    abs_theta_normalized = np.abs(valid_thetas) / np.pi

    # Create masks for each direction for the *subset* of valid pixels
    # Direction 0: VERTICAL (θ ≈ 0° or 180°)
    vertical_cond = ((abs_theta_normalized >= 0.0) & (abs_theta_normalized < 0.05)) | (
        (abs_theta_normalized > 0.9) & (abs_theta_normalized <= 1.0)
    )

    # Direction 1: HORIZONTAL (θ ≈ 90°)
    horizontal_cond = (abs_theta_normalized > 0.45) & (abs_theta_normalized < 0.55)

    # Direction 2/3: DIAGONALS
    lower_diag_cond = (abs_theta_normalized > 0.05) & (abs_theta_normalized < 0.45)
    upper_diag_cond = (abs_theta_normalized > 0.55) & (abs_theta_normalized < 0.9)

    # Now, assign directions to the *original* direction_map using the combined mask
    # This is where we need to be careful to map the conditions for `valid_thetas` back to `direction_map`

    # Create a temporary array to hold quantized directions for valid pixels
    quantized_valid_directions = np.full(valid_thetas.shape, -1, dtype=np.int8)

    quantized_valid_directions[vertical_cond] = 0
    quantized_valid_directions[horizontal_cond] = 1

    # Diagonals
    quantized_valid_directions[lower_diag_cond & (valid_thetas > 0)] = 2
    quantized_valid_directions[lower_diag_cond & (valid_thetas <= 0)] = 3
    quantized_valid_directions[upper_diag_cond & (valid_thetas > 0)] = 3
    quantized_valid_directions[upper_diag_cond & (valid_thetas <= 0)] = 2

    # Place the quantized directions back into the full direction_map using the original mask
    direction_map[valid_edge_mask] = quantized_valid_directions

    return direction_map


def _process_tile(args):
    """
    Process a single tile for edge consensus voting.
    Helper function for parallel processing.

    Args:
        args: Tuple of (tr, tc, tile_data, edge_threshold)

    Returns:
        Tuple of (tr, tc, consensus_direction)
    """
    tr, tc, tile_data, edge_threshold = args

    # Count directions (matching shader lines 48-50)
    buckets = [0, 0, 0, 0]
    for direction in range(4):
        buckets[direction] = np.sum(tile_data == direction)

    # Find most common direction (matching shader lines 55-60)
    max_value = 0
    common_edge_index = -1
    for j in range(4):
        if buckets[j] > max_value:
            common_edge_index = j
            max_value = buckets[j]

    # Discard if not enough edge pixels (matching shader line 63)
    if max_value < edge_threshold:
        common_edge_index = -1

    return (tr, tc, common_edge_index)


def tile_based_edge_consensus(
    direction_map, tile_size=8, edge_threshold=8, parallel=True
):
    """
    Apply tile-based voting for dominant edge direction.
    This implements the groupshared voting mechanism from ASCII.compute lines 44-66.

    For each 8×8 tile:
    1. Count edges in each of 4 directions
    2. Select most common direction
    3. Discard if count < threshold

    Args:
        direction_map: Per-pixel direction map (-1 or 0-3)
        tile_size: Size of voting tiles (default 8×8)
        edge_threshold: Minimum votes needed for consensus
        parallel: Use multiprocessing for tile processing

    Returns:
        tile_direction_map: Direction per tile (upsampled to original size)
    """
    height, width = direction_map.shape

    # Calculate tile dimensions
    tile_rows = height // tile_size
    tile_cols = width // tile_size

    # Result: one direction per tile
    tile_directions = np.full((tile_rows, tile_cols), -1, dtype=np.int8)

    if parallel and tile_rows * tile_cols > 100:  # Only parallelize if enough tiles
        # Prepare tile data for parallel processing
        tile_args = []
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                tile = direction_map[
                    tr * tile_size : (tr + 1) * tile_size,
                    tc * tile_size : (tc + 1) * tile_size,
                ]
                tile_args.append((tr, tc, tile, edge_threshold))

        # Process tiles in parallel
        num_processes = min(cpu_count(), 8)  # Cap at 8 processes
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_tile, tile_args)

        # Fill in results
        for tr, tc, consensus in results:
            tile_directions[tr, tc] = consensus
    else:
        # Serial processing (original implementation)
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                # Extract tile
                tile = direction_map[
                    tr * tile_size : (tr + 1) * tile_size,
                    tc * tile_size : (tc + 1) * tile_size,
                ]

                # Count directions (matching shader lines 48-50)
                buckets = [0, 0, 0, 0]
                for direction in range(4):
                    buckets[direction] = np.sum(tile == direction)

                # Find most common direction (matching shader lines 55-60)
                max_value = 0
                common_edge_index = -1
                for j in range(4):
                    if buckets[j] > max_value:
                        common_edge_index = j
                        max_value = buckets[j]

                # Discard if not enough edge pixels (matching shader line 63)
                if max_value < edge_threshold:
                    common_edge_index = -1

                tile_directions[tr, tc] = common_edge_index

    # Upsample tile directions back to full resolution
    # Each pixel gets the direction of its containing tile
    tile_direction_map = np.repeat(
        np.repeat(tile_directions, tile_size, axis=0), tile_size, axis=1
    )

    # Crop to original size in case of dimension mismatch
    tile_direction_map = tile_direction_map[:height, :width]

    return tile_direction_map


def select_ascii_character_shader_style(
    tile_direction,
    luminance,
    exposure=1.0,
    attenuation=1.0,
    no_edges=False,
    no_fill=False,
):
    """
    Select ASCII character based on edge direction or luminance.
    Implements the character selection logic from ASCII.compute lines 82-102.

    Args:
        tile_direction: Consensus edge direction for this tile (-1 or 0-3)
        luminance: Brightness value (0-1)
        exposure: Brightness multiplier
        attenuation: Contrast adjustment (power/gamma)
        no_edges: Disable edge-based characters
        no_fill: Disable luminance-based fill

    Returns:
        ASCII character
    """
    # Edge characters by direction (matching shader edge texture lookup)
    edge_chars = {
        0: "|",  # Vertical
        1: "-",  # Horizontal
        2: "/",  # Diagonal /
        3: "\\",  # Diagonal \
    }

    # Luminance characters (10 levels, matching shader)
    # Ordered from darkest to lightest
    luminance_chars = " .*:o&8?@█"

    # If we have a valid edge direction and edges aren't disabled
    if tile_direction >= 0 and not no_edges:
        return edge_chars[tile_direction]

    # Otherwise use luminance-based character (if fill isn't disabled)
    elif not no_fill:
        # Apply exposure and attenuation (matching shader lines 93-95)
        adjusted_lum = np.clip(np.abs(luminance * exposure) ** attenuation, 0, 1)

        # Quantize to 10 levels (matching shader line 95)
        # luminance = max(0, (floor(luminance * 10) - 1)) / 10.0f
        level = int(max(0, np.floor(adjusted_lum * 10) - 1))
        level = min(level, len(luminance_chars) - 1)

        return luminance_chars[level]

    else:
        return " "


def create_ascii_art_shader_style(
    image,
    tile_size=16,
    sigma=2.0,
    sigma_scale=1.6,
    tau=1.0,
    dog_threshold=0.005,
    edge_threshold=8,
    exposure=1.0,
    attenuation=1.0,
    no_edges=False,
    no_fill=False,
    debug_mode=None,
    parallel=True,
    verbose=False,
):
    """
    Main function that replicates the Unity shader pipeline.

    Args:
        image: Input image (BGR or grayscale)
        tile_size: Size of ASCII character tiles (default 8×8)
        sigma: Base Gaussian blur sigma
        sigma_scale: Second Gaussian scale factor (k)
        tau: DoG subtraction weight
        dog_threshold: DoG edge threshold
        edge_threshold: Minimum votes for tile edge consensus
        exposure: Brightness multiplier
        attenuation: Contrast adjustment
        no_edges: Disable edge-based characters
        no_fill: Disable luminance-based fill
        debug_mode: Visualization mode ('dog', 'sobel', 'directions', 'tiles', None)
        parallel: Use multiprocessing for tile processing
        verbose: Print timing information

    Returns:
        ascii_array: 2D array of ASCII characters
        debug_image: Debug visualization (if debug_mode is set)
        timings: Dictionary of timing information (if verbose=True)
    """
    timings = {}
    start_total = time.time()

    # Step 1: Extract luminance (Pass 1)
    t0 = time.time()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_normalized = gray.astype(np.float32) / 255.0
    timings["luminance_extraction"] = time.time() - t0

    # Step 2: Difference of Gaussians (Pass 3-4)
    t0 = time.time()
    dog_edges = difference_of_gaussians(
        gray_normalized,
        sigma=sigma,
        sigma_scale=sigma_scale,
        tau=tau,
        threshold=dog_threshold,
    )
    timings["difference_of_gaussians"] = time.time() - t0

    # Step 3: Sobel edge detection (Pass 5-6)
    t0 = time.time()
    magnitude, theta, mask = sobel_edge_detection_shader_style(dog_edges)
    timings["sobel_edge_detection"] = time.time() - t0

    # Step 4: Quantize edge directions per pixel
    t0 = time.time()
    direction_map = quantize_edge_direction(theta, mask)
    timings["quantize_directions"] = time.time() - t0

    # Step 5: Tile-based consensus voting
    t0 = time.time()
    tile_direction_map = tile_based_edge_consensus(
        direction_map,
        tile_size=tile_size,
        edge_threshold=edge_threshold,
        parallel=parallel,
    )
    timings["tile_consensus"] = time.time() - t0

    # Step 6: Downscale luminance to tile resolution
    t0 = time.time()
    height, width = gray.shape
    tile_rows = height // tile_size
    tile_cols = width // tile_size

    luminance_downscaled = cv2.resize(
        gray_normalized, (tile_cols, tile_rows), interpolation=cv2.INTER_AREA
    )
    timings["luminance_downscale"] = time.time() - t0

    # Step 7: Generate ASCII art (vectorized)
    t0 = time.time()

    # Sample tile directions at tile centers
    tile_directions_sampled = tile_direction_map[::tile_size, ::tile_size][
        :tile_rows, :tile_cols
    ]

    # Edge characters by direction
    edge_chars = {
        0: "|",  # Vertical
        1: "-",  # Horizontal
        2: "/",  # Diagonal /
        3: "\\",  # Diagonal \
    }

    # Luminance characters (10 levels, darkest to lightest)
    luminance_chars = " .*:o&8?@█"

    # Apply exposure and attenuation to luminance
    adjusted_lum = np.clip(np.abs(luminance_downscaled * exposure) ** attenuation, 0, 1)

    # Quantize to 10 levels
    lum_levels = np.floor(adjusted_lum * 10).astype(int) - 1
    lum_levels = np.clip(lum_levels, 0, len(luminance_chars) - 1)

    # Create ASCII array
    ascii_array = np.empty((tile_rows, tile_cols), dtype=object)

    # Vectorized character assignment (matching shader logic)
    # Tiles with edge consensus (commonEdgeIndex != -1): use edge direction characters
    # Tiles without consensus (commonEdgeIndex == -1): use luminance fill characters

    if not no_edges:
        # Apply edge characters where we have valid edge direction consensus
        for direction, char in edge_chars.items():
            mask = tile_directions_sampled == direction
            ascii_array[mask] = char

    if not no_fill:
        # Fill remaining positions (no edge consensus) with luminance-based characters
        no_edge_mask = tile_directions_sampled == -1
        for i in range(len(luminance_chars)):
            char_mask = no_edge_mask & (lum_levels == i)
            ascii_array[char_mask] = luminance_chars[i]

    # Handle case where both no_edges and no_fill are True, or unfilled positions
    ascii_array[ascii_array == None] = " "

    timings["ascii_generation"] = time.time() - t0

    # Debug visualizations
    t0 = time.time()
    debug_image = None
    if debug_mode == "dog":
        debug_image = (dog_edges * 255).astype(np.uint8)
    elif debug_mode == "sobel":
        debug_image = magnitude
    elif debug_mode == "directions":
        # Visualize edge directions with colors
        debug_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Crop direction_map to match actual image dimensions
        dir_map_cropped = direction_map[:height, :width]
        debug_image[dir_map_cropped == 0] = [0, 0, 255]  # Red: Vertical
        debug_image[dir_map_cropped == 1] = [0, 255, 0]  # Green: Horizontal
        debug_image[dir_map_cropped == 2] = [0, 255, 255]  # Cyan: Diagonal /
        debug_image[dir_map_cropped == 3] = [255, 255, 0]  # Yellow: Diagonal \
    elif debug_mode == "tiles":
        # Visualize tile consensus
        # Use the actual tile_direction_map dimensions for debug image
        tile_h, tile_w = tile_direction_map.shape
        debug_image = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        debug_image[tile_direction_map == 0] = [255, 0, 0]
        debug_image[tile_direction_map == 1] = [0, 255, 0]
        debug_image[tile_direction_map == 2] = [0, 255, 255]
        debug_image[tile_direction_map == 3] = [255, 255, 0]

        # Draw tile grid
        for i in range(0, tile_h, tile_size):
            if i < tile_h:
                debug_image[i, :] = [128, 128, 128]
        for j in range(0, tile_w, tile_size):
            if j < tile_w:
                debug_image[:, j] = [128, 128, 128]

    if debug_mode:
        timings["debug_visualization"] = time.time() - t0

    timings["total"] = time.time() - start_total

    if verbose:
        print("\n=== Performance Timing ===")
        print(
            f"Luminance extraction:    {timings['luminance_extraction'] * 1000:7.2f} ms"
        )
        print(
            f"Difference of Gaussians: {timings['difference_of_gaussians'] * 1000:7.2f} ms"
        )
        print(
            f"Sobel edge detection:    {timings['sobel_edge_detection'] * 1000:7.2f} ms"
        )
        print(
            f"Quantize directions:     {timings['quantize_directions'] * 1000:7.2f} ms"
        )
        print(f"Tile consensus voting:   {timings['tile_consensus'] * 1000:7.2f} ms")
        print(
            f"Luminance downscale:     {timings['luminance_downscale'] * 1000:7.2f} ms"
        )
        print(f"ASCII generation:        {timings['ascii_generation'] * 1000:7.2f} ms")
        if "debug_visualization" in timings:
            print(
                f"Debug visualization:     {timings['debug_visualization'] * 1000:7.2f} ms"
            )
        print(f"{'=' * 26}")
        print(
            f"TOTAL:                   {timings['total'] * 1000:7.2f} ms ({timings['total']:.3f} s)"
        )
        print()

    return ascii_array, debug_image, timings


def main():
    image_path = "image.jpg"

    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print("Processing image with shader-style algorithm...")
    print(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]}")

    # Downscale if image is too large (above 1920x1080)
    # Use a more reasonable max size to preserve detail
    max_width, max_height = 1920, 1920  # Allow taller images
    height, width = original_image.shape[:2]

    if width > max_width or height > max_height:
        # Calculate scale factor - preserve aspect ratio
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        print(
            f"Downscaling image to {new_width}x{new_height} ({scale:.2%} of original)"
        )
        original_image = cv2.resize(
            original_image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    else:
        print(f"Image size OK (below {max_width}x{max_height})")

    print(f"Processing image size: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"Available CPU cores: {cpu_count()}")

    # Parameters matching Unity shader defaults
    tile_size = 8
    sigma = 2.0
    sigma_scale = 1.6
    tau = 1.0
    dog_threshold = 0.01  # With inversion: lower = more area gets edges, higher = more luminance fill

    # Edge threshold: minimum number of edge pixels in a tile to trigger edge mode
    # tile_size^2 = 64 pixels per tile
    # Higher value = fewer edges detected = more luminance-based fill
    # Recommended values:
    #   - Very selective (mostly fill): 48-60 (75-95% of pixels must be edge)
    #   - Balanced: 32-40 (50-60% of pixels must be edge)
    #   - Edge-heavy: 16-24 (25-40% of pixels must be edge)
    edge_threshold = 12

    exposure = 1.0
    attenuation = 1.0

    # Generate ASCII art with parallelization and timing
    print("\n--- WITH PARALLELIZATION ---")
    ascii_array, _, timings_parallel = create_ascii_art_shader_style(
        original_image,
        tile_size=tile_size,
        sigma=sigma,
        sigma_scale=sigma_scale,
        tau=tau,
        dog_threshold=dog_threshold,
        edge_threshold=edge_threshold,
        exposure=exposure,
        attenuation=attenuation,
        no_edges=False,
        no_fill=False,
        debug_mode=None,
        parallel=True,
        verbose=True,
    )

    # Print to terminal
    print_ascii_art(ascii_array)

    # Save text version
    print("Saving ASCII art text...")
    ascii_text = "\n".join("".join(row) for row in ascii_array)
    with open("ascii_art.txt", "w", encoding="utf-8") as f:
        f.write(ascii_text)
    print(f"Saved ascii_art.txt")

    # Generate debug visualizations
    print("\nGenerating debug visualizations...")

    debug_modes = ["dog", "sobel", "directions", "tiles"]
    for mode in debug_modes:
        _, debug_img, _ = create_ascii_art_shader_style(
            original_image,
            tile_size=tile_size,
            sigma=sigma,
            sigma_scale=sigma_scale,
            tau=tau,
            dog_threshold=dog_threshold,
            edge_threshold=edge_threshold,
            exposure=exposure,
            attenuation=attenuation,
            debug_mode=mode,
            parallel=True,
            verbose=False,
        )

        if debug_img is not None:
            cv2.imwrite(f"debug_{mode}.png", debug_img)
            print(f"Saved debug_{mode}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
