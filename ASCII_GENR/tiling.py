import numpy as np
from multiprocessing import Pool, cpu_count

def _process_tile(args):
    """
    Helper function for parallel processing. Must be top-level for pickle.
    """
    tr, tc, tile_data, edge_threshold = args

    buckets = [0, 0, 0, 0]
    for direction in range(4):
        buckets[direction] = np.sum(tile_data == direction)

    max_value = 0
    common_edge_index = -1
    for j in range(4):
        if buckets[j] > max_value:
            common_edge_index = j
            max_value = buckets[j]

    if max_value < edge_threshold:
        common_edge_index = -1

    return (tr, tc, common_edge_index)

def tile_based_edge_consensus(direction_map, tile_size=8, edge_threshold=8, parallel=True):
    """
    Apply tile-based voting for dominant edge direction.
    """
    height, width = direction_map.shape
    tile_rows = height // tile_size
    tile_cols = width // tile_size
    tile_directions = np.full((tile_rows, tile_cols), -1, dtype=np.int8)

    if parallel and tile_rows * tile_cols > 100:
        tile_args = []
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                tile = direction_map[
                    tr * tile_size : (tr + 1) * tile_size,
                    tc * tile_size : (tc + 1) * tile_size,
                ]
                tile_args.append((tr, tc, tile, edge_threshold))

        num_processes = min(cpu_count(), 8)
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_tile, tile_args)

        for tr, tc, consensus in results:
            tile_directions[tr, tc] = consensus
    else:
        # Serial processing
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                tile = direction_map[
                    tr * tile_size : (tr + 1) * tile_size,
                    tc * tile_size : (tc + 1) * tile_size,
                ]
                args = (tr, tc, tile, edge_threshold)
                _, _, consensus = _process_tile(args)
                tile_directions[tr, tc] = consensus

    # Upsample tile directions back to full resolution
    tile_direction_map = np.repeat(
        np.repeat(tile_directions, tile_size, axis=0), tile_size, axis=1
    )
    return tile_direction_map[:height, :width]

def extract_tile_colors(image, tile_size=8):
    """
    Extract color from center pixel of each tile.
    """
    height, width = image.shape[:2]
    tile_rows = height // tile_size
    tile_cols = width // tile_size

    # Handle grayscale vs BGR
    is_color = len(image.shape) == 3
    channels = 3
    color_array = np.zeros((tile_rows, tile_cols, channels), dtype=np.uint8)

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            center_y = tr * tile_size + tile_size // 2
            center_x = tc * tile_size + tile_size // 2

            if is_color:
                bgr = image[center_y, center_x]
                color_array[tr, tc] = [bgr[2], bgr[1], bgr[0]]  # BGR to RGB
            else:
                val = image[center_y, center_x]
                color_array[tr, tc] = [val, val, val]

    return color_array
