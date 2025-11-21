import numpy as np
from performance import GlobalTimer
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

@GlobalTimer.time
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

        num_processes = max(1, cpu_count() - 1)
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
