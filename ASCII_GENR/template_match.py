import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from scipy.ndimage import shift
from helpers import calcCentroids, NCC, SAD
from performance import GlobalTimer

def _process_row(args):
    """
    Worker function to process a single row of patches.
    Must be top-level to be picklable by multiprocessing.
    """
    (row_idx,
     img_row_slice,
     mask_row_slice,
     patch_width,
     patch_height,
     template_library,
     template_centroids,
     char_set) = args

    # FIX: Use the mask length to determine the number of patches.
    # The image might be padded (larger), but we only process valid mask tiles.
    num_patches_x = mask_row_slice.shape[0]

    row_best_chars = []

    for patch_x in range(num_patches_x):
        idx = patch_x * patch_width

        # Check mask (1D array for this row's patch indices)
        if not mask_row_slice[patch_x]:
            row_best_chars.append(" ")
            continue

        # Extract the specific patch from this row strip
        # Because img_row_slice is padded, this slice is always safe.
        patch = img_row_slice[:, idx : idx + patch_width]

        # Compute centroid
        pcx, pcy = calcCentroids(patch)

        maxScore = -1.0
        bestChar = " "

        minDistance = 9999999999


        # Iterate through all templates
        for char, in char_set:
            template = template_library[char]
            tcx, tcy = template_centroids[char]

            # Calculate shift
            dcx = pcx - tcx
            dcy = pcy - tcy

            # Shift template image
            shifted_template = shift(template, (dcy, dcx), cval=0)

            # Use NCC
            # score = NCC(patch, shifted_template)
            #
            # if score > maxScore:
            #     maxScore = score
            #     bestChar = char

            # Use SAD
            distance = SAD(patch, shifted_template)

            if distance < minDistance:
                minDistance = distance
                bestChar = char

        row_best_chars.append(bestChar)

    return row_idx, row_best_chars

@GlobalTimer.time
def matchTemplates(image, no_edge_mask, template_library, template_centroids, char_set):

    # NOTE: Maybe de-noise image before matching

    if len(template_library) == 0:
        raise ValueError("Error: No templates provided for matching")

    PATCH_H, PATCH_W = next(iter(template_library.values())).shape

    num_patches_y, num_patches_x = no_edge_mask.shape

    # Prepare arguments for parallel processing
    process_args = []

    for patch_y in range(num_patches_y):
        idy = patch_y * PATCH_H

        # Slice the image strip for this specific row of patches
        img_row_slice = image[idy : idy + PATCH_H, :]

        # Slice the mask row
        mask_row_slice = no_edge_mask[patch_y, :]

        args = (
            patch_y,
            img_row_slice,
            mask_row_slice,
            PATCH_W,
            PATCH_H,
            template_library,
            template_centroids,
            char_set
        )
        process_args.append(args)

    # Determine number of cores
    num_processes = max(1, cpu_count() - 1)
    #print(f"Matching templates using {num_processes} processes...")

    # Execute parallel processing
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_row, process_args)

    # Reassemble results into the final 2D array
    bestChars = np.full((num_patches_y, num_patches_x), " ", dtype=object)

    for row_idx, row_chars in results:
        bestChars[row_idx, :] = row_chars

    return bestChars, template_library
