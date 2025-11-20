import numpy as np
import cv2
from template_gen import TemplateGenerator
from helpers import calcCentroids, SAD
from scipy.ndimage import shift

def matchTemplates(image, no_edge_mask, patch_size=(8, 8), font_size=10, font_file=None):

    # Restore original variables
    PATCH_H, PATCH_W = patch_size
    padded_img = image
    padded_height, padded_width = image.shape

    # Generate ascii raster templates
    generator = TemplateGenerator(patch_w=PATCH_W, patch_h=PATCH_H, font_size=font_size, font_file=font_file)
    template_library, template_centroids = generator.generate()

    # For each patch
    num_patches_y, num_patches_x = no_edge_mask.shape
    minChars = np.full((num_patches_y, num_patches_x), " ", dtype=object)   # stores the closest chars for each patch
    for patch_y in range(num_patches_y):
        idy = patch_y * PATCH_H
        for patch_x in range(num_patches_x):
            idx = patch_x * PATCH_W

            # skip patches identified as edges
            if not no_edge_mask[patch_y, patch_x]:
                continue;

            # get the patch from the image
            patch = padded_img[idy:idy+PATCH_H, idx:idx+PATCH_W]

            # compute its centroid
            pcx, pcy = calcCentroids(patch)

            # for each template
            minDistance = 9999999999999
            minChar = None
            for char, template in template_library.items():
                tcx, tcy = template_centroids[char]

                # calculate difference between patch center and template center
                dcx = pcx - tcx
                dcy = pcy - tcy

                # shift template image
                shifted_template = shift(template, (dcy, dcx), cval=0)

                # calculate the distance
                distance = SAD(patch, shifted_template)

                if distance < minDistance:
                    minDistance = distance
                    minChar = char

            # Save closest char for later
            minChars[patch_y, patch_x] = minChar

    return minChars, template_library



