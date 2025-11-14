# import cupy as cp
import numpy as np
import cv2
from templates import TemplateGenerator
from helpers import calcCentroids, SAD
from scipy.ndimage import shift

# PARAMETERS
PATCH_W, PATCH_H = 10, 10
SCALE_FACTOR = 3.0
COLOR = True
generator = TemplateGenerator(patch_w=PATCH_W, patch_h=PATCH_H)
template_library, template_centroids = generator.generate()

# get input image
filepath = './cloud.png'
if COLOR:
    img_array = cv2.imread(filepath)
else:
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

if img_array is None:
    print("Error loading image")
    exit()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_array = clahe.apply(img_array)

# Upscale the image using Bicubic interpolation for high quality
img_array = cv2.resize(img_array,
                       dsize=None,
                       fx=SCALE_FACTOR,
                       fy=SCALE_FACTOR,
                       interpolation=cv2.INTER_CUBIC)

# pad image to be divisible by PATCH_W and PATCH_H
img_height, img_width = img_array.shape # Correct: (rows, cols) -> (height, width)

# Calculate padding needed to make dimensions divisible
pad_right = img_width % PATCH_W
pad_bottom = img_height % PATCH_H

# Pad on the bottom and right, not top and left
padded_img = np.pad(img_array, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
padded_height, padded_width = padded_img.shape

# For each patch
minTemplates = []   # stores the closest templates for each patch
num_patches_y = int(padded_height / PATCH_H) # 'y' is height
num_patches_x = int(padded_width / PATCH_W)  # 'x' is width

for patch_y in range(num_patches_y):
    idy = patch_y * PATCH_H
    for patch_x in range(num_patches_x):
        idx = patch_x * PATCH_W

        # get the patch from the image
        patch = padded_img[idy:idy+PATCH_H, idx:idx+PATCH_W]

        # compute its centroid
        pcx, pcy = calcCentroids(patch)

        # for each template
        minDistance = 9999999999999
        minTemplate = None
        for i in range(len(template_library)):
            template = template_library[i]
            tcx, tcy = template_centroids[i]

            # calculate difference between patch center and template center
            dcx = pcx - tcx
            dcy = pcy - tcy

            # shift template image
            shifted_template = shift(template, (dcy, dcx), cval=0)

            # calculate the distance
            distance = SAD(patch, shifted_template)

            if distance < minDistance:
                minDistance = distance
                minTemplate = shifted_template

        # Save closest template for later
        minTemplates.append(minTemplate)

# Render full image
grid_of_templates = []
for r in range(num_patches_y):

    # Get all templates for the current row
    start_index = r * num_patches_x
    end_index = start_index + num_patches_x
    row_list = minTemplates[start_index:end_index]

    # Add this row to our grid
    grid_of_templates.append(row_list)

# This one function does all the horizontal and vertical stitching
final_image = np.block(grid_of_templates)

# Save image to a file
output_filename = "ascii_art_result.png" # .png or .jpg

# Convert to correct data type and range
image_to_save = np.clip(final_image, 0, 255).astype(np.uint8)

# Save file
cv2.imwrite(output_filename, image_to_save)

print(f"Image successfully saved to {output_filename}")
