from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def ansi_color_code(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def ansi_reset():
    return "\033[0m"

def print_ascii_art(ascii_array, color_array=None):
    """
    Print ASCII art directly to terminal.
    """
    height, width = ascii_array.shape
    print(f"\nASCII Art ({height}x{width} characters):\n")

    if color_array is not None:
        for row in range(height):
            line = ""
            for col in range(width):
                r, g, b = color_array[row, col]
                char = ascii_array[row, col]
                line += ansi_color_code(r, g, b) + char + ansi_reset()
            print(line)
    else:
        for row in range(height):
            print("".join(ascii_array[row, :]))
    print(f"\n")

def render_ascii_to_image(ascii_array, template_library, color_array=None, output_path="output/ascii_art_rendered.png"):

    num_patches_y, num_patches_x = ascii_array.shape
    grid_of_templates = []
    for r in range(num_patches_y):

        # Get all templates for the current row
        row_list = ascii_array[r]
        row_images = [template_library[char] for char in row_list]

        # Add this row to our grid
        grid_of_templates.append(row_images)

    # This one function does all the horizontal and vertical stitching
    final_image = np.block(grid_of_templates) * 255

    # Convert to correct data type and range
    image_to_save = np.clip(final_image, 0, 255).astype(np.uint8)

    # Save file
    cv2.imwrite(output_path, image_to_save)

    return image_to_save
