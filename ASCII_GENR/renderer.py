from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from performance import GlobalTimer

@GlobalTimer.time
def render_ascii_to_image(ascii_array, template_library, color_array=None, output_path="output/ascii_art_rendered.png", write_to_file=True):

    num_patches_y, num_patches_x = ascii_array.shape
    grid_of_templates = []

    # Stitch the template dictionary keys into a grid of template images
    for r in range(num_patches_y):
        # Get all templates for the current row
        row_list = ascii_array[r]
        row_images = [template_library[char] for char in row_list]
        grid_of_templates.append(row_images)

    # Create the grayscale image which will act as a mask
    grayscale_mask = np.block(grid_of_templates)

    # Apply Color or Grayscale
    if color_array is not None:

        # Determine patch dimensions from the first template in the library
        first_template = list(template_library.values())[0]
        patch_h, patch_w = first_template.shape

        # Expand grayscale mask to 3 dimensions to allow broadcasting: (H, W, 1)
        mask_3d = grayscale_mask[:, :, np.newaxis]

        # Multiply the mask (brightness) by the color
        # Since mask is 0.0-1.0 and color is 0-255, the result is correctly weighted
        final_image = mask_3d * color_array

        # Clip and Convert to uint8
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    else:
        # Fallback to white text on black background
        final_image = (grayscale_mask * 255).astype(np.uint8)

    # Save file
    if write_to_file:
        cv2.imwrite(output_path, final_image)

    return final_image

# Takes a video tensor and writes to a file
@GlobalTimer.time
def write_video_from_tensor(
    tensor: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> None:

    # Ensure correct shape
    if tensor.ndim != 4:
        print(f"Error: Input tensor must have 4 dimensions (T, H, W, C), got {tensor.ndim}.")
        return

    # Extract dimensions
    num_frames, height, width, channels = tensor.shape

    # Ensure the tensor is in the correct format (uint8 for video saving)
    if tensor.dtype != np.uint8:
        print("Warning: Converting tensor to np.uint8. Ensure pixel values are in 0-255 range.")
        tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    # Convert FourCC codec string to integer
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Handle Grayscale (C=1) vs Color (C=3)
    is_color = channels == 3

    if channels not in [1, 3]:
        print(f"Error: Expected 1 or 3 channels, got {channels}.")
        return

    # Initialize VideoWriter
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
    except Exception as e:
        print(f"Error initializing VideoWriter: {e}. Check if the codec ('{codec}') is supported on your system and the output file path is valid.")
        return

    # Write Frames
    for i in range(num_frames):
        frame = tensor[i]

        # If it's grayscale (C=1), ensure it's HxW, not HxWx1
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame.squeeze(axis=2)

        out.write(frame)

    # Release Writer
    out.release()
    print(f"Successfully saved video to: {output_path} ({num_frames} frames @ {fps} FPS)")

