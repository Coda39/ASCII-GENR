import cv2
import numpy as np
import time
import os
from performance import GlobalTimer
import string

# Project imports
from filters import difference_of_gaussians, sobel_edge_detection_shader_style, quantize_edge_direction
from tiling import tile_based_edge_consensus
from template_match import matchTemplates
from renderer import render_ascii_to_image, write_video_from_tensor
from preprocessing import Preprocessor, smooth_frames
from template_gen import TemplateGenerator

class ASCIIConverter:
    def __init__(self,
                 input_path,
                 color_mode=False,
                 patch_size=8,
                 sigma=2.0,
                 sigma_scale=1.6,
                 tau=1.0,
                 dog_threshold=0.04,
                 edge_threshold=8,
                 exposure=1.0,
                 attenuation=1.0,
                 no_edges=False,
                 no_fill=False,
                 debug_mode=None,
                 parallel=True,
                 verbose=True,
                 font_size=8,
                 font_file=None,
                 max_dim=1600,
                 template_matching=True,
                 temporal_smoothing=True,
                 smoothing_alpha=0.6,
                 denoise=True,
                 luminance_char_set=" .*:o&8?@█",   # must be ordered least-bright to most-bright
                 template_char_set=" .*:o?|/\\-"
                 #template_char_set=string.printable[0:95]
                 ):

        # Core Input/Output
        self.input_path = input_path
        self.color_mode = color_mode
        self.patch_size = patch_size

        # Edge Detection (Difference of Gaussians)
        self.sigma = sigma
        self.sigma_scale = sigma_scale
        self.tau = tau
        self.dog_threshold = dog_threshold

        # Tiling and Edge Filtering
        self.edge_threshold = edge_threshold

        # Tone/Luminance Adjustment
        self.exposure = exposure
        self.attenuation = attenuation

        # Mode Flags
        self.no_edges = no_edges
        self.no_fill = no_fill
        self.debug_mode = debug_mode
        self.parallel = parallel
        self.verbose = verbose
        self.template_matching = template_matching

        # Render settings
        self.font_size = font_size
        self.font_file = font_file
        self.max_dim = max_dim
        self.luminance_char_set = luminance_char_set # only for non-edges, no duplicates allowed
        self.template_char_set = template_char_set

        # Performance measurements
        self.timings = {}

        # Template matching resources
        self.template_library = None
        self.template_centroids = None

        # Temporal smoothing
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.denoise = denoise

    def run(self):
        # Initialize preprocessor
        preprocessor = Preprocessor(self.input_path, patch_size=self.patch_size, max_dim=self.max_dim, denoise=self.denoise)
        try:
            frame_list_gray, frame_list_color, framerate = preprocessor.process_file()
        except Exception as e:
            print(e)
            exit()

        # Input smoothing (reduces flickering)
        if self.temporal_smoothing:
            frame_list_gray = smooth_frames(frame_list_gray, self.smoothing_alpha)

        # Generate ASCII template images
        generator = TemplateGenerator(
            patch_w=self.patch_size,
            patch_h=self.patch_size,
            font_size=self.font_size,
            font_file=self.font_file,
            char_set=set(self.luminance_char_set + self.template_char_set)
        )
        self.template_library, self.template_centroids = generator.generate()

        # Generate output directory
        os.makedirs("output", exist_ok=True)

        # If the file is an image
        if len(frame_list_gray) == 1:

            print("Processing single image...")
            ascii_array, debug_images = self._frameToAscii(frame_list_gray[0])

            # Render Image
            color_data = frame_list_color[0] if self.color_mode else None
            render_ascii_to_image(ascii_array, self.template_library, color_array=color_data, output_path="output/ascii_image_output.png")
            print("Saved output/ascii_image_output.png")

            if debug_images is not None:
                d_dog, d_sobel, d_dir, d_tile, ne_array, e_array = debug_images
                cv2.imwrite(f"output/debug_dog.png", d_dog)
                cv2.imwrite(f"output/debug_sobel.png", d_sobel)
                cv2.imwrite(f"output/debug_directions.png", d_dir)
                cv2.imwrite(f"output/debug_tiles.png", d_tile)
                cv2.imwrite(f"output/debug_preprocessed.png", frame_list_gray[0])

                render_ascii_to_image(ne_array, self.template_library, color_array=color_data, output_path="output/debug_noedge.png")
                render_ascii_to_image(e_array, self.template_library, color_array=color_data, output_path="output/debug_edge.png")
                print("Saved debug images to output/")

        # If the file is a video
        else:

            print(f"Processing video ({len(frame_list_gray)} frames)...")
            ascii_raster_list = []

            # Lists to store debug frames
            debug_lists = {
                "dog": [], "sobel": [], "directions": [], "tiles": [], "noedge": [], "edge": []
            }

            for i, frame in enumerate(frame_list_gray):
                if i % 10 == 0:
                    print(f"\rProgress: {(i / len(frame_list_gray)) * 100.0:.2f}%", end="")

                # Convert frame to ASCII
                ascii_array, debug_images = self._frameToAscii(frame)

                color_data = frame_list_color[i] if self.color_mode else None
                ascii_raster = render_ascii_to_image(
                    ascii_array,
                    self.template_library,
                    color_array=color_data,
                    write_to_file=False
                )
                ascii_raster_list.append(ascii_raster)

                # Store debug frames
                if debug_images is not None:
                    d_dog, d_sobel, d_dir, d_tile, ne_array, e_array = debug_images

                    # Ensure grayscale images have 3 dimensions (H, W, 1) for stacking
                    if len(d_dog.shape) == 2: d_dog = d_dog[:, :, np.newaxis]
                    if len(d_sobel.shape) == 2: d_sobel = d_sobel[:, :, np.newaxis]

                    # Convert ascii arrays to raster images
                    noedge_frame = render_ascii_to_image(ne_array, self.template_library, color_array=color_data, write_to_file=False)
                    edge_frame = render_ascii_to_image(e_array, self.template_library, color_array=color_data, write_to_file=False)

                    # Add color dimension if it does not exist
                    if not self.color_mode:
                        noedge_frame = noedge_frame[:, :, np.newaxis]
                        edge_frame = edge_frame[:, :, np.newaxis]

                    debug_lists["dog"].append(d_dog)
                    debug_lists["sobel"].append(d_sobel)
                    debug_lists["directions"].append(d_dir)
                    debug_lists["tiles"].append(d_tile)
                    debug_lists["noedge"].append(noedge_frame)
                    debug_lists["edge"].append(edge_frame)

            print(f"\rProgress: 100.00%")
            print("\nVideo processing complete. Saving output...")

            # Stack frames into tensor (T, H, W, C)
            if ascii_raster_list:
                ascii_video_array = np.stack(ascii_raster_list, axis=0) # Stack along time
                if not self.color_mode:
                    ascii_video_array = ascii_video_array[:, :, :, np.newaxis] # add a color channel if there isn't one already

                # Write video file
                output_video_path = "output/ascii_video_output.mp4"
                write_video_from_tensor(ascii_video_array, output_video_path, fps=framerate)

            # Write Debug Videos
            if self.debug_mode and debug_lists["dog"]:
                print("Saving debug videos...")
                for name, frames in debug_lists.items():
                    if frames:
                        tensor = np.stack(frames, axis=0)
                        write_video_from_tensor(tensor, f"output/debug_{name}.mp4", fps=framerate)

                # Pre-processed video
                tensor = np.stack(frame_list_gray, axis=0)[:, :, :, np.newaxis]
                write_video_from_tensor(tensor, f"output/debug_preprocessed.mp4", fps=framerate)

    def _frameToAscii(self, frame):
        """
        Process a single frame (grayscale) into an ASCII character array.
        """

        # 1. Normalize (already grayscale)
        frame = frame.astype(np.float32) / 255.0

        # 2. DoG (Difference of Gaussians)
        dog_edges = difference_of_gaussians(
            frame,
            self.sigma,
            self.sigma_scale,
            self.tau,
            self.dog_threshold
        )

        # 3. Sobel Edge Detection
        magnitude, theta, mask = sobel_edge_detection_shader_style(dog_edges)

        # 4. Quantize Directions
        direction_map = quantize_edge_direction(theta, mask)

        # 5. Tile Consensus
        tile_direction_map = tile_based_edge_consensus(
            direction_map,
            self.patch_size,
            self.edge_threshold,
            self.parallel
        )

        # 6. Downscale Luminance (for resizing checks)
        height, width = frame.shape # Original dimensions
        tile_rows = frame.shape[0] // self.patch_size
        tile_cols = frame.shape[1] // self.patch_size

        # 7. Generate ASCII
        # Sample the top-left corner of every tile in the consensus map
        tile_directions_sampled = tile_direction_map[::self.patch_size, ::self.patch_size]

        # Ensure sampling matches the grid size (trim if necessary due to padding logic nuances)
        tile_directions_sampled = tile_directions_sampled[:tile_rows, :tile_cols]

        # Possible edge candidates
        edge_chars = {0: "|", 1: "-", 2: "/", 3: "\\"}

        # Populate array with the chosen edges
        edge_ascii_array = np.full((tile_rows, tile_cols), " ", dtype=object)

        if not self.no_edges:
            for direction, char in edge_chars.items():
                dir_mask = tile_directions_sampled == direction
                edge_ascii_array[dir_mask] = char

        # Populate a completely disjoint array with template matched characters
        no_edge_mask = tile_directions_sampled == -1

        if not self.no_fill:

            # Use template matching to fill non-edge patches
            if self.template_matching:
                noedge_ascii_array, _ = matchTemplates(
                    frame,
                    no_edge_mask,
                    self.template_library,
                    self.template_centroids,
                    self.template_char_set
                )
            # Use luminance matching to fill non-edge patches
            else:
                # Luminance characters (10 levels, darkest to lightest)
                luminance_chars = self.luminance_char_set

                luminance_downscaled = cv2.resize(
                    frame, (tile_cols, tile_rows), interpolation=cv2.INTER_AREA
                )

                # Apply exposure and attenuation to luminance
                adjusted_lum = np.clip(np.abs(luminance_downscaled * self.exposure) ** self.attenuation, 0, 1)

                # Quantize to 10 levels
                lum_levels = np.floor(adjusted_lum * 10).astype(int) - 1
                lum_levels = np.clip(lum_levels, 0, len(luminance_chars) - 1)

                no_edge_mask = tile_directions_sampled == -1
                noedge_ascii_array = np.full((tile_rows, tile_cols), " ", dtype=object)
                for i in range(len(luminance_chars)):
                    char_mask = no_edge_mask & (lum_levels == i)
                    noedge_ascii_array[char_mask] = luminance_chars[i]

        # Combine edge array and matching array using the mask
        ascii_array = np.where(no_edge_mask, noedge_ascii_array, edge_ascii_array)

        # Ensure no None values exist
        ascii_array[ascii_array == None] = " "

        # --- Debugging ---
        debug_images = None
        if self.debug_mode:
            # DOG
            debug_dog = (dog_edges * 255).astype(np.uint8)

            # SOBEL
            debug_sobel = magnitude

            # DIRECTIONS
            debug_directions = np.zeros((height, width, 3), dtype=np.uint8)
            # Crop direction map to original size for visualization
            dir_map_cropped = direction_map[:height, :width]
            colors = {0: [0,0,255], 1: [0,255,0], 2: [0,255,255], 3: [255,255,0]} # BGR
            for k, v in colors.items():
                debug_directions[dir_map_cropped == k] = v

            # TILES
            tile_h, tile_w = tile_direction_map.shape
            debug_tiles = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            colors = {0: [255,0,0], 1: [0,255,0], 2: [0,255,255], 3: [255,255,0]} # BGR
            for k, v in colors.items():
                debug_tiles[tile_direction_map == k] = v

            debug_images = (debug_dog, debug_sobel, debug_directions, debug_tiles, noedge_ascii_array, edge_ascii_array)

        return ascii_array, debug_images
