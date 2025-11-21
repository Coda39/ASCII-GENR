import os
import cv2
import numpy as np
from performance import GlobalTimer

class Preprocessor:
    def __init__(self, input_path, max_dim=1920, patch_size=8):
        self.IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.max_dim = max_dim
        self.patch_size = patch_size
        self.input_path = input_path

    @GlobalTimer.time
    def process_file(self):
        print("Preprocessing input")
        input_path = self.input_path
        root, extension = os.path.splitext(input_path)

        frame_rate = None

        if extension in self.IMAGE_EXTENSIONS:
            mode = 'image'
        elif extension in self.VIDEO_EXTENSIONS:
            mode = 'video'
        else:
            raise ValueError("Input file is not a valid image or video extension")

        frame_list_gray = []
        frame_list_color = []
        if mode == 'video':

            # Open video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise IOError("Could not open video file")

            # Get framerate for use later
            frame_rate = cap.get(cv2.CAP_PROP_FPS)

            while True:
                # read a frame
                ret, frame = cap.read()

                # Exit if no frames left to read
                if not ret:
                    break

                # Frame preprocessing
                resized_frame = resize_frame(frame, self.max_dim)
                padded_frame = pad_frame(resized_frame, self.patch_size)
                gray_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)

                # Save frames
                frame_list_color.append(padded_frame)
                frame_list_gray.append(gray_frame)

        else:

            # Open image
            frame = cv2.imread(input_path)
            if frame is None:
                raise IOError("Could not open image")

            # Frame preprocessing
            resized_frame = resize_frame(frame, self.max_dim)
            padded_frame = pad_frame(resized_frame, self.patch_size)
            gray_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)

            # Save frame
            frame_list_color.append(padded_frame)
            frame_list_gray.append(gray_frame)

        print("Preprocessing complete")
        return frame_list_gray, frame_list_color, frame_rate

@GlobalTimer.time
def resize_frame(frame, max_dim):

    # Downscale logic
    h, w = frame.shape[:2]
    if w > max_dim or h > max_dim:
        scale = min(max_dim / w, max_dim / h)
        new_w, new_h = int(w * scale), int(h * scale)
        #print(f"Downscaling to {new_w}x{new_h}")
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_frame

    return frame

@GlobalTimer.time
def pad_frame(frame, patch_size):
    # pad image to be divisible by tile_size
    img_height, img_width, _ = frame.shape # Correct: (rows, cols) -> (height, width)

    # Calculate padding needed to make dimensions divisible
    pad_right = (patch_size - (img_width % patch_size)) % patch_size
    pad_bottom = (patch_size - (img_height % patch_size)) % patch_size

    # Pad on the bottom and right, not top and left
    if pad_right == 0 and pad_bottom == 0:
        return frame

    # Pad logic handling 3D (Color) vs 2D (Grayscale)
    if len(frame.shape) == 3:
        # Pad Height, Pad Width, Don't Pad Channels
        # format: ((top, bottom), (left, right), (no_pad, no_pad))
        frame = np.pad(frame, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
    else:
        # format: ((top, bottom), (left, right))
        frame = np.pad(frame, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)

    return frame

