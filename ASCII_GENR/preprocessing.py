import os
import cv2
import numpy as np
from performance import GlobalTimer
from helpers import get_file_type

class Preprocessor:
    def __init__(self, input_path, max_dim=1280, patch_size=8, denoise=True):
        self.max_dim = max_dim
        self.patch_size = patch_size
        self.input_path = input_path

        # For input smoothing
        self.denoise = denoise

    @GlobalTimer.time
    def process_file(self):
        print("Preprocessing input")
        input_path = self.input_path
        mode = get_file_type(input_path)

        frame_rate = None

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
                denoised_frame = denoise_frame(gray_frame) if self.denoise else gray_frame

                # Save frames
                frame_list_color.append(padded_frame)
                frame_list_gray.append(denoised_frame)

        elif mode == 'image':

            # Open image
            frame = cv2.imread(input_path)
            if frame is None:
                raise IOError("Could not open image")

            # Frame preprocessing
            resized_frame = resize_frame(frame, self.max_dim)
            padded_frame = pad_frame(resized_frame, self.patch_size)
            gray_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)
            denoised_frame = denoise_frame(gray_frame)

            # Save frame
            frame_list_color.append(padded_frame)
            frame_list_gray.append(denoised_frame)

        else:
            raise ValueError(f"Cannot process file {input_path}. Invalid file type.")

        print("Preprocessing complete")
        return frame_list_gray, frame_list_color, frame_rate

@GlobalTimer.time
def smooth_frames(frame_list, alpha=0.6):

    accumulator = None
    processed_frame_list = []
    for frame in frame_list:

        if accumulator is None:
            accumulator = frame.astype(np.float32)
            processed_frame_list.append(frame)
            continue

        current = frame.astype(np.float32)
        cv2.accumulateWeighted(current, accumulator, alpha)
        processed_frame_list.append(accumulator.astype(np.uint8))

    return processed_frame_list


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

@GlobalTimer.time
def denoise_frame(frame, d=5, sigmaColor=75, sigmaSpace=75):
    if frame.dtype != np.uint8:
        frame_uint8 = (frame * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(frame_uint8, d, sigmaColor, sigmaSpace)
        return denoised.astype(np.float32) / 255.0

    return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)


