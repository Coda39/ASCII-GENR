import cv2
import numpy as np
import time
import functools
import os
from performance import GlobalTimer
import subprocess

def calcCentroids(img):
    moments = cv2.moments(img)
    M00 = moments['m00']

    # Handle the case of a completely black image
    if M00 == 0:
        H, W = img.shape
        return W / 2, H / 2 # Return geometric center

    # Calculate centroids
    cx = moments['m10'] / M00
    cy = moments['m01'] / M00

    return cx, cy

def SAD(img1, img2):

    if not img1.shape == img2.shape:
        print(f"Error calculating SAD. Images are not the same size ({img1.shape}) vs ({img2.shape})")
        return None

    img1_cast = img1.astype(np.float32)
    img2_cast = img2.astype(np.float32)

    dif = np.abs(img1_cast - img2_cast)

    sad_value = np.sum(dif)

    return sad_value

def NCC(img1, img2):

    if img1.shape != img2.shape:
        return -1

    # Flatten arrays to treat them as vectors
    v1 = img1.flatten().astype(np.float32)
    v2 = img2.flatten().astype(np.float32)

    # Zero-mean (remove brightness bias)
    # v1 -= np.mean(v1)
    # v2 -= np.mean(v2)

    # Normalize (remove contrast bias)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return -1 # Avoid division by zero for flat patches

    correlation = np.dot(v1, v2) / (norm1 * norm2)

    return correlation

def evaluate(input_path):

    filetype = get_file_type(input_path)
    if filetype == 'image':
        command = ["python", "evaluation/evaluation.py", input_path, "output/ascii_image_output.png"]
    elif filetype == 'video':
        command = ["python", "evaluation/evaluation.py", input_path, "output/ascii_video_output.mp4", "--video"]
    else:
        print(f"[ERROR] Cannot run evaluation on {input_path}. Invalid file type.")
        raise ValueError()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')

    return_code = process.wait()

    if return_code != 0:
        print(f"[ERROR]: Evaluation failed with return code {return_code}")
        print(process.stderr.read())

def get_file_type(file_path):
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    root, extension = os.path.splitext(file_path)

    if extension in IMAGE_EXTENSIONS:
        return 'image'
    elif extension in VIDEO_EXTENSIONS:
        return 'video'
    else:
        return None


