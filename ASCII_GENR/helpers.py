import cv2
import numpy as np
import time
import functools

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


