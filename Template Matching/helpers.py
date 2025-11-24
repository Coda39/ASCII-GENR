import cv2
import numpy as np

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

    sad_value = np.sum(dif, dtype=np.int64)

    return sad_value
