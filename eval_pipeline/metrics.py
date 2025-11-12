"""Metrics for evaluating ASCII conversion quality.

Implemented metrics:
- structural_similarity_score: SSIM using scikit-image.
- edge_preservation_ratio: Canny edges + dilation to compute preserved edge ratio.
- temporal_consistency_score: average normalized cross-correlation between consecutive frames.
- detail_retention_index: local variance based detail retention index.

The functions accept either numpy arrays (H,W[,C]) or file paths.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import imageio
from skimage.metrics import structural_similarity as _ssim
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import disk, dilation


def load_image(img: Union[str, np.ndarray]) -> np.ndarray:
    """Load image from path or return array unchanged.

    Returns float ndarray in range [0,1].
    """
    if isinstance(img, str):
        arr = imageio.v2.imread(img)
    else:
        arr = np.asarray(img)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        # convert to float in [0,1]
        arr = arr.astype(np.float32)
        if arr.max() > 1:
            arr = arr / 255.0
    return arr


def _to_gray(img: Union[str, np.ndarray]) -> np.ndarray:
    im = load_image(img)
    if im.ndim == 3 and im.shape[2] == 3:
        return rgb2gray(im)
    if im.ndim == 3 and im.shape[2] == 4:
        # drop alpha
        return rgb2gray(im[..., :3])
    return im


def structural_similarity_score(
    original: Union[str, np.ndarray],
    ascii_img: Union[str, np.ndarray],
    **ssim_kwargs,
) -> float:
    """Compute SSIM between original and ASCII-converted image.

    Both inputs may be file paths or numpy arrays. Returns SSIM in [-1,1].
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    # ensure same shape
    if a.shape != b.shape:
        raise ValueError("Original and ASCII images must have the same shape")
    # structural_similarity expects data_range (max-min). We use 1.0 for normalized floats.
    score = _ssim(a, b, data_range=1.0, **ssim_kwargs)
    return float(score)


def edge_preservation_ratio(
    original: Union[str, np.ndarray],
    ascii_img: Union[str, np.ndarray],
    sigma: float = 1.0,
    dilate_radius: int = 1,
) -> float:
    """Compute the Edge Preservation Ratio (EPR).

    EPR = (# of original edge pixels that have a matching ascii edge nearby) / (# original edge pixels)

    Uses Canny edge detector. A small dilation is applied to ASCII edges to allow for small shifts.
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    if a.shape != b.shape:
        raise ValueError("Original and ASCII images must have the same shape")
    edges_a = canny(a, sigma=sigma)
    edges_b = canny(b, sigma=sigma)
    # dilate ascii edges
    if dilate_radius > 0:
        selem = disk(dilate_radius)
        edges_b_d = dilation(edges_b, selem)
    else:
        edges_b_d = edges_b
    orig_count = float(edges_a.sum())
    if orig_count == 0:
        # No edges in original: define preservation as 1.0 when no original edges exist
        return 1.0
    preserved = float((edges_a & edges_b_d).sum())
    return float(preserved / orig_count)


def temporal_consistency_score(
    frames: Iterable[Union[str, np.ndarray]],
    max_frames: Optional[int] = None,
) -> float:
    """Compute temporal consistency score for a sequence of ASCII frames.

    The score is the mean Pearson correlation coefficient between consecutive frames.
    Inputs are images (or paths). Frames are converted to grayscale and flattened.
    Returns a score in [-1,1], where values close to 1 indicate high temporal stability.
    """
    # load frames into list
    imgs = []
    for i, f in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        imgs.append(_to_gray(f))
    if len(imgs) < 2:
        return 1.0
    # ensure shapes match
    base_shape = imgs[0].shape
    for im in imgs:
        if im.shape != base_shape:
            raise ValueError("All frames must have same shape")
    cors = []
    for i in range(len(imgs) - 1):
        x = imgs[i].ravel()
        y = imgs[i + 1].ravel()
        # handle constant arrays
        sx = x.std()
        sy = y.std()
        if sx == 0 or sy == 0:
            # if both constant and equal, correlation 1.0 else 0.0
            cors.append(1.0 if np.allclose(x, y) else 0.0)
            continue
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            cors.append(0.0)
        else:
            cors.append(float(corr))
    return float(np.mean(cors))


def detail_retention_index(
    original: Union[str, np.ndarray],
    ascii_img: Union[str, np.ndarray],
    patch_size: int = 16,
    eps: float = 1e-8,
) -> float:
    """Compute Detail Retention Index (DRI) via local variance ratio.

    DRI = sum(patch_variance_ascii) / (sum(patch_variance_original) + eps)
    Higher values indicate more detail preserved (1.0 would indicate equal total variance).
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    if a.shape != b.shape:
        raise ValueError("Original and ASCII images must have the same shape")
    H, W = a.shape
    # pad to multiple of patch_size
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    if pad_h or pad_w:
        a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="reflect")
        b = np.pad(b, ((0, pad_h), (0, pad_w)), mode="reflect")
    h2, w2 = a.shape
    a_patches = a.reshape(h2 // patch_size, patch_size, w2 // patch_size, patch_size)
    a_patches = a_patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size)
    b_patches = b.reshape(h2 // patch_size, patch_size, w2 // patch_size, patch_size)
    b_patches = b_patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size)
    var_a = np.array([p.var() for p in a_patches])
    var_b = np.array([p.var() for p in b_patches])
    total_a = var_a.sum()
    total_b = var_b.sum()
    if total_a == 0:
        # if no variance in original, define DRI as 1.0 when ascii also flat
        return 1.0 if total_b == 0 else 0.0
    return float(total_b / (total_a + eps))


def read_video_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Utility to read video frames (as numpy arrays) using imageio.
    """
    r = imageio.get_reader(path)
    frames = []
    for i, fr in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(fr)
    r.close()
    return frames
