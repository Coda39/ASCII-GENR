"""Metrics for evaluating ASCII conversion quality.

Primary metrics for ASCII art evaluation:
- edge_preservation_ratio: Measures how well edges are preserved (most relevant for ASCII)
- detail_retention_index: Measures local variance/detail retention (most relevant for ASCII)

Additional metrics:
- structural_similarity_score: SSIM (less meaningful for ASCII art transformation)
- temporal_consistency_score: For video sequences

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
from skimage.transform import resize


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


def _align_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize images to match dimensions (upscales smaller to match larger)."""
    if a.shape == b.shape:
        return a, b
    
    target_h = max(a.shape[0], b.shape[0])
    target_w = max(a.shape[1], b.shape[1])
    
    if a.shape != (target_h, target_w):
        a = resize(a, (target_h, target_w), preserve_range=True, anti_aliasing=True)
    if b.shape != (target_h, target_w):
        b = resize(b, (target_h, target_w), preserve_range=True, anti_aliasing=True)
    
    return a, b


def structural_similarity_score(
    original: Union[str, np.ndarray],
    ascii_img: Union[str, np.ndarray],
    **ssim_kwargs,
) -> float:
    """Compute SSIM between original and ASCII-converted image.

    Note: SSIM is less meaningful for ASCII art due to fundamental structural changes.
    Consider using edge_preservation_ratio and detail_retention_index instead.

    Both inputs may be file paths or numpy arrays. Returns SSIM in [-1,1].
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    a, b = _align_shapes(a, b)
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
    
    Args:
        original: Original image (path or array)
        ascii_img: ASCII-converted image (path or array)
        sigma: Gaussian sigma for Canny edge detection
        dilate_radius: Dilation radius for matching nearby edges
    
    Returns:
        Float in [0,1] where 1.0 means perfect edge preservation
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    a, b = _align_shapes(a, b)
    
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
    """Compute Detail Retention Index (DRI) via normalized variance similarity.

    Measures how similar the variance distributions are between original and ASCII images.
    Uses normalized MSE on variance maps, converted to a similarity score in [0, 1].
    
    Formula: DRI = max(0, 1 - sqrt(MSE))
    
    Values in [0, 1] where:
    - 1.0 = identical variance distribution (MSE=0)
    - 0.5 = moderate similarity (MSE=0.25)
    - 0.0 = very different (MSE≥1)
    
    Taking sqrt of MSE makes the metric more sensitive to differences.
    
    Args:
        original: Original image (path or array)
        ascii_img: ASCII-converted image (path or array)
        patch_size: Size of patches for computing local variance
        eps: Small constant to avoid division by zero
    
    Returns:
        Float in [0, 1] where higher values indicate better detail retention
    """
    a = _to_gray(original)
    b = _to_gray(ascii_img)
    a, b = _align_shapes(a, b)
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
    
    # Normalize both variance maps to [0, 1] for fair comparison
    if var_a.max() > 0:
        var_a_norm = var_a / var_a.max()
    else:
        var_a_norm = var_a
        
    if var_b.max() > 0:
        var_b_norm = var_b / var_b.max()
    else:
        var_b_norm = var_b
    
    # Compute normalized MSE
    mse = np.mean((var_a_norm - var_b_norm) ** 2)
    
    # Convert MSE to similarity score: 1 - sqrt(MSE)
    # sqrt makes it more sensitive to differences
    dri = max(0.0, 1.0 - np.sqrt(mse))
    return float(dri)


def read_video_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Utility to read video frames (as numpy arrays) using imageio.
    """

    try:
        r = imageio.get_reader(path, format="FFMPEG")
    except Exception:
        # If FFMPEG backend is unavailable or fails, fall back to auto-detection
        print("[WARNING]: Could not use FFMPEG backend.")
        r = imageio.get_reader(path)

    frames = []
    for i, fr in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(fr)
    r.close()
    return frames


def evaluate_video_quality(
    original_video: Union[str, List[np.ndarray]],
    ascii_video: Union[str, List[np.ndarray]],
    max_frames: Optional[int] = None,
    sample_rate: int = 1,
    progress_callback=None,
    status_callback=None,
) -> dict:
    """Evaluate ASCII video conversion quality across all frames.
    
    Computes frame-by-frame metrics and aggregates them, plus temporal consistency.
    
    Args:
        original_video: Path to original video or list of frames
        ascii_video: Path to ASCII-converted video or list of frames
        max_frames: Maximum number of frames to evaluate (None = all frames)
        sample_rate: Evaluate every Nth frame (1 = every frame, 2 = every other frame)
        progress_callback: Optional callback function(current, total) for progress updates
        status_callback: Optional callback function(message) for status updates
    
    Returns:
        Dictionary with mean/std of EdgePres, DRI, SSIM, plus temporal consistency scores
    """
    # Load frames
    if isinstance(original_video, str):
        o_frames = read_video_frames(original_video, max_frames)
    else:
        o_frames = original_video[:max_frames] if max_frames else original_video
    
    if isinstance(ascii_video, str):
        a_frames = read_video_frames(ascii_video, max_frames)
    else:
        a_frames = ascii_video[:max_frames] if max_frames else ascii_video
    
    # Sample frames if needed
    o_frames = o_frames[::sample_rate]
    a_frames = a_frames[::sample_rate]
    
    if len(o_frames) != len(a_frames):
        raise ValueError(f"Frame count mismatch: {len(o_frames)} vs {len(a_frames)}")
    
    if len(o_frames) == 0:
        raise ValueError("No frames to evaluate")
    
    # Compute per-frame metrics
    edge_scores = []
    dri_scores = []
    ssim_scores = []
    
    total_frames = len(o_frames)
    for idx, (o_frame, a_frame) in enumerate(zip(o_frames, a_frames)):
        edge_scores.append(edge_preservation_ratio(o_frame, a_frame))
        dri_scores.append(detail_retention_index(o_frame, a_frame))
        ssim_scores.append(structural_similarity_score(o_frame, a_frame))
        
        if progress_callback:
            progress_callback(idx + 1, total_frames)
    
    # Compute temporal consistency
    if status_callback:
        status_callback("Computing temporal consistency for original video...")
    temporal_orig = temporal_consistency_score(o_frames)
    
    if status_callback:
        status_callback("Computing temporal consistency for ASCII video...")
    temporal_ascii = temporal_consistency_score(a_frames)
    
    return {
        "num_frames": len(o_frames),
        "edge_preservation": {
            "mean": float(np.mean(edge_scores)),
            "std": float(np.std(edge_scores)),
            "min": float(np.min(edge_scores)),
            "max": float(np.max(edge_scores)),
        },
        "detail_retention": {
            "mean": float(np.mean(dri_scores)),
            "std": float(np.std(dri_scores)),
            "min": float(np.min(dri_scores)),
            "max": float(np.max(dri_scores)),
        },
        "ssim": {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "min": float(np.min(ssim_scores)),
            "max": float(np.max(ssim_scores)),
        },
        "temporal_consistency": {
            "original": temporal_orig,
            "ascii": temporal_ascii,
            "degradation": temporal_orig - temporal_ascii,
        },
    }
