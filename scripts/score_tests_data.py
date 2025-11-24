from pathlib import Path
import sys
import imageio
# ensure repo root is on sys.path so `eval_pipeline` can be imported when running this script directly
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from eval_pipeline import (
    structural_similarity_score,
    edge_preservation_ratio,
    detail_retention_index,
)


def main():
    data_dir = Path(__file__).resolve().parents[1] / "tests" / "data"
    imgs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]) if data_dir.exists() else []
    if len(imgs) < 2:
        print("Need at least two images in tests/data/ to compute scores")
        return 2
    o, a = imgs[0], imgs[1]
    print("Using files:", o, a)
    img_o = imageio.v2.imread(str(o))
    img_a = imageio.v2.imread(str(a))
    # crop to common top-left size if needed
    mh = min(img_o.shape[0], img_a.shape[0])
    mw = min(img_o.shape[1], img_a.shape[1])
    if img_o.ndim == 3:
        img_o = img_o[:mh, :mw, ...]
    else:
        img_o = img_o[:mh, :mw]
    if img_a.ndim == 3:
        img_a = img_a[:mh, :mw, ...]
    else:
        img_a = img_a[:mh, :mw]

    s = structural_similarity_score(img_o, img_a)
    e = edge_preservation_ratio(img_o, img_a)
    d = detail_retention_index(img_o, img_a)
    print(f"SSIM = {s:.6f}")
    print(f"Edge Preservation Ratio = {e:.6f}")
    print(f"Detail Retention Index = {d:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
