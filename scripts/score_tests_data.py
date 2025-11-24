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
    print("Using files:", o.name, "vs", a.name)
    print()
    
    # Metrics now handle size differences automatically
    e = edge_preservation_ratio(str(o), str(a))
    d = detail_retention_index(str(o), str(a))
    s = structural_similarity_score(str(o), str(a))
    
    print("ASCII Art Quality Metrics")
    print("=" * 40)
    print(f"Edge Preservation Ratio = {e:.6f}")
    print(f"Detail Retention Index  = {d:.6f}")
    print(f"SSIM                    = {s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
