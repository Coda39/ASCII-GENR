"""Simple CLI to run evaluation metrics on image pairs or directories."""
import argparse
from pathlib import Path
from eval_pipeline import (
    structural_similarity_score,
    edge_preservation_ratio,
    temporal_consistency_score,
    detail_retention_index,
    read_video_frames,
)


def main():
    p = argparse.ArgumentParser(description="Evaluate ASCII conversion quality")
    p.add_argument("original", help="Path to original image or directory (or video)")
    p.add_argument("ascii", help="Path to ASCII-converted image or directory (or video)")
    p.add_argument("--video", action="store_true", help="Treat inputs as videos and evaluate temporal consistency")
    args = p.parse_args()
    orig = Path(args.original)
    asc = Path(args.ascii)
    if args.video:
        o_frames = read_video_frames(str(orig))
        a_frames = read_video_frames(str(asc))
        tscore = temporal_consistency_score(a_frames)
        print(f"Temporal Consistency (ascii): {tscore:.4f}")
        # Also compute temporal for original as reference
        tscore_o = temporal_consistency_score(o_frames)
        print(f"Temporal Consistency (original): {tscore_o:.4f}")
        return
    # if directories, pair by name
    if orig.is_dir() and asc.is_dir():
        # gather common filenames
        ofiles = {p.name: p for p in orig.iterdir() if p.is_file()}
        afiles = {p.name: p for p in asc.iterdir() if p.is_file()}
        common = sorted(set(ofiles) & set(afiles))
        if not common:
            print("No common files found in directories")
            return
        for name in common:
            o = str(ofiles[name])
            a = str(afiles[name])
            s = structural_similarity_score(o, a)
            e = edge_preservation_ratio(o, a)
            d = detail_retention_index(o, a)
            print(f"{name}: SSIM={s:.4f}, EdgePres={e:.4f}, DRI={d:.4f}")
        return
    # else treat as files
    s = structural_similarity_score(str(orig), str(asc))
    e = edge_preservation_ratio(str(orig), str(asc))
    d = detail_retention_index(str(orig), str(asc))
    print(f"SSIM={s:.4f}\nEdgePres={e:.4f}\nDRI={d:.4f}")


if __name__ == "__main__":
    main()
