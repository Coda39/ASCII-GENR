"""Simple CLI to run evaluation metrics on image pairs or directories."""
import argparse
from pathlib import Path
import time
import sys
from eval_pipeline import (
    edge_preservation_ratio,
    detail_retention_index,
    structural_similarity_score,
    temporal_consistency_score,
    read_video_frames,
    evaluate_video_quality,
)


def main():
    p = argparse.ArgumentParser(description="Evaluate ASCII conversion quality")
    p.add_argument("original", help="Path to original image/video or directory")
    p.add_argument("ascii", help="Path to ASCII-converted image/video or directory")
    p.add_argument("--video", action="store_true", help="Treat inputs as videos and evaluate comprehensively")
    p.add_argument("--max-frames", type=int, help="Maximum frames to evaluate for video (default: all)")
    p.add_argument("--sample-rate", type=int, default=1, help="Evaluate every Nth frame (default: 1)")
    args = p.parse_args()
    orig = Path(args.original)
    asc = Path(args.ascii)
    
    if args.video:
        start_time = time.time()
        print("Evaluating video quality...")
        print("=" * 70)
        
        # Progress callback
        def show_progress(current, total):
            elapsed = time.time() - start_time
            percent = (current / total) * 100
            avg_time = elapsed / current
            remaining = (total - current) * avg_time
            
            # Clear line and print progress
            sys.stdout.write(f"\rProcessing frame {current}/{total} ({percent:.1f}%) - "
                           f"Elapsed: {elapsed:.1f}s - Remaining: ~{remaining:.1f}s")
            sys.stdout.flush()
        
        # Status callback
        def show_status(message):
            print(f"\n{message}", end='', flush=True)
        
        results = evaluate_video_quality(
            str(orig), 
            str(asc), 
            max_frames=args.max_frames,
            sample_rate=args.sample_rate,
            progress_callback=show_progress,
            status_callback=show_status
        )
        elapsed_time = time.time() - start_time
        
        print(f" done\n\nVideo Analysis ({results['num_frames']} frames evaluated in {elapsed_time:.2f}s)")
        print("=" * 70)
        
        print(f"  Edge Preservation Ratio:")
        print(f"    Mean: {results['edge_preservation']['mean']:.4f}  "
              f"(±{results['edge_preservation']['std']:.4f})")
        print(f"    Range: [{results['edge_preservation']['min']:.4f}, "
              f"{results['edge_preservation']['max']:.4f}]")
        
        print(f"\n  Detail Retention Index:")
        print(f"    Mean: {results['detail_retention']['mean']:.4f}  "
              f"(±{results['detail_retention']['std']:.4f})")
        print(f"    Range: [{results['detail_retention']['min']:.4f}, "
              f"{results['detail_retention']['max']:.4f}]")
        
        print(f"\n  Temporal Consistency:")
        print(f"    Original video: {results['temporal_consistency']['original']:.4f}")
        print(f"    ASCII video:    {results['temporal_consistency']['ascii']:.4f}")
        print(f"    Degradation:    {results['temporal_consistency']['degradation']:.4f}")
        
        print(f"\n  SSIM Score:")
        print(f"    Mean: {results['ssim']['mean']:.4f}  "
              f"(±{results['ssim']['std']:.4f})")
        print(f"    Range: [{results['ssim']['min']:.4f}, "
              f"{results['ssim']['max']:.4f}]")
        return
    
    # if directories, pair by name
    if orig.is_dir() and asc.is_dir():
        start_time = time.time()
        # gather common filenames
        ofiles = {p.name: p for p in orig.iterdir() if p.is_file()}
        afiles = {p.name: p for p in asc.iterdir() if p.is_file()}
        common = sorted(set(ofiles) & set(afiles))
        if not common:
            print("No common files found in directories")
            return
        print(f"Evaluating {len(common)} image pairs...")
        print(f"{'Filename':<30} {'EdgePres':>10} {'DRI':>10} {'SSIM':>10}")
        print("-" * 62)
        for idx, name in enumerate(common, 1):
            o = str(ofiles[name])
            a = str(afiles[name])
            e = edge_preservation_ratio(o, a)
            d = detail_retention_index(o, a)
            s = structural_similarity_score(o, a)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(common) - idx) * avg_time
            
            print(f"{name:<30} {e:>10.4f} {d:>10.4f} {s:>10.4f}  "
                  f"[{idx}/{len(common)}, ~{remaining:.0f}s remaining]")
        elapsed_time = time.time() - start_time
        print("-" * 62)
        print(f"Completed in {elapsed_time:.2f}s ({elapsed_time/len(common):.2f}s per image)")
        return
    
    # else treat as files
    start_time = time.time()
    print("Evaluating image...")
    print("=" * 50)
    
    # Step 1: Edge Preservation
    print("Step 1/3: Computing edge preservation...", end='', flush=True)
    step_start = time.time()
    e = edge_preservation_ratio(str(orig), str(asc))
    step_time = time.time() - step_start
    print(f" done ({step_time:.1f}s)")
    
    # Step 2: Detail Retention
    print("Step 2/3: Computing detail retention...", end='', flush=True)
    step_start = time.time()
    d = detail_retention_index(str(orig), str(asc))
    step_time = time.time() - step_start
    print(f" done ({step_time:.1f}s)")
    
    # Step 3: SSIM
    print("Step 3/3: Computing SSIM...", end='', flush=True)
    step_start = time.time()
    s = structural_similarity_score(str(orig), str(asc))
    step_time = time.time() - step_start
    print(f" done ({step_time:.1f}s)")
    
    elapsed_time = time.time() - start_time
    
    print("\nASCII Art Quality Metrics")
    print("=" * 50)
    print(f"Edge Preservation Ratio: {e:.4f} ")
    print(f"Detail Retention Index:  {d:.4f} ")
    print(f"SSIM Score:              {s:.4f}")
    print(f"\nTotal evaluation time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
