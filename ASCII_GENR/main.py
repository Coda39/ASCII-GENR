import argparse
from ASCIIConverter import ASCIIConverter
from performance import GlobalTimer

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate ASCII art")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--color", action="store_true", help="Enable colored ASCII output")
    parser.add_argument("--tile-size", type=int, default=8, help="ASCII character tile size")
    parser.add_argument("--edge-threshold", type=int, default=12, help="Min edge pixels for detection")
    parser.add_argument("--dog-threshold", type=float, default=0.04, help="Min edge pixels for detection")
    parser.add_argument("--debug", action="store_true", help="Skip debug visualizations")
    parser.add_argument("--font-size", type=int, default=8, help="Font size for rendered image")
    parser.add_argument("--font-file", help="Font size for rendered image")
    parser.add_argument("--template-mode", action="store_true", help="Use template matching to fill non-edges instead of luminance values (slower, but more accurate)")
    args = parser.parse_args()

    # Create ASCIIConverter
    conv = ASCIIConverter(
        # Core Input
        input_path=args.input_path,

        # Flags & Size
        color_mode=args.color,
        patch_size=args.tile_size,

        # Thresholds
        edge_threshold=args.edge_threshold,
        dog_threshold=args.dog_threshold,

        # Mode Flags (Assumed default values for simplicity)
        sigma=2.0,
        sigma_scale=1.6,
        tau=1.0,
        exposure=1.0,
        attenuation=1.0,
        no_edges=False,
        no_fill=False,
        debug_mode=args.debug,
        parallel=True,
        verbose=False,
        font_size=args.font_size,
        font_file=args.font_file,
        template_matching=args.template_mode,
        char_set='-/|\\ .*:o&8?'
    )

    conv.run()


if __name__ == "__main__":
    GlobalTimer.start()
    main()
    GlobalTimer.end()
    GlobalTimer.print_stats()
