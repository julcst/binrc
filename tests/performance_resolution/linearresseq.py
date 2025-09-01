#!/usr/bin/env python3
import math
import argparse

FHD_PIXELS = 1920 * 1080  # FullHD reference

def linear_pixel_sequence(steps, max_pixels=FHD_PIXELS):
    """Generate square widths so that pixel counts grow linearly up to max_pixels."""
    seq = []
    step = max_pixels / steps
    for i in range(1, steps + 1):
        target = i * step
        width = round(math.sqrt(target))
        seq.append(width)
    return seq

def main():
    parser = argparse.ArgumentParser(
        description="Generate square image widths with linearly increasing pixel counts."
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=9,
        help="Number of steps in the sequence"
    )
    parser.add_argument(
        "--max", "-m", default="FHD",
        help="Maximum pixel count (integer) or 'FHD' (default: FullHD = 1920x1080)"
    )
    parser.add_argument(
        "--show-pixels", action="store_true",
        help="Also show the pixel counts corresponding to the widths"
    )

    args = parser.parse_args()

    if args.max.upper() == "FHD":
        max_pixels = FHD_PIXELS
    else:
        max_pixels = int(args.max)

    seq = linear_pixel_sequence(args.steps, max_pixels)

    if args.show_pixels:
        for w in seq:
            print(f"{w} -> {w*w:,} pixels")
    else:
        print(seq)

if __name__ == "__main__":
    main()