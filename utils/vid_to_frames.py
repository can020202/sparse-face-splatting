#!/usr/bin/env python3
"""
extract_frames.py

Extracts all frames from an MP4 video and saves them as JPEG images in the specified output directory.

Usage:
    python extract_frames.py /path/to/video.mp4 /path/to/output_dir

Dependencies:
    pip install opencv-python tqdm
"""

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, prefix: str = "frame", ext: str = ".jpg") -> None:
    """Extract all frames from *video_path* and save them as *ext* images in *output_dir*.

    Args:
        video_path: Path to the input .mp4 file.
        output_dir: Directory where extracted frames will be written.
        prefix: Filename prefix for saved frames (default "frame").
        ext: File extension for saved images (default ".jpg").
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_padding = len(str(total_frames))  # zero‑pad filenames so they sort naturally

    with tqdm(total=total_frames, unit="frame", desc="Extracting") as pbar:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            filename = f"{prefix}_{idx:0{num_padding}d}{ext}"
            cv2.imwrite(str(output_dir / filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            idx += 1
            pbar.update(1)

    cap.release()
    print(f"Saved {idx} frames to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video and save them as JPEG images.")
    parser.add_argument("video", type=Path, help="Path to the .mp4 file")
    parser.add_argument("output", type=Path, help="Directory to write extracted frames")
    parser.add_argument("--prefix", default="frame", help="Filename prefix for saved frames (default: 'frame')")
    parser.add_argument("--ext", default=".jpg", choices=[".jpg", ".jpeg"], help="Image file extension (default: .jpg)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, prefix=args.prefix, ext=args.ext)


if __name__ == "__main__":
    main()
