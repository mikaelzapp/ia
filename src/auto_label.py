"""Pre-labelling module.

Given a folder of source videos this script:

1. Extracts ``--frames-per-video`` evenly spaced frames from each clip and
   writes them as JPEG images into ``--frames-dir``.
2. Estimates the *content* region (class 1) of every clip using a lightweight
   motion/standard-deviation heuristic built on top of OpenCV. The heuristic
   assumes that the moving part of a video is its real content, while static
   borders, logos and frames are interface (class 0).
3. Writes the suggested class-1 boxes to a CSV **and** a JSON file so the next
   stage (the human curation tool) has a sensible starting point.

The suggestions are intentionally rough: their only purpose is to save the
human curator a few clicks. They are never used directly for training.

Example:
    python -m src.auto_label \\
        --input data/raw_videos \\
        --frames-dir data/frames \\
        --frames-per-video 8
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import (
    CLASS_CONTENT,
    ensure_dir,
    get_logger,
    list_videos,
)

LOGGER = get_logger(__name__)

#: Number of frames used purely to *estimate* motion (independent of how many
#: frames are exported for labelling). More frames give a steadier estimate.
_MOTION_SAMPLE_FRAMES: int = 40


def _evenly_spaced_indices(total: int, count: int) -> List[int]:
    """Return ``count`` frame indices spread across ``[0, total)``.

    Args:
        total: Total number of frames in the video (>= 1).
        count: Number of indices to return.

    Returns:
        A sorted list of unique frame indices. When ``total`` is small the
        list may contain fewer than ``count`` indices.
    """
    if total <= 0:
        return []
    count = max(1, min(count, total))
    # np.linspace endpoints are inclusive; clip to a valid frame index.
    idx = np.linspace(0, total - 1, num=count)
    return sorted({int(round(i)) for i in idx})


def estimate_content_bbox(
    video_path: Path, sample_frames: int = _MOTION_SAMPLE_FRAMES
) -> Optional[Tuple[float, float, float, float]]:
    """Estimate the content (class 1) bounding box of a single video.

    The heuristic samples up to ``sample_frames`` greyscale frames, computes a
    per-pixel temporal standard deviation map (pixels that change a lot over
    time are likely "content"), thresholds it and takes the bounding rectangle
    of the largest connected region of motion.

    Args:
        video_path: Path to the video file.
        sample_frames: How many frames to sample for the estimate.

    Returns:
        A normalised ``(x_center, y_center, width, height)`` tuple in
        ``[0, 1]``, or ``None`` if the box could not be estimated (e.g. an
        unreadable or completely static video).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Could not open video for motion estimate: %s", video_path.name)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    if total <= 0 or width <= 0 or height <= 0:
        cap.release()
        LOGGER.warning("Video reports invalid dimensions: %s", video_path.name)
        return None

    # Downscale for speed; the bbox is rescaled back to normalised coords.
    scale = 240.0 / max(height, 1)
    small_w = max(1, int(width * scale))
    small_h = max(1, int(height * scale))

    frames: List[np.ndarray] = []
    for fidx in _evenly_spaced_indices(total, sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32))
    cap.release()

    if len(frames) < 2:
        LOGGER.warning("Not enough readable frames for estimate: %s", video_path.name)
        return None

    stack = np.stack(frames, axis=0)
    std_map = stack.std(axis=0)
    if float(std_map.max()) <= 1e-6:
        # Completely static video: nothing moved.
        return None

    # Normalise and threshold the motion map.
    norm = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 1 or h <= 1:
        return None

    # Convert from the downscaled space back to normalised coordinates.
    xc = (x + w / 2.0) / small_w
    yc = (y + h / 2.0) / small_h
    nw = w / small_w
    nh = h / small_h
    return (xc, yc, nw, nh)


def extract_frames(
    video_path: Path, frames_dir: Path, frames_per_video: int
) -> List[Path]:
    """Extract evenly spaced frames from a video to disk as JPEGs.

    Args:
        video_path: Source video file.
        frames_dir: Destination directory for the JPEG frames.
        frames_per_video: Number of frames to export (typically 5-10).

    Returns:
        The list of written frame paths (possibly empty on read failure).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error("Could not open video: %s", video_path.name)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    written: List[Path] = []
    stem = video_path.stem
    for n, fidx in enumerate(_evenly_spaced_indices(total, frames_per_video)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = frames_dir / f"{stem}_f{n:02d}.jpg"
        if cv2.imwrite(str(out_path), frame):
            written.append(out_path)
    cap.release()
    return written


def run(
    input_dir: str,
    frames_dir: str,
    frames_per_video: int = 8,
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> List[dict]:
    """Run the full pre-labelling stage over a folder of videos.

    Args:
        input_dir: Folder containing the source videos.
        frames_dir: Folder where extracted frames are written.
        frames_per_video: Frames to export per video (clamped to >= 1).
        csv_path: Optional CSV output path (defaults to
            ``<frames_dir>/auto_labels.csv``).
        json_path: Optional JSON output path (defaults to
            ``<frames_dir>/auto_labels.json``).

    Returns:
        The list of suggestion records, one per exported frame.
    """
    frames_root = ensure_dir(frames_dir)
    videos = list_videos(input_dir)
    if not videos:
        LOGGER.warning("No videos found in %s", input_dir)
        return []

    frames_per_video = max(1, frames_per_video)
    csv_out = Path(csv_path) if csv_path else frames_root / "auto_labels.csv"
    json_out = Path(json_path) if json_path else frames_root / "auto_labels.json"

    records: List[dict] = []
    for video in videos:
        LOGGER.info("Processing %s", video.name)
        bbox = estimate_content_bbox(video)
        if bbox is None:
            # Elegant fallback: assume a centred 80%% content region.
            bbox = (0.5, 0.5, 0.8, 0.8)
            LOGGER.warning(
                "Heuristic failed for %s; using centred fallback box.", video.name
            )
        xc, yc, w, h = bbox

        frame_paths = extract_frames(video, frames_root, frames_per_video)
        for frame_path in frame_paths:
            records.append(
                {
                    "video": video.name,
                    "frame_file": frame_path.name,
                    "class": CLASS_CONTENT,
                    "x_center": round(xc, 6),
                    "y_center": round(yc, 6),
                    "width": round(w, 6),
                    "height": round(h, 6),
                }
            )

    _write_csv(csv_out, records)
    json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    LOGGER.info(
        "Wrote %d frame suggestions -> %s and %s",
        len(records),
        csv_out.name,
        json_out.name,
    )
    return records


def _write_csv(csv_out: Path, records: List[dict]) -> None:
    """Write suggestion ``records`` to ``csv_out`` (creates header even if empty)."""
    fieldnames = [
        "video",
        "frame_file",
        "class",
        "x_center",
        "y_center",
        "width",
        "height",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the pre-labelling stage."""
    parser = argparse.ArgumentParser(description="Auto pre-label videos for YOLO.")
    parser.add_argument("--input", required=True, help="Folder with source videos.")
    parser.add_argument(
        "--frames-dir", required=True, help="Output folder for extracted frames."
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Number of frames to export per video (5-10 recommended).",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--json", default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    run(
        input_dir=args.input,
        frames_dir=args.frames_dir,
        frames_per_video=args.frames_per_video,
        csv_path=args.csv,
        json_path=args.json,
    )


if __name__ == "__main__":
    main()
