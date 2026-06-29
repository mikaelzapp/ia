"""Inference + rendering module (the critical stage).

Takes a new video and a trained model (``best.pt``) and produces a stabilised
9:16 (1080x1920) crop:

1. **Detection** - sample frames across the video and run YOLO. Both classes
   are detected, but the crop is *anchored strictly on class 1 (content)*.
2. **Stabilisation (static mode)** - gather the class-1 boxes from every
   sampled frame and take the mathematical **median** of ``[x, y, w, h]`` to
   define a single crop rectangle for the whole video, so the output never
   shakes.
3. **Rendering (MoviePy)** - crop and resize to 1080x1920, in either
   ``letterbox`` (black bars, exact aspect) or ``fill`` (zoom/cover) mode, and
   write the result with ``codec="libx264"`` + ``audio_codec="aac"`` so the
   original audio is preserved.

If class 1 is never found, a graceful centred-crop fallback is used and a
warning is logged instead of crashing.

Example:
    python -m src.inference \\
        --video input.mp4 --model models/autocrop/weights/best.pt \\
        --output output/input_916.mp4 --mode letterbox
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import ensure_dir, get_logger, select_device

LOGGER = get_logger(__name__)

#: Target output resolution (width, height) for 9:16 vertical video.
TARGET_W: int = 1080
TARGET_H: int = 1920

CLASS_CONTENT: int = 1


@dataclass(frozen=True)
class CropBox:
    """Integer pixel crop rectangle ``[x1, y1, x2, y2]`` within a frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Crop width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Crop height in pixels."""
        return self.y2 - self.y1


def _sample_frame_indices(total: int, sample_frames: int) -> List[int]:
    """Return evenly spaced frame indices used for detection."""
    if total <= 0:
        return []
    sample_frames = max(1, min(sample_frames, total))
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, sample_frames)})


def _read_sampled_frames(
    video_path: Path, sample_frames: int
) -> Tuple[List[np.ndarray], int, int]:
    """Read evenly spaced frames from ``video_path``.

    Args:
        video_path: Path to the source video.
        sample_frames: Approximate number of frames to read.

    Returns:
        ``(frames, width, height)`` where ``frames`` is a list of BGR arrays
        and ``width``/``height`` are the source dimensions (0 on failure).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    frames: List[np.ndarray] = []
    for fidx in _sample_frame_indices(total, sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames, width, height


def _detect_content_boxes(
    model: object,
    frames: List[np.ndarray],
    conf: float,
    device: str,
) -> List[Tuple[float, float, float, float]]:
    """Run YOLO on ``frames`` and return the best class-1 box per frame.

    For each frame the highest-confidence class-1 detection is kept (if any).

    Args:
        model: A loaded Ultralytics ``YOLO`` model.
        frames: BGR frames to run detection on.
        conf: Confidence threshold.
        device: Compute device string (``"cuda:0"`` / ``"cpu"``).

    Returns:
        A list of ``(x1, y1, x2, y2)`` pixel boxes, one per frame that had a
        class-1 detection.
    """
    boxes: List[Tuple[float, float, float, float]] = []
    if not frames:
        return boxes

    results = model.predict(  # type: ignore[attr-defined]
        frames, conf=conf, device=device, verbose=False
    )
    for res in results:
        det = getattr(res, "boxes", None)
        if det is None or det.xyxy is None or len(det) == 0:
            continue
        xyxy = det.xyxy.cpu().numpy()
        classes = det.cls.cpu().numpy().astype(int)
        confs = det.conf.cpu().numpy()

        content_mask = classes == CLASS_CONTENT
        if not content_mask.any():
            continue
        content_xyxy = xyxy[content_mask]
        content_conf = confs[content_mask]
        best = content_xyxy[int(np.argmax(content_conf))]
        boxes.append((float(best[0]), float(best[1]), float(best[2]), float(best[3])))
    return boxes


def _median_box(
    boxes: List[Tuple[float, float, float, float]], img_w: int, img_h: int
) -> CropBox:
    """Compute the stabilised median crop box from per-frame detections.

    The boxes are converted to centre/size form, the median of each component
    is taken across frames, and the result is clamped to the frame bounds.

    Args:
        boxes: Per-frame ``(x1, y1, x2, y2)`` detections (non-empty).
        img_w: Source frame width.
        img_h: Source frame height.

    Returns:
        A clamped integer :class:`CropBox`.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]

    mcx, mcy, mw, mh = (
        float(np.median(cx)),
        float(np.median(cy)),
        float(np.median(w)),
        float(np.median(h)),
    )
    x1 = mcx - mw / 2.0
    y1 = mcy - mh / 2.0
    x2 = mcx + mw / 2.0
    y2 = mcy + mh / 2.0
    return _clamp_box(x1, y1, x2, y2, img_w, img_h)


def _clamp_box(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> CropBox:
    """Clamp float corners to integer pixels inside ``[0, img_w] x [0, img_h]``.

    Guarantees a minimum 2px width/height so downstream resizing never divides
    by zero.
    """
    ix1 = int(max(0, min(round(x1), img_w - 2)))
    iy1 = int(max(0, min(round(y1), img_h - 2)))
    ix2 = int(max(ix1 + 2, min(round(x2), img_w)))
    iy2 = int(max(iy1 + 2, min(round(y2), img_h)))
    return CropBox(ix1, iy1, ix2, iy2)


def _center_fallback_box(img_w: int, img_h: int) -> CropBox:
    """Return a centred crop with the target 9:16 aspect ratio.

    Used when no content is detected. The largest centred rectangle matching
    ``TARGET_W:TARGET_H`` that fits inside the frame is selected.

    Args:
        img_w: Source frame width.
        img_h: Source frame height.

    Returns:
        A centred :class:`CropBox` with 9:16 aspect.
    """
    target_ratio = TARGET_W / TARGET_H
    frame_ratio = img_w / img_h
    if frame_ratio > target_ratio:
        # Frame is wider than 9:16 -> limit by height.
        crop_h = img_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = img_w
        crop_h = int(round(crop_w / target_ratio))
    x1 = (img_w - crop_w) // 2
    y1 = (img_h - crop_h) // 2
    return _clamp_box(x1, y1, x1 + crop_w, y1 + crop_h, img_w, img_h)


def compute_crop_box(
    model: object,
    video_path: Path,
    conf: float,
    sample_frames: int,
    device: str,
) -> Tuple[CropBox, bool]:
    """Compute the single stabilised crop box for a whole video.

    Args:
        model: Loaded YOLO model.
        video_path: Source video.
        conf: Detection confidence threshold.
        sample_frames: Number of frames to sample for detection.
        device: Compute device.

    Returns:
        ``(crop_box, used_fallback)``. ``used_fallback`` is ``True`` when no
        class-1 detection was found and the centred fallback was used.

    Raises:
        RuntimeError: If the video cannot be read at all.
    """
    frames, img_w, img_h = _read_sampled_frames(video_path, sample_frames)
    if not frames or img_w == 0 or img_h == 0:
        raise RuntimeError(f"Could not read frames from {video_path}.")

    boxes = _detect_content_boxes(model, frames, conf=conf, device=device)
    if not boxes:
        LOGGER.warning(
            "No class-1 (content) detected in %s; using centred fallback crop.",
            video_path.name,
        )
        return _center_fallback_box(img_w, img_h), True

    LOGGER.info(
        "Stabilising crop from %d/%d frames with content detections.",
        len(boxes),
        len(frames),
    )
    return _median_box(boxes, img_w, img_h), False


def _render(
    video_path: Path,
    output_path: Path,
    crop: CropBox,
    mode: str,
    fps: Optional[float],
) -> None:
    """Crop, resize and write the final 9:16 video using MoviePy.

    Args:
        video_path: Source video (audio is read from here).
        output_path: Destination ``.mp4``.
        crop: Stabilised crop rectangle.
        mode: ``"letterbox"`` (pad to keep aspect) or ``"fill"`` (cover/zoom).
        fps: Output FPS; ``None`` keeps the source FPS.

    Raises:
        ValueError: If ``mode`` is not ``"letterbox"`` or ``"fill"``.
    """
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import crop as mp_crop
    from moviepy.video.fx.all import margin as mp_margin
    from moviepy.video.fx.all import resize as mp_resize

    if mode not in ("letterbox", "fill"):
        raise ValueError(f"Unknown mode {mode!r}; use 'letterbox' or 'fill'.")

    clip = VideoFileClip(str(video_path))
    try:
        cropped = mp_crop(clip, x1=crop.x1, y1=crop.y1, x2=crop.x2, y2=crop.y2)
        cw, ch = crop.width, crop.height

        if mode == "letterbox":
            scale = min(TARGET_W / cw, TARGET_H / ch)
            new_w = max(2, int(round(cw * scale)))
            new_h = max(2, int(round(ch * scale)))
            resized = mp_resize(cropped, newsize=(new_w, new_h))
            pad_w = TARGET_W - new_w
            pad_h = TARGET_H - new_h
            left = pad_w // 2
            top = pad_h // 2
            final = mp_margin(
                resized,
                left=left,
                right=pad_w - left,
                top=top,
                bottom=pad_h - top,
                color=(0, 0, 0),
            )
        else:  # fill / zoom (cover)
            scale = max(TARGET_W / cw, TARGET_H / ch)
            new_w = max(TARGET_W, int(round(cw * scale)))
            new_h = max(TARGET_H, int(round(ch * scale)))
            resized = mp_resize(cropped, newsize=(new_w, new_h))
            final = mp_crop(
                resized,
                width=TARGET_W,
                height=TARGET_H,
                x_center=new_w / 2.0,
                y_center=new_h / 2.0,
            )

        ensure_dir(output_path.parent)
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            threads=4,
            logger=None,
        )
    finally:
        clip.close()


def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    mode: str = "letterbox",
    conf: float = 0.25,
    sample_frames: int = 60,
    fps: Optional[float] = None,
    prefer_gpu: bool = True,
) -> Path:
    """Run the full detection -> stabilisation -> render pipeline for one video.

    Args:
        video_path: Source video path.
        model_path: Trained YOLO weights (``best.pt``).
        output_path: Output ``.mp4`` path.
        mode: ``"letterbox"`` or ``"fill"``.
        conf: Detection confidence threshold.
        sample_frames: Frames to sample for detection/stabilisation.
        fps: Output FPS (``None`` keeps source FPS).
        prefer_gpu: Use CUDA when available.

    Returns:
        The output path that was written.

    Raises:
        FileNotFoundError: If the video or the model file does not exist.
    """
    from ultralytics import YOLO

    video = Path(video_path)
    model_file = Path(model_path)
    out = Path(output_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_file}")

    device = select_device(prefer_gpu)
    LOGGER.info("Loading model %s on %s", model_file.name, device)
    model = YOLO(str(model_file))

    crop, used_fallback = compute_crop_box(
        model, video, conf=conf, sample_frames=sample_frames, device=device
    )
    LOGGER.info(
        "Crop box [x1=%d y1=%d x2=%d y2=%d]%s -> rendering (%s).",
        crop.x1,
        crop.y1,
        crop.x2,
        crop.y2,
        " (fallback)" if used_fallback else "",
        mode,
    )
    _render(video, out, crop, mode=mode, fps=fps)
    LOGGER.info("Saved cropped video -> %s", out)
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Detect, stabilise and crop a video to 9:16.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", required=True, help="Trained YOLO weights (best.pt).")
    parser.add_argument("--output", required=True, help="Output .mp4 path.")
    parser.add_argument(
        "--mode",
        choices=("letterbox", "fill"),
        default="letterbox",
        help="Resize strategy for 1080x1920 output.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument(
        "--sample-frames", type=int, default=60, help="Frames sampled for stabilisation."
    )
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (default: source).")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    process_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        mode=args.mode,
        conf=args.conf,
        sample_frames=args.sample_frames,
        fps=args.fps,
        prefer_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
