"""Shared utilities for the video auto-crop pipeline.

This module centralises helpers that are reused across the pre-labelling,
curation, training and inference stages: logging configuration, device
selection (GPU/CPU), filesystem helpers and conversions between pixel
bounding boxes and the normalised YOLO label format.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Class definitions (shared by every module so they never drift apart).
# --------------------------------------------------------------------------- #
CLASS_INTERFACE: int = 0
CLASS_CONTENT: int = 1
CLASS_NAMES: dict[int, str] = {CLASS_INTERFACE: "interface", CLASS_CONTENT: "content"}

#: Video container extensions that the pipeline knows how to read.
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

#: Frame image extensions written/read by the pipeline.
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def get_logger(name: str) -> logging.Logger:
    """Return a process-wide configured :class:`logging.Logger`.

    The logger writes human readable, timestamped lines to ``stdout`` and is
    safe to call repeatedly (handlers are only attached once).

    Args:
        name: Logger name, conventionally ``__name__`` of the caller.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def select_device(prefer_gpu: bool = True) -> str:
    """Pick the best available compute device for Ultralytics/torch.

    Prefers CUDA when available and ``prefer_gpu`` is set, otherwise falls
    back to CPU. The function imports :mod:`torch` lazily so that modules that
    do not need a model (e.g. the curation tool) can run without torch warming
    up the CUDA context.

    Args:
        prefer_gpu: When ``True`` (default) use CUDA if it is available.

    Returns:
        ``"cuda:0"`` when a CUDA device should be used, otherwise ``"cpu"``.
    """
    if not prefer_gpu:
        return "cpu"
    try:
        import torch  # Imported lazily on purpose.

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover - torch missing or broken CUDA stack.
        pass
    return "cpu"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``.

    Args:
        path: Directory to create.

    Returns:
        The created directory as a :class:`pathlib.Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(
    folder: str | os.PathLike[str], extensions: Sequence[str]
) -> List[Path]:
    """List files in ``folder`` whose suffix matches ``extensions``.

    Args:
        folder: Directory to scan (non-recursive).
        extensions: Allowed lowercase suffixes, e.g. ``(".mp4", ".mov")``.

    Returns:
        A sorted list of matching file paths. Empty when the folder is
        missing or contains no matches.
    """
    root = Path(folder)
    if not root.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def list_videos(folder: str | os.PathLike[str]) -> List[Path]:
    """Return the video files found directly inside ``folder``."""
    return list_files(folder, VIDEO_EXTENSIONS)


def list_images(folder: str | os.PathLike[str]) -> List[Path]:
    """Return the image files found directly inside ``folder``."""
    return list_files(folder, IMAGE_EXTENSIONS)


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in absolute pixel coordinates.

    The corners are stored as ``(x1, y1)`` top-left and ``(x2, y2)``
    bottom-right with ``x2 >= x1`` and ``y2 >= y1`` guaranteed by
    :meth:`normalised_corners`.

    Attributes:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        cls: Class index (see :data:`CLASS_NAMES`).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int

    def normalised_corners(self) -> Tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)`` with the corners correctly ordered."""
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        """Box width in pixels."""
        x1, _, x2, _ = self.normalised_corners()
        return x2 - x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        _, y1, _, y2 = self.normalised_corners()
        return y2 - y1

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[int, float, float, float, float]:
        """Convert to the normalised YOLO tuple ``(cls, xc, yc, w, h)``.

        Args:
            img_w: Source image width in pixels (must be > 0).
            img_h: Source image height in pixels (must be > 0).

        Returns:
            ``(cls, x_center, y_center, width, height)`` with the four
            geometric values normalised to ``[0, 1]``.
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError("Image dimensions must be positive.")
        x1, y1, x2, y2 = self.normalised_corners()
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return (self.cls, _clip01(xc), _clip01(yc), _clip01(w), _clip01(h))

    @classmethod
    def from_yolo(
        cls,
        class_id: int,
        xc: float,
        yc: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> "BBox":
        """Build a pixel :class:`BBox` from a normalised YOLO label.

        Args:
            class_id: Class index.
            xc: Normalised x centre in ``[0, 1]``.
            yc: Normalised y centre in ``[0, 1]``.
            w: Normalised width in ``[0, 1]``.
            h: Normalised height in ``[0, 1]``.
            img_w: Target image width in pixels.
            img_h: Target image height in pixels.

        Returns:
            The equivalent :class:`BBox` in pixel coordinates.
        """
        bw = w * img_w
        bh = h * img_h
        cx = xc * img_w
        cy = yc * img_h
        return cls(
            x1=cx - bw / 2.0,
            y1=cy - bh / 2.0,
            x2=cx + bw / 2.0,
            y2=cy + bh / 2.0,
            cls=class_id,
        )


def _clip01(value: float) -> float:
    """Clamp ``value`` into the closed interval ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def write_yolo_label(
    label_path: str | os.PathLike[str], boxes: Iterable[Tuple[int, float, float, float, float]]
) -> None:
    """Write YOLO label lines to ``label_path``.

    Each box is written as ``"cls xc yc w h"`` with six decimal places.
    Writing an empty iterable creates an empty file, which YOLO interprets as
    a valid "background" image with no objects.

    Args:
        label_path: Destination ``.txt`` path.
        boxes: Iterable of ``(cls, xc, yc, w, h)`` normalised tuples.
    """
    lines = [
        f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for cls, xc, yc, w, h in boxes
    ]
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(
    label_path: str | os.PathLike[str],
) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO label lines from ``label_path``.

    Args:
        label_path: Source ``.txt`` path. A missing file yields an empty list.

    Returns:
        A list of ``(cls, xc, yc, w, h)`` tuples.
    """
    path = Path(label_path)
    if not path.is_file():
        return []
    boxes: List[Tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        boxes.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return boxes
