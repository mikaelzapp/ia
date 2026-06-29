"""Human curation tool (pure OpenCV GUI).

A deliberately dependency-light annotation tool built only on top of OpenCV's
``highgui`` window + mouse/keyboard callbacks. It loads the frames extracted by
:mod:`src.auto_label`, pre-populates the suggested content box, and lets a human
draw/adjust the interface (class 0) and content (class 1) rectangles before
exporting normalised YOLO ``.txt`` labels next to each frame.

Controls
--------
* **Left mouse drag** : draw a new rectangle using the active class.
* **0**               : set active class to *interface* (class 0).
* **1**               : set active class to *content* (class 1).
* **R**               : reset (remove) all boxes on the current frame.
* **U**               : undo the last drawn box.
* **SPACE**           : save the current frame's labels and go to the next.
* **A / D**           : previous / next frame without forcing a save.
* **Q** or **ESC**    : quit the tool.

Each saved frame produces a sibling ``<frame>.txt`` file in YOLO format
(``cls x_center y_center width height`` normalised to ``[0, 1]``).

Example:
    python -m src.curation_tool --frames-dir data/frames
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .utils import (
    CLASS_CONTENT,
    CLASS_INTERFACE,
    CLASS_NAMES,
    BBox,
    get_logger,
    list_images,
    read_yolo_label,
    write_yolo_label,
)

LOGGER = get_logger(__name__)

_WINDOW: str = "Curation Tool  |  0=interface 1=content  R=reset U=undo  SPACE=save+next  Q=quit"

#: BGR colours per class for drawing.
_COLORS: Dict[int, Tuple[int, int, int]] = {
    CLASS_INTERFACE: (0, 165, 255),  # orange
    CLASS_CONTENT: (0, 255, 0),  # green
}

#: Largest window edge in pixels; frames are scaled to fit and boxes are
#: converted back to full-resolution coordinates on save.
_MAX_DISPLAY: int = 1100


class _FrameState:
    """Mutable per-frame annotation state used by the OpenCV mouse callback.

    Attributes:
        boxes: Current list of :class:`BBox` in *display* pixel coordinates.
        active_class: Class index applied to the next drawn box.
        drawing: Whether a drag is currently in progress.
        start: Drag origin in display coordinates.
        cursor: Current cursor position in display coordinates.
    """

    def __init__(self) -> None:
        self.boxes: List[BBox] = []
        self.active_class: int = CLASS_CONTENT
        self.drawing: bool = False
        self.start: Tuple[int, int] = (0, 0)
        self.cursor: Tuple[int, int] = (0, 0)


def _load_suggestions(frames_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Load auto-label suggestions keyed by frame file name.

    Looks for ``auto_labels.json`` first, then ``auto_labels.csv``. Returns an
    empty mapping when neither exists.

    Args:
        frames_dir: Folder containing the frames and suggestion file.

    Returns:
        Mapping ``frame_file -> [(cls, xc, yc, w, h), ...]`` (normalised).
    """
    suggestions: Dict[str, List[Tuple[int, float, float, float, float]]] = {}

    json_path = frames_dir / "auto_labels.json"
    csv_path = frames_dir / "auto_labels.csv"

    records: List[dict] = []
    if json_path.is_file():
        records = json.loads(json_path.read_text(encoding="utf-8"))
    elif csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))

    for rec in records:
        frame_file = str(rec["frame_file"])
        entry = (
            int(float(rec["class"])),
            float(rec["x_center"]),
            float(rec["y_center"]),
            float(rec["width"]),
            float(rec["height"]),
        )
        suggestions.setdefault(frame_file, []).append(entry)
    return suggestions


def _compute_scale(img_w: int, img_h: int) -> float:
    """Return a scale factor so the largest image edge fits ``_MAX_DISPLAY``."""
    longest = max(img_w, img_h)
    return min(1.0, _MAX_DISPLAY / longest) if longest > 0 else 1.0


def _make_mouse_callback(state: _FrameState):
    """Create an OpenCV mouse callback bound to ``state``."""

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.start = (x, y)
            state.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            state.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            x0, y0 = state.start
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                state.boxes.append(
                    BBox(x1=x0, y1=y0, x2=x, y2=y, cls=state.active_class)
                )

    return _on_mouse


def _render(canvas, state: _FrameState):
    """Draw all stored boxes plus the in-progress rubber band onto ``canvas``."""
    view = canvas.copy()
    for box in state.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.normalised_corners())
        color = _COLORS.get(box.cls, (255, 255, 255))
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            view,
            CLASS_NAMES.get(box.cls, str(box.cls)),
            (x1 + 3, max(15, y1 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if state.drawing:
        color = _COLORS.get(state.active_class, (255, 255, 255))
        cv2.rectangle(view, state.start, state.cursor, color, 1)

    active_name = CLASS_NAMES.get(state.active_class, str(state.active_class))
    cv2.putText(
        view,
        f"active: {active_name}  (boxes: {len(state.boxes)})",
        (8, view.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _COLORS.get(state.active_class, (255, 255, 255)),
        2,
        cv2.LINE_AA,
    )
    return view


def _save_labels(frame_path: Path, state: _FrameState, scale: float, img_w: int, img_h: int) -> None:
    """Persist the current boxes as a YOLO ``.txt`` next to ``frame_path``.

    Display coordinates are converted back to full-resolution pixels (dividing
    by ``scale``) before normalising.

    Args:
        frame_path: The image being annotated.
        state: Current annotation state.
        scale: Display-to-original scale factor that was applied.
        img_w: Original image width in pixels.
        img_h: Original image height in pixels.
    """
    yolo_boxes = []
    for box in state.boxes:
        full = BBox(
            x1=box.x1 / scale,
            y1=box.y1 / scale,
            x2=box.x2 / scale,
            y2=box.y2 / scale,
            cls=box.cls,
        )
        yolo_boxes.append(full.to_yolo(img_w, img_h))
    label_path = frame_path.with_suffix(".txt")
    write_yolo_label(label_path, yolo_boxes)
    LOGGER.info("Saved %d boxes -> %s", len(yolo_boxes), label_path.name)


def _initial_boxes(
    frame_path: Path,
    suggestions: Dict[str, List[Tuple[int, float, float, float, float]]],
    scale: float,
    img_w: int,
    img_h: int,
) -> List[BBox]:
    """Build the starting boxes for a frame in display coordinates.

    Existing ``.txt`` labels take priority (resuming work); otherwise the
    auto-label suggestions are used.
    """
    existing = read_yolo_label(frame_path.with_suffix(".txt"))
    source = existing if existing else suggestions.get(frame_path.name, [])
    boxes: List[BBox] = []
    for cls, xc, yc, w, h in source:
        full = BBox.from_yolo(cls, xc, yc, w, h, img_w, img_h)
        boxes.append(
            BBox(
                x1=full.x1 * scale,
                y1=full.y1 * scale,
                x2=full.x2 * scale,
                y2=full.y2 * scale,
                cls=cls,
            )
        )
    return boxes


def run(frames_dir: str) -> None:
    """Launch the interactive curation loop over every frame in ``frames_dir``.

    Args:
        frames_dir: Folder containing the extracted frames (and optional
            ``auto_labels.json``/``.csv`` suggestions).

    Raises:
        FileNotFoundError: If the folder has no image frames.
    """
    root = Path(frames_dir)
    frames = list_images(root)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir!r}.")

    suggestions = _load_suggestions(root)
    LOGGER.info(
        "Loaded %d frames (%d with suggestions).", len(frames), len(suggestions)
    )

    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)

    idx = 0
    while 0 <= idx < len(frames):
        frame_path = frames[idx]
        image = cv2.imread(str(frame_path))
        if image is None:
            LOGGER.warning("Skipping unreadable frame: %s", frame_path.name)
            idx += 1
            continue

        img_h, img_w = image.shape[:2]
        scale = _compute_scale(img_w, img_h)
        canvas = cv2.resize(
            image, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA
        )

        state = _FrameState()
        state.boxes = _initial_boxes(frame_path, suggestions, scale, img_w, img_h)
        cv2.setMouseCallback(_WINDOW, _make_mouse_callback(state))

        LOGGER.info("[%d/%d] %s", idx + 1, len(frames), frame_path.name)

        advance = 0  # -1 prev, +1 next, 0 stay.
        while advance == 0:
            cv2.imshow(_WINDOW, _render(canvas, state))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                cv2.destroyAllWindows()
                LOGGER.info("Curation aborted by user.")
                return
            elif key == ord("0"):
                state.active_class = CLASS_INTERFACE
            elif key == ord("1"):
                state.active_class = CLASS_CONTENT
            elif key == ord("r"):
                state.boxes.clear()
            elif key == ord("u") and state.boxes:
                state.boxes.pop()
            elif key == ord(" "):  # save + next
                _save_labels(frame_path, state, scale, img_w, img_h)
                advance = 1
            elif key == ord("d"):  # next without forcing save
                advance = 1
            elif key == ord("a"):  # previous
                advance = -1

        idx += advance

    cv2.destroyAllWindows()
    LOGGER.info("Curation complete.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the curation tool."""
    parser = argparse.ArgumentParser(description="OpenCV curation tool for YOLO labels.")
    parser.add_argument(
        "--frames-dir", required=True, help="Folder with frames to annotate."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    run(frames_dir=args.frames_dir)


if __name__ == "__main__":
    main()
