"""Dataset assembly + YOLOv8 training module.

This stage turns the annotated frames produced by the curation tool into a
ready-to-train YOLO dataset and launches training:

1. Collect every frame that has a sibling ``.txt`` label.
2. Shuffle and split into ``train`` (80%) and ``val`` (20%).
3. Lay out the canonical Ultralytics directory structure
   (``images/{train,val}`` + ``labels/{train,val}``) and copy files in.
4. Generate ``dataset.yaml``.
5. Fine-tune ``yolov8n.pt`` with early stopping (``patience``) to curb
   overfitting on small custom datasets.

Example:
    python -m src.train_pipeline \\
        --frames-dir data/frames \\
        --dataset-dir data/dataset \\
        --epochs 100 --patience 20
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .utils import (
    CLASS_NAMES,
    ensure_dir,
    get_logger,
    list_images,
    select_device,
)

LOGGER = get_logger(__name__)


def _collect_labelled_frames(frames_dir: Path) -> List[Tuple[Path, Path]]:
    """Return ``(image, label)`` pairs for frames that have a ``.txt`` label.

    Args:
        frames_dir: Folder containing frames and their YOLO labels.

    Returns:
        A list of ``(image_path, label_path)`` tuples. Frames without a
        matching label file are skipped.
    """
    pairs: List[Tuple[Path, Path]] = []
    for image_path in list_images(frames_dir):
        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            pairs.append((image_path, label_path))
    return pairs


def _split(
    pairs: List[Tuple[Path, Path]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Shuffle and split ``pairs`` into (train, val).

    At least one sample is always kept in each split when possible so YOLO
    never receives an empty validation set.

    Args:
        pairs: Labelled ``(image, label)`` pairs.
        val_ratio: Fraction reserved for validation (e.g. ``0.2``).
        seed: RNG seed for a reproducible split.

    Returns:
        ``(train_pairs, val_pairs)``.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1:
        n_val = max(1, min(n_val, len(shuffled) - 1))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def _materialise_split(
    pairs: List[Tuple[Path, Path]], images_dir: Path, labels_dir: Path
) -> None:
    """Copy ``(image, label)`` pairs into the YOLO image/label folders."""
    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


def build_dataset(
    frames_dir: str,
    dataset_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Build the YOLO dataset folder and ``dataset.yaml`` from labelled frames.

    Args:
        frames_dir: Folder with annotated frames + ``.txt`` labels.
        dataset_dir: Destination root for the YOLO dataset.
        val_ratio: Validation fraction (default ``0.2`` => 80/20 split).
        seed: RNG seed for the split.

    Returns:
        The path to the generated ``dataset.yaml``.

    Raises:
        FileNotFoundError: If no labelled frames are found.
    """
    frames_root = Path(frames_dir)
    pairs = _collect_labelled_frames(frames_root)
    if not pairs:
        raise FileNotFoundError(
            f"No labelled frames (.txt) found in {frames_dir!r}. "
            "Run the curation tool first."
        )

    train, val = _split(pairs, val_ratio=val_ratio, seed=seed)
    LOGGER.info("Dataset split: %d train / %d val", len(train), len(val))

    root = ensure_dir(dataset_dir)
    _materialise_split(train, root / "images" / "train", root / "labels" / "train")
    _materialise_split(val, root / "images" / "val", root / "labels" / "val")

    yaml_path = root / "dataset.yaml"
    config = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in CLASS_NAMES.items()},
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    LOGGER.info("Wrote dataset config -> %s", yaml_path)
    return yaml_path


def train(
    dataset_yaml: str,
    epochs: int = 100,
    patience: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    base_model: str = "yolov8n.pt",
    project: str = "models",
    name: str = "autocrop",
    prefer_gpu: bool = True,
) -> Optional[Path]:
    """Train YOLOv8n on the prepared dataset with early stopping.

    Args:
        dataset_yaml: Path to the generated ``dataset.yaml``.
        epochs: Maximum number of training epochs.
        patience: Early-stopping patience (epochs without val improvement).
            This is the main guard against overfitting on small datasets.
        imgsz: Training image size.
        batch: Batch size.
        base_model: Pretrained checkpoint to fine-tune (nano by default).
        project: Output project directory for runs.
        name: Run name; weights land in ``<project>/<name>/weights/best.pt``.
        prefer_gpu: Use CUDA when available, else CPU.

    Returns:
        Path to ``best.pt`` if it exists after training, otherwise ``None``.
    """
    from ultralytics import YOLO  # Imported lazily to keep CLI startup fast.

    device = select_device(prefer_gpu)
    LOGGER.info("Training on device: %s", device)

    model = YOLO(base_model)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,  # early stopping
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )

    # Resolve the real output directory from the trainer rather than guessing,
    # since Ultralytics may nest runs under its configured runs directory.
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / name))
    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        LOGGER.info("Training complete. Best weights: %s", best)
        return best
    LOGGER.warning("Training finished but best.pt was not found at %s", best)
    return None


def run(
    frames_dir: str,
    dataset_dir: str,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    val_ratio: float,
    base_model: str,
    project: str,
    name: str,
    prefer_gpu: bool,
) -> Optional[Path]:
    """Build the dataset and run training end-to-end."""
    yaml_path = build_dataset(
        frames_dir=frames_dir, dataset_dir=dataset_dir, val_ratio=val_ratio
    )
    return train(
        dataset_yaml=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        base_model=base_model,
        project=project,
        name=name,
        prefer_gpu=prefer_gpu,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Build dataset and train YOLOv8n.")
    parser.add_argument("--frames-dir", required=True, help="Annotated frames folder.")
    parser.add_argument(
        "--dataset-dir", default="data/dataset", help="Output dataset folder."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience", type=int, default=20, help="Early-stopping patience."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--project", default="models")
    parser.add_argument("--name", default="autocrop")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    run(
        frames_dir=args.frames_dir,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        val_ratio=args.val_ratio,
        base_model=args.base_model,
        project=args.project,
        name=args.name,
        prefer_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
