"""Video auto-crop pipeline (YOLOv8 + OpenCV + MoviePy).

Modules:
    auto_label:     frame extraction + motion-heuristic pre-labelling.
    curation_tool:  pure-OpenCV GUI for human label curation.
    train_pipeline: dataset assembly + YOLOv8n training with early stopping.
    inference:      detection, median stabilisation and 9:16 rendering.
"""

__all__ = ["auto_label", "curation_tool", "train_pipeline", "inference", "utils"]
