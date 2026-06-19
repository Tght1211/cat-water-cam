from __future__ import annotations

from catcam.geometry import Rect


def filter_cat_boxes(
    detections: list[tuple[str, float, Rect]], confidence: float
) -> list[Rect]:
    return [box for name, conf, box in detections if name == "cat" and conf >= confidence]


class CatDetector:
    def __init__(self, model, confidence: float):
        self.model = model
        self.confidence = confidence

    @classmethod
    def from_path(cls, model_path: str, confidence: float) -> "CatDetector":
        from ultralytics import YOLO

        return cls(YOLO(model_path), confidence)

    def detect_cats(self, frame) -> list[Rect]:
        results = self.model(frame, verbose=False)
        detections: list[tuple[str, float, Rect]] = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                detections.append((self.model.names[cls], conf, (x1, y1, x2, y2)))
        return filter_cat_boxes(detections, self.confidence)
