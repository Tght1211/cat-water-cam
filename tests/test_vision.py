from catcam.vision import filter_cat_boxes, CatDetector


def test_filter_keeps_only_confident_cats():
    detections = [
        ("cat", 0.9, (0, 0, 10, 10)),
        ("dog", 0.95, (1, 1, 2, 2)),
        ("cat", 0.2, (3, 3, 4, 4)),     # 置信度不够
    ]
    boxes = filter_cat_boxes(detections, confidence=0.4)
    assert boxes == [(0, 0, 10, 10)]


def test_filter_empty():
    assert filter_cat_boxes([], confidence=0.4) == []


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: "cat", 1: "dog"}

    def __call__(self, frame, verbose=False):
        return [
            _FakeResult([
                _FakeBox(0, 0.9, (0.0, 0.0, 10.0, 10.0)),
                _FakeBox(1, 0.99, (1.0, 1.0, 2.0, 2.0)),
            ])
        ]


def test_cat_detector_with_fake_model():
    det = CatDetector(_FakeModel(), confidence=0.4)
    boxes = det.detect_cats(frame=None)
    assert boxes == [(0.0, 0.0, 10.0, 10.0)]
