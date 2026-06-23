import numpy as np
import pytest
import cv2

from catcam.trainer import count_images, prepare_dataset, train_classifier


def _seed(training_dir, drinking_n, not_n):
    for cls, n in (("drinking", drinking_n), ("not_drinking", not_n)):
        d = training_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{cls}_{i}.jpg"), np.full((16, 16, 3), 100, np.uint8))


def test_count_images(tmp_path):
    _seed(tmp_path, 3, 2)
    assert count_images(tmp_path) == {"drinking": 3, "not_drinking": 2}


def test_prepare_dataset_splits_train_val(tmp_path):
    _seed(tmp_path, 8, 8)
    ds = tmp_path / "ds"
    counts = prepare_dataset(tmp_path, ds, val_ratio=0.25)
    assert counts == {"drinking": 8, "not_drinking": 8}
    for split in ("train", "val"):
        for cls in ("drinking", "not_drinking"):
            assert (ds / split / cls).is_dir()
            assert list((ds / split / cls).glob("*.jpg"))


def test_train_refuses_when_too_few(tmp_path):
    _seed(tmp_path, 1, 1)
    with pytest.raises(ValueError):
        train_classifier(tmp_path, tmp_path / "work", "yolov8n-cls.pt", 1, 32)
