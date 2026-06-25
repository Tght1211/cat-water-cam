import numpy as np
import pytest
import cv2

from catcam.trainer import count_images, prepare_dataset, train_classifier, training_progress, balance_target


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


def test_training_progress_basic():
    # 跑完 6/15 轮、耗时 12s → 进度 0.4，按 2s/轮 外推剩余 9 轮 = 18s
    p = training_progress(6, 15, 12.0)
    assert p["progress"] == pytest.approx(6 / 15)
    assert p["eta_seconds"] == pytest.approx(18.0)


def test_training_progress_edges():
    # 还没开始：没法估 ETA
    assert training_progress(0, 15, 0.0)["eta_seconds"] is None
    # 跑完最后一轮：进度封顶 1.0、不再外推
    last = training_progress(15, 15, 30.0)
    assert last["progress"] == 1.0
    assert last["eta_seconds"] is None
    # total 为 0：安全兜底
    assert training_progress(0, 0, 5.0) == {"progress": 0.0, "eta_seconds": None}


def test_balance_target_is_min():
    assert balance_target({"drinking": 6, "not_drinking": 30}) == 6
    assert balance_target({"drinking": 0, "not_drinking": 9}) == 0


def test_prepare_dataset_balances_train_keeps_val(tmp_path):
    # 8👍 / 40👎，val_ratio 0.25 → val: 2/10（原分布）；train 原本 6/30 → 平衡到 6/6
    _seed(tmp_path, 8, 40)
    ds = tmp_path / "ds"
    prepare_dataset(tmp_path, ds, val_ratio=0.25, balance=True)
    n_train = lambda c: len(list((ds / "train" / c).glob("*.jpg")))
    n_val = lambda c: len(list((ds / "val" / c).glob("*.jpg")))
    assert n_train("drinking") == n_train("not_drinking") == 6
    assert n_val("drinking") == 2 and n_val("not_drinking") == 10
