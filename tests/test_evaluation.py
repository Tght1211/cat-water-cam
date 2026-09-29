import json

import numpy as np
import pytest

from catcam.evaluation import holdout_split, sample_group, classification_metrics, release_assessment
from catcam.models import ModelRegistry


def test_grouped_holdout_stays_fixed_as_labels_and_data_change(tmp_path):
    names = [f"clip_{1_800_000 * i + j}.mp4" for i in range(12) for j in (1, 2)]
    y = np.array([i % 2 for i in range(12) for _ in (1, 2)])
    path = tmp_path / "holdout.json"
    train, val = holdout_split(names, y, path)
    assert not {sample_group(names[i]) for i in train} & {sample_group(names[i]) for i in val}
    assert set(y[train]) == set(y[val]) == {0, 1}
    original = json.loads(path.read_text())["groups"]
    extra = names + ["new.mp4"]
    holdout_split(extra, np.append(y, 1), path)
    updated = json.loads(path.read_text())["groups"]
    assert all(updated[k] == v for k, v in original.items())
    train2, val2 = holdout_split(names[::-1], y[::-1], path)
    assert {names[i] for i in val} == {names[::-1][i] for i in val2}
    corrected = y.copy()
    corrected[0] = 1 - corrected[0]
    holdout_split(names, corrected, path)
    assert all(json.loads(path.read_text())["groups"][k] == v for k, v in original.items())


def test_single_session_cannot_be_its_own_validation(tmp_path):
    with pytest.raises(ValueError, match="跨时段"):
        holdout_split([f"clip_{i}.mp4" for i in range(8)], np.array([0, 1] * 4), tmp_path / "s.json")


def test_invalid_ratio(tmp_path):
    with pytest.raises(ValueError, match="val_ratio"):
        holdout_split([], np.array([]), tmp_path / "s.json", ratio=1)


class AlwaysNegative:
    def predict(self, x):
        return False, 0.01


def test_majority_accuracy_does_not_pass_release_gate():
    y = np.array([1] * 10 + [0] * 90)
    metrics = classification_metrics(AlwaysNegative(), np.zeros((100, 2)), y)
    assert metrics["top1"] == .9
    assert metrics["drinking_recall"] == 0
    assert metrics["balanced_accuracy"] == .5
    assert not release_assessment(metrics)["eligible"]


def test_perfect_but_tiny_validation_does_not_pass_release_gate():
    metrics = {"val_counts": {"drinking": 2, "not_drinking": 2},
               "drinking_recall": 1, "drinking_precision": 1, "balanced_accuracy": 1}
    assert not release_assessment(metrics)["eligible"]


def test_registry_blocks_unvalidated_video_gate_but_allows_shadow(tmp_path):
    reg = ModelRegistry(tmp_path / "r.json")
    reg.add(path="x.npz", top1=1, image_counts={}, label_counts={}, base="s3d+head",
            epochs=1, imgsz=224, created_ts=1)
    reg.set_active("v1", "shadow")
    with pytest.raises(ValueError, match="独立人工验证"):
        reg.set_active("v1", "gate")
    assert reg.active_mode() == "shadow"


def test_release_evidence_survives_restart(tmp_path):
    path = tmp_path / "r.json"
    reg = ModelRegistry(path)
    evidence = {"release": {"eligible": True, "reasons": []}}
    reg.add(path="x.npz", top1=1, image_counts={}, label_counts={}, base="s3d+head",
            epochs=1, imgsz=224, created_ts=1, evaluation=evidence)
    restored = ModelRegistry(path)
    assert restored.get("v1")["evaluation"] == evidence
    restored.set_active("v1", "gate")
