import json
from pathlib import Path
from catcam.config import Config, load_config


def test_defaults_have_expected_values():
    c = Config()
    assert c.max_clips == 1000
    assert c.bowl_roi == (0.3, 0.3, 0.7, 0.7)
    assert c.fps == 10
    assert c.video_source == ""
    assert c.web_host == "127.0.0.1"


def test_derived_paths(tmp_path):
    c = Config(data_dir=str(tmp_path))
    assert c.clips_dir == tmp_path / "clips"
    assert c.training_dir == tmp_path / "training"
    assert c.db_path == tmp_path / "catcam.db"


def test_load_config_creates_default_file_when_missing(tmp_path):
    cfg_path = tmp_path / "config.json"
    c = load_config(cfg_path)
    assert cfg_path.exists()
    assert c.max_clips == 1000
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["max_clips"] == 1000


def test_load_config_reads_existing_overrides(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"max_clips": 5, "dwell_seconds": 2.0}))
    c = load_config(cfg_path)
    assert c.max_clips == 5
    assert c.dwell_seconds == 2.0
    # 未指定字段用默认
    assert c.fps == 10


def test_ai_label_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.json")
    assert cfg.ai_label_enabled is True   # 默认开（仍受 ai_api_key 把关，无 key 不动作）
    assert cfg.ai_base_url == "https://openrouter.ai/api/v1"
    assert cfg.ai_api_key == ""
    assert cfg.ai_model == "google/gemma-4-31b-it:free"
    assert cfg.ai_label_frames == 3


def test_ai_label_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    load_config(p)  # 生成默认
    raw = json.loads(p.read_text())
    raw["ai_label_enabled"] = True
    raw["ai_api_key"] = "sk-test"
    p.write_text(json.dumps(raw))
    cfg = load_config(p)
    assert cfg.ai_label_enabled is True
    assert cfg.ai_api_key == "sk-test"
