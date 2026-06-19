from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    camera_index: int = 0
    # 视频源：留空则用 camera_index 摄像头；填一个视频文件路径则改用该文件
    # （方便还没接摄像头时，用一段录好的视频测试识别/录制/标注全流程）。
    video_source: str = ""
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 10
    # 水碗区域，0-1 比例 (x1, y1, x2, y2)，与分辨率无关
    bowl_roi: tuple[float, float, float, float] = (0.3, 0.3, 0.7, 0.7)
    # 猫框与水碗框的交集面积 / 水碗面积 >= 此值才算「猫在水碗」
    min_overlap_ratio: float = 0.15
    dwell_seconds: float = 3.0
    cooldown_seconds: float = 60.0
    clip_seconds: float = 4.0
    max_clips: int = 10
    yolo_model: str = "yolov8n.pt"
    cat_confidence: float = 0.4
    data_dir: str = "data"
    # 默认只绑定本机回环：用户远程进 Mac Mini 后用 localhost 访问，画面不出本机。
    # 如需局域网内其它设备直接访问，可在 config.json 显式改为 "0.0.0.0"（会暴露给整个局域网）。
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    @property
    def clips_dir(self) -> Path:
        return Path(self.data_dir) / "clips"

    @property
    def training_dir(self) -> Path:
        return Path(self.data_dir) / "training"

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "catcam.db"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        c = Config()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(c), indent=2))
        return c
    raw = json.loads(path.read_text())
    known = {f.name for f in fields(Config)}
    data = {k: v for k, v in raw.items() if k in known}
    if "bowl_roi" in data:
        data["bowl_roi"] = tuple(data["bowl_roi"])
    return Config(**data)
