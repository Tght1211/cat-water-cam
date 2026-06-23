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
    max_clips: int = 100
    # 整段会话录制：从猫开始喝到离开都录下来（变长），而非固定 clip_seconds。
    record_session: bool = True
    preroll_seconds: float = 3.0          # 触发前补录这么久（猫凑近的过程）
    session_end_grace_seconds: float = 3.0  # 猫离开持续这么久才算结束、收尾存盘
    max_session_seconds: float = 90.0      # 单段封顶，防卡死无限录
    yolo_model: str = "yolov8n.pt"
    cat_confidence: float = 0.4
    data_dir: str = "data"
    # 默认只绑定本机回环：用户远程进 Mac Mini 后用 localhost 访问，画面不出本机。
    # 如需局域网内其它设备直接访问，可在 config.json 显式改为 "0.0.0.0"（会暴露给整个局域网）。
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    # 邮件提醒：猫来喝水时发一封带照片+今日次数+周/月趋势的 HTML 邮件。
    # 凭据写在 config.json（已 gitignore，不进库）。SMTP 密码是邮箱「授权码」，不是登录密码。
    mail_enabled: bool = False
    smtp_host: str = "smtp.qq.com"
    smtp_port: int = 465  # QQ 邮箱 SSL
    smtp_user: str = ""   # 发件邮箱（如 2890549308@qq.com）
    smtp_password: str = ""  # 授权码
    mail_to: str = ""     # 收件邮箱
    # 限流：同一时间窗内最多发一封，防止误判刷屏（模型没训好时尤其重要）。
    mail_min_interval_seconds: float = 600.0

    # 模型训练（网页一键）：用 👍/👎 标注帧训练「真喝水/没喝」分类器。
    cls_base_model: str = "yolov8n-cls.pt"
    train_epochs: int = 15
    train_imgsz: int = 96

    # 夜间/弱光（此摄像头无红外）：整体亮度低于阈值算「暗」，暗帧做弱光增强后
    # 用「画面变化」识别行为。record_at_night=False 则天黑直接不记录。
    night_brightness_threshold: float = 50.0
    record_at_night: bool = True

    # 检测节奏：每隔这么久跑一次 YOLO 识别（与采集/预览解耦，预览始终流畅）。
    detect_interval_seconds: float = 0.2

    @property
    def clips_dir(self) -> Path:
        return Path(self.data_dir) / "clips"

    @property
    def training_dir(self) -> Path:
        return Path(self.data_dir) / "training"

    @property
    def models_dir(self) -> Path:
        return Path(self.data_dir) / "models"

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
