from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
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
    # 最多保留多少段视频；超量时优先删最旧的「没喝」，喝水/未判定永不自动删（见 recorder.prune_dir）。
    max_clips: int = 1000
    # 整段会话录制：从猫开始喝到离开都录下来（变长），而非固定 clip_seconds。
    record_session: bool = True
    preroll_seconds: float = 3.0          # 触发前补录这么久（猫凑近的过程）
    session_end_grace_seconds: float = 3.0  # 猫离开持续这么久才算结束、收尾存盘
    max_session_seconds: float = 15.0      # 单段封顶，防卡死无限录；超过 15s 没有更多信息量
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

    # AI 自动标注 / 整段裁判（外部视觉大模型，OpenRouter / OpenAI 兼容）：录一段自动判「喝/没喝」，
    # 它是「发邮件 + 记次数」的唯一权威，并把判定写进训练数据。
    # 默认开：仍受 ai_api_key 把关——没填 key 则不动作、不上传。
    # ⚠️ 填了 key 开始判定后，猫的画面帧会上传外部视觉大模型，违背「画面只在本机/局域网」原则。
    ai_label_enabled: bool = True
    ai_base_url: str = "https://openrouter.ai/api/v1"
    ai_api_key: str = ""                              # OpenRouter key，写在 config.json（已 gitignore）
    ai_model: str = "google/gemma-4-31b-it:free"      # 主模型
    # 兜底/轮换模型：连同主模型一起组成模型池，每次调用轮换起始模型（分摊各免费模型额度），
    # 某个失败（限流/报错）就顺位换下一个，全失败才算这段标注失败。都要支持视觉输入。
    ai_fallback_models: list[str] = field(default_factory=list)
    ai_label_frames: int = 3                          # 每段送几帧给模型

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
