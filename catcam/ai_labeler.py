"""AI 自动标注：把一段 clip 的若干帧送给外部视觉大模型，判「猫是否真喝水」。

纯函数（encode_frames / build_messages / parse_label）不碰网络、可单测；
AILabeler 持有可注入的 OpenAI 兼容 client，测试塞假 client。
⚠️ 启用后画面帧会上传外部服务器——默认关闭，由 config.ai_label_enabled 显式开启。
"""
from __future__ import annotations

import base64
import json
import re
import tempfile
import time
from pathlib import Path

from catcam.feedback import extract_frames

PROMPT = (
    "你是猫咪饮水监控的标注助手。下面是同一段监控视频按时间顺序抽取的若干帧。"
    "请判断画面里的猫是否在【真的喝水】：舌头/嘴接触水面、低头舔水算喝水；"
    "只是凑近水碗、嗅闻、路过、趴着、洗脸都不算喝水。"
    "只输出一个 JSON 对象，不要任何多余文字，格式："
    '{"drinking": true 或 false, "confidence": 0到1的小数, "reason": "简短中文理由"}'
)


def encode_frames(frame_paths) -> list[str]:
    """把若干 jpg 文件读成 base64 data URL（OpenAI 兼容 image_url 用）。"""
    urls: list[str] = []
    for p in frame_paths:
        b = Path(p).read_bytes()
        urls.append("data:image/jpeg;base64," + base64.b64encode(b).decode())
    return urls


def build_messages(frames_b64: list[str]) -> list[dict]:
    """组装 vision 提示消息：一段提示词 + 若干张图。"""
    content: list[dict] = [{"type": "text", "text": PROMPT}]
    for url in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def parse_label(content: str) -> dict:
    """从模型回复里抠出标注 JSON，容忍 ```json 围栏与前后杂质；非法则 raise ValueError。"""
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError(f"模型回复里找不到 JSON：{content!r}")
    obj = json.loads(m.group(0))
    drinking = obj.get("drinking")
    if isinstance(drinking, str):
        drinking = drinking.strip().lower() in ("true", "yes", "1", "喝", "drinking")
    drinking = bool(drinking)
    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    reason = str(obj.get("reason", "") or "")
    return {"drinking": drinking, "confidence": conf, "reason": reason}
