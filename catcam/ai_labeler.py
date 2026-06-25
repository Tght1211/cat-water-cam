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


class AILabeler:
    """录一段调一次：抽帧 → 送模型 → 解析 → 写标注（source='ai'）+ 抽帧进训练目录。

    client 可注入（测试塞假 client）；失败抛异常由调用方吞掉记日志，绝不影响录制。
    人工 > AI：已人工标注的段直接跳过、不覆盖。
    """

    def __init__(self, store, training_dir, model, frames=3, client=None,
                 train_frames=5, sleep=time.sleep, log=print):
        self.store = store
        self.training_dir = Path(training_dir)
        self.model = model
        self.frames = frames
        self.client = client
        self.train_frames = train_frames
        self.sleep = sleep
        self.log = log

    @classmethod
    def from_config(cls, store, cfg):
        """按 config 造一个真 client 的 AILabeler；未启用/缺 key 返回 None。"""
        if not cfg.ai_label_enabled or not cfg.ai_api_key:
            return None
        import httpx
        from openai import OpenAI
        # trust_env=False：忽略环境里的 VPN/SOCKS 代理（HTTP_PROXY/ALL_PROXY）。AI 端点是本机代理，
        # 走系统 SOCKS 代理反而会因缺 socksio 直接崩、或把本地请求绕进隧道——直连才对。
        client = OpenAI(base_url=cfg.ai_base_url, api_key=cfg.ai_api_key,
                        http_client=httpx.Client(trust_env=False))
        return cls(store, cfg.training_dir, cfg.ai_model, frames=cfg.ai_label_frames, client=client)

    def _call(self, messages) -> str:
        last = None
        for attempt in range(2):  # 一次 429 退避重试
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages)
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt == 0:
                    self.sleep(2)
        raise last

    def label(self, clip_path) -> dict | None:
        clip_path = Path(clip_path)
        if self.store.label_source(clip_path.name) == "human":
            self.log(f"AI 跳过（已人工标注）：{clip_path.name}")
            return None
        with tempfile.TemporaryDirectory() as td:
            frame_paths = extract_frames(clip_path, Path(td), self.frames)
            if not frame_paths:
                self.log(f"AI 跳过（抽帧为空）：{clip_path.name}")
                return None
            messages = build_messages(encode_frames(frame_paths))
        content = self._call(messages)
        result = parse_label(content)
        self.store.label_clip(
            clip_path, result["drinking"], max_frames=self.train_frames,
            source="ai", confidence=result["confidence"], reason=result["reason"],
        )
        conf = result["confidence"]
        conf_txt = f"{conf:.2f}" if isinstance(conf, float) else "—"
        self.log(f"AI 标注 {clip_path.name}：{'喝水' if result['drinking'] else '没喝'}"
                 f"（{conf_txt}）{result['reason']}")
        return result
