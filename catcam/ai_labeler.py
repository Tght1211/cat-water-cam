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
import threading
import time
import math
import hashlib
from catcam.database import connect
from pathlib import Path

from catcam.feedback import extract_frames
from catcam.videojudge import read_temporal_windows
import cv2

PROMPT = (
    "你是猫咪饮水监控的标注助手。下面是同一段监控视频按时间顺序抽取的若干帧。"
    "这是一台【自动饮水机】，水面【一直】有水波纹和流动——所以【绝对不要】把「水面有波动/波纹/水流」"
    "当作喝水的证据，那是饮水机本身造成的，跟猫喝没喝无关。"
    "判断猫是否在【真的喝水】，唯一标准是看到猫的动作：猫凑近水碗、伸出【粉红色的舌头】、"
    "有【舔水/舔舐】的动作 → 算喝水。"
    "只是凑近、嗅闻、盯着看、路过、趴着、洗脸，但没有伸出舌头舔水 → 不算喝水；"
    "注意连续的口鼻贴近水面、反复舔舐和头部微动；舌头可能被遮挡。"
    "画面过暗、遮挡、抽帧太少而无法确定时，drinking 必须返回 null，不要把看不清当成没喝。"
    "只输出一个 JSON 对象，不要任何多余文字，格式："
    '{"drinking": true 或 false 或 null, "confidence": 0到1的小数, "reason": "简短中文理由"}'
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
    if "drinking" not in obj:
        # 漏了 drinking 字段就当解析失败（抛错由调用方吞掉、该段不标），别静默写成「没喝」污染训练集。
        raise ValueError(f"模型回复缺少 drinking 字段：{content!r}")
    drinking = obj.get("drinking")
    if isinstance(drinking, str):
        value = drinking.strip().lower()
        if value in ('true','yes','1','喝','drinking'):
            drinking = True
        elif value in ('false','no','0','没喝','not_drinking'):
            drinking = False
        elif value in ('unknown','uncertain','null'):
            drinking = None
        else:
            raise ValueError('模型的喝水判断不是有效布尔值')
    if drinking is not None and not isinstance(drinking,bool):
        raise ValueError('模型的喝水判断不是有效布尔值')
    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError) as e:
        raise ValueError('模型置信度无效') from e
    if conf is not None and (not math.isfinite(conf) or not 0 <= conf <= 1):
        raise ValueError('模型置信度必须在0到1之间')
    reason = str(obj.get("reason", "") or "")
    return {"drinking": drinking, "confidence": conf, "reason": reason}


class AILabeler:
    """录一段调一次：抽帧 → 送模型 → 解析 → 写标注（source='ai'）+ 抽帧进训练目录。

    client 可注入（测试塞假 client）；失败抛异常由调用方吞掉记日志，绝不影响录制。
    人工 > AI：已人工标注的段直接跳过、不覆盖。
    """

    def __init__(self, store, training_dir, model, frames=3, client=None,
                 train_frames=5, sleep=time.sleep, log=print, retries=3, retry_base=5.0,
                 budget=None, input_mode='frames', candidate_check=None):
        self.store = store
        self.training_dir = Path(training_dir)
        # model 可传单个字符串或一组模型；组成「模型池」做轮换 + 兜底。
        self.models = [model] if isinstance(model, str) else [m for m in model if m]
        self.frames = frames
        self.client = client
        self.train_frames = train_frames
        self.sleep = sleep
        self.log = log
        # 整池都失败时再重试几轮（免费模型常一阵子全 429）：retries 轮、轮间指数退避 retry_base*2^k 秒。
        self.retries = max(1, retries)
        self.retry_base = retry_base
        self._rr = 0                       # 轮换起始下标
        self._rr_lock = threading.Lock()
        self.budget = budget
        self.input_mode = input_mode
        self.candidate_check = candidate_check
        self.roi_provider = None
        self._label_lock = threading.Lock()
        self._seen = set()
        with connect(store.db_path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS ai_annotations (hash TEXT PRIMARY KEY, clip TEXT NOT NULL, result TEXT NOT NULL, ts REAL NOT NULL)')

    @classmethod
    def from_config(cls, store, cfg):
        """按 config 造一个真 client 的 AILabeler；未启用/缺 key 返回 None。"""
        if not cfg.ai_label_enabled or not cfg.ai_api_key:
            return None
        import httpx
        from openai import OpenAI
        # trust_env=False：忽略环境里的 VPN/SOCKS 代理（HTTP_PROXY/ALL_PROXY）。AI 端点是本机代理，
        # 走系统 SOCKS 代理反而会因缺 socksio 直接崩、或把本地请求绕进隧道——直连才对。
        client = OpenAI(base_url=cfg.ai_base_url, api_key=cfg.ai_api_key, max_retries=0, timeout=90,
                        http_client=httpx.Client(trust_env=False))
        # 主模型 + 兜底模型组成池（去重保序）。
        models, seen = [], set()
        for m in [cfg.ai_model, *getattr(cfg, "ai_fallback_models", [])]:
            if m and m not in seen:
                seen.add(m)
                models.append(m)
        from catcam.ai_budget import AIBudget
        budget = AIBudget(store.db_path,cfg.ai_daily_budget,cfg.ai_min_interval_seconds)
        return cls(store, cfg.training_dir, models, frames=cfg.ai_label_frames, client=client,
                   budget=budget,input_mode=cfg.ai_input_mode,retries=1)

    def _call(self, messages, clip_name=None) -> str:
        """模型池轮换 + 顺位兜底 + 整池失败退避重试：每次换起点分摊额度，某模型失败试下一个；
        一整轮都失败就退避后再来一轮（免费模型常一阵子全 429），重试 self.retries 轮仍失败才抛。"""
        n = len(self.models)
        last = None
        for attempt in range(self.retries):
            with self._rr_lock:
                start = self._rr
                self._rr = (self._rr + 1) % n
            for i in range(n):
                model = self.models[(start + i) % n]
                request_id = self.budget.reserve(model, clip_name) if self.budget else None
                if self.budget and request_id is None:
                    raise RuntimeError('AI每日预算已用完或处于调用冷却，片段保留待人工复核')
                try:
                    resp = self.client.chat.completions.create(model=model, messages=messages)
                    choices = getattr(resp, "choices", None)
                    if not choices:   # 有些上游在HTTP 200里返回错误对象。
                        detail=getattr(resp,'error',None)
                        raise RuntimeError(f"返回空 choices：{detail or '无响应内容'}")
                    if self.budget:
                        self.budget.finish(request_id,'success')
                    return choices[0].message.content
                except Exception as e:  # noqa: BLE001
                    if self.budget:
                        self.budget.finish(request_id,'failed')
                    last = e
                    self.log(f"模型 {model} 调用失败（{e}），换下一个")
            if attempt < self.retries - 1:   # 整池这轮全挂 → 退避后再来一轮
                wait = self.retry_base * (2 ** attempt)
                self.log(f"整池都失败（第 {attempt + 1}/{self.retries} 轮），等 {wait:.0f}s 重试")
                self.sleep(wait)
        raise last

    def label(self, clip_path) -> dict | None:
        # 串行化，同一段在并发请求和重启后不会重复消耗成功的标注预算。
        with self._label_lock:
            return self._label(clip_path)

    def _label(self, clip_path):
        clip_path = Path(clip_path)
        if self.store.label_source(clip_path.name) in ("human", "ai"):
            self.log(f"AI 跳过（已人工标注）：{clip_path.name}")
            return None
        fingerprint = hashlib.sha256(clip_path.read_bytes()).hexdigest() if clip_path.exists() else None
        if fingerprint and fingerprint in self._seen:
            return None
        with connect(self.store.db_path) as c:
            if fingerprint and c.execute('SELECT 1 FROM ai_annotations WHERE hash=?',(fingerprint,)).fetchone():
                return None
        if self.candidate_check is not None and not self.candidate_check(clip_path):
            self.log(f'AI跳过无有效猫咪候选：{clip_path.name}')
            return None
        with tempfile.TemporaryDirectory() as td:
            windows = read_temporal_windows(clip_path,n=self.frames)
            if not windows:
                self.log(f"AI 跳过（抽帧为空）：{clip_path.name}")
                return None
            if self.input_mode == 'video' and clip_path.stat().st_size <= 8*1024*1024:
                url = 'data:video/mp4;base64,' + base64.b64encode(clip_path.read_bytes()).decode()
                messages = [{'role':'user','content':[{'type':'text','text':PROMPT},
                            {'type':'video_url','video_url':{'url':url}}]}]
            else:
                content = [{'type':'text','text':PROMPT+' 以下分组是连续两秒窗口；同组内帧连续，组间可能跳过时间。'}]
                roi = self.roi_provider() if self.roi_provider else None
                for index,frames in enumerate(windows):
                    content.append({'type':'text','text':f'连续窗口 {index+1}，按时间顺序：'})
                    for rgb in frames:
                        h,w=rgb.shape[:2]
                        if roi:
                            x1,y1,x2,y2=roi
                            rgb=rgb[int(max(0,y1-.1)*h):int(min(1,y2+.1)*h),int(max(0,x1-.1)*w):int(min(1,x2+.1)*w)]
                        ok,jpg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
                        if not ok:
                            continue
                        url='data:image/jpeg;base64,'+base64.b64encode(jpg).decode()
                        content.append({'type':'image_url','image_url':{'url':url}})
                messages=[{'role':'user','content':content}]
        content = self._call(messages, clip_path.name)
        result = parse_label(content)
        if fingerprint:
            self._seen.add(fingerprint)
            with connect(self.store.db_path) as c:
                c.execute('INSERT OR REPLACE INTO ai_annotations VALUES (?,?,?,?)',
                          (fingerprint,clip_path.name,json.dumps(result,ensure_ascii=False),time.time()))
        if result['drinking'] is None or result['confidence'] is not None and result['confidence'] < .8:
            self.log(f"AI判断不确定，保留人工复核：{clip_path.name} {result['reason']}")
            return None
        written=self.store.label_clip(
            clip_path, result["drinking"], max_frames=self.train_frames,
            source="ai", confidence=result["confidence"], reason=result["reason"],
        )
        if not written:
            return None
        conf = result["confidence"]
        conf_txt = f"{conf:.2f}" if isinstance(conf, float) else "—"
        self.log(f"AI 标注 {clip_path.name}：{'喝水' if result['drinking'] else '没喝'}"
                 f"（{conf_txt}）{result['reason']}")
        return result
