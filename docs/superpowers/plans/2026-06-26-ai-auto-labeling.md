# AI 自动标注 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 录完一段视频即后台调外部视觉大模型判「猫是否真喝水」，直接写入标注与训练数据，并在训练时做类别平衡，让模型尽快训好。

**Architecture:** 在现有采集链路尾部挂一个 fire-and-forget 后台标注（与发邮件同款），新建 `ai_labeler.py`（纯函数 + 可注入 client 的 `AILabeler`），复用 `FeedbackStore.label_clip` 写库+抽帧；`labels` 表加来源/置信度/理由三列；`trainer.prepare_dataset` 加类别平衡。默认关闭，需显式配 key 开启。

**Tech Stack:** Python, OpenAI SDK（指向 OpenRouter，OpenAI 兼容），SQLite，OpenCV，FastAPI，原生 JS。

---

## File Structure

- `catcam/config.py`（改）——新增 `ai_*` 配置字段。
- `catcam/feedback.py`（改）——`labels` 表加 `source/confidence/reason` 列；`label_clip` 扩展；新增 `label_source` / `label_meta`。
- `catcam/ai_labeler.py`（新）——纯函数 `encode_frames` / `build_messages` / `parse_label` + 类 `AILabeler`。
- `catcam/trainer.py`（改）——`prepare_dataset` 加类别平衡 + 纯函数 `balance_target`。
- `catcam/app.py`（改）——装配 `AILabeler`，会话录完起后台线程 `_ai_label_async`。
- `catcam/web.py`（改）——`/api/clips` 多带来源元数据；视频卡显示来源徽标。
- `pyproject.toml` / `requirements.txt`（改）——加 `openai` 依赖。
- `config.json`（改，运行值，gitignore）——填入 key、开启开关。
- `CLAUDE.md`（改）——补一段 AI 标注说明 + 隐私警示。
- `tests/test_ai_labeler.py`（新）、`tests/test_feedback.py` / `tests/test_trainer.py`（改）——测试。

---

## Task 1: 配置字段

**Files:**
- Modify: `catcam/config.py:49-52`（在训练配置块后插入）
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_config.py` 末尾追加（若文件不存在则创建，顶部 `from catcam.config import load_config, Config`）：

```python
def test_ai_label_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.json")
    assert cfg.ai_label_enabled is False
    assert cfg.ai_base_url == "https://openrouter.ai/api/v1"
    assert cfg.ai_api_key == ""
    assert cfg.ai_model == "google/gemma-4-31b-it:free"
    assert cfg.ai_label_frames == 3


def test_ai_label_roundtrip(tmp_path):
    import json
    p = tmp_path / "config.json"
    load_config(p)  # 生成默认
    raw = json.loads(p.read_text())
    raw["ai_label_enabled"] = True
    raw["ai_api_key"] = "sk-test"
    p.write_text(json.dumps(raw))
    cfg = load_config(p)
    assert cfg.ai_label_enabled is True
    assert cfg.ai_api_key == "sk-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL（`AttributeError` / 缺字段）。

- [ ] **Step 3: Write minimal implementation**

在 `catcam/config.py` 的 `train_imgsz: int = 96` 这一行之后插入：

```python

    # AI 自动标注（外部视觉大模型，OpenRouter / OpenAI 兼容）：录一段自动判「喝/没喝」直接进训练。
    # ⚠️ 开启后猫的画面帧会上传到外部服务器，违背「画面只在本机/局域网」原则——默认关，需显式开。
    ai_label_enabled: bool = False
    ai_base_url: str = "https://openrouter.ai/api/v1"
    ai_api_key: str = ""                              # OpenRouter key，写在 config.json（已 gitignore）
    ai_model: str = "google/gemma-4-31b-it:free"
    ai_label_frames: int = 3                          # 每段送几帧给模型
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add catcam/config.py tests/test_config.py
git commit -m "feat(config): 新增 AI 自动标注配置字段"
```

---

## Task 2: labels 表来源元数据 + label_clip 扩展

**Files:**
- Modify: `catcam/feedback.py:41-69`（迁移 + `label_clip`），新增 `label_source` / `label_meta`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_feedback.py` 追加（顶部已有 `from catcam.feedback import FeedbackStore` 与造 clip 的辅助；若没有，用下方自带的 `_make_clip`）：

```python
import numpy as np, cv2
from catcam.feedback import FeedbackStore

def _make_clip(path, frames=6):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 5, (16, 16))
    for _ in range(frames):
        w.write(np.full((16, 16, 3), 120, np.uint8))
    w.release()

def test_ai_label_stores_source_meta(tmp_path):
    clip = tmp_path / "c1.mp4"; _make_clip(clip)
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    store.label_clip(clip, True, max_frames=3, source="ai", confidence=0.8, reason="舌头接触水面")
    meta = store.label_meta("c1.mp4")
    assert meta == {"is_drinking": True, "source": "ai", "confidence": 0.8, "reason": "舌头接触水面"}
    assert store.label_source("c1.mp4") == "ai"
    assert list((tmp_path / "training" / "drinking").glob("*.jpg"))

def test_human_label_defaults_source_human(tmp_path):
    clip = tmp_path / "c2.mp4"; _make_clip(clip)
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    store.label_clip(clip, False)  # 不传 source → human
    assert store.label_source("c2.mp4") == "human"
    m = store.label_meta("c2.mp4")
    assert m["source"] == "human" and m["confidence"] is None and m["reason"] is None

def test_label_source_none_when_unlabeled(tmp_path):
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    assert store.label_source("nope.mp4") is None
    assert store.label_meta("nope.mp4") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: FAIL（`label_clip()` 不接受 `source` 等关键字 / 无 `label_meta`）。

- [ ] **Step 3: Write minimal implementation**

在 `catcam/feedback.py` 的迁移块（`if "trained_version" not in cols:` 之后）追加三列迁移：

```python
            if "source" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN source TEXT")
            if "confidence" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN confidence REAL")
            if "reason" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN reason TEXT")
```

把 `label_clip` 整体替换为：

```python
    def label_clip(
        self, clip_path: Path, is_drinking: bool, max_frames: int = 5,
        source: str = "human", confidence: float | None = None, reason: str | None = None,
    ) -> None:
        clip_path = Path(clip_path)
        # 改标注 = 数据变了 → trained_version 清回 NULL，让它重新算「未训练」。
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts, trained_version, source, confidence, reason) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?) "
                "ON CONFLICT(clip_name) DO UPDATE SET "
                "is_drinking=excluded.is_drinking, ts=excluded.ts, trained_version=NULL, "
                "source=excluded.source, confidence=excluded.confidence, reason=excluded.reason",
                (clip_path.name, 1 if is_drinking else 0, time.time(), source, confidence, reason),
            )
        sub = "drinking" if is_drinking else "not_drinking"
        extract_frames(clip_path, self.training_dir / sub, max_frames)
```

在 `get_label` 之后新增两个查询方法：

```python
    def label_source(self, clip_name: str) -> str | None:
        """该段标注来源：'human' / 'ai' / None（未标注）。AILabeler 用它实现「人工 > AI」。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source FROM labels WHERE clip_name = ?", (clip_name,)
            ).fetchone()
        return None if row is None else row[0]

    def label_meta(self, clip_name: str) -> dict | None:
        """该段标注的完整元信息，供视频页显示来源徽标；未标注返回 None。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT is_drinking, source, confidence, reason FROM labels WHERE clip_name = ?",
                (clip_name,),
            ).fetchone()
        if row is None:
            return None
        return {"is_drinking": bool(row[0]), "source": row[1],
                "confidence": row[2], "reason": row[3]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add catcam/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): labels 表加来源/置信度/理由，label_clip 支持 AI 来源"
```

---

## Task 3: ai_labeler 纯函数（encode/build/parse）

**Files:**
- Create: `catcam/ai_labeler.py`
- Test: `tests/test_ai_labeler.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_ai_labeler.py`：

```python
import base64
import pytest
from catcam.ai_labeler import encode_frames, build_messages, parse_label

def test_encode_frames_data_url(tmp_path):
    p = tmp_path / "f.jpg"; p.write_bytes(b"\xff\xd8jpegbytes")
    urls = encode_frames([p])
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(urls[0].split(",", 1)[1]) == b"\xff\xd8jpegbytes"

def test_build_messages_has_all_images_and_prompt():
    msgs = build_messages(["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"])
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    texts = [c for c in content if c["type"] == "text"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert "喝水" in texts[0]["text"] and "JSON" in texts[0]["text"]

def test_parse_label_plain_json():
    out = parse_label('{"drinking": true, "confidence": 0.9, "reason": "舔水"}')
    assert out == {"drinking": True, "confidence": 0.9, "reason": "舔水"}

def test_parse_label_with_code_fence_and_noise():
    raw = '好的：\n```json\n{"drinking": false, "confidence": 0.6, "reason": "只是凑近"}\n```\n'
    out = parse_label(raw)
    assert out["drinking"] is False and out["confidence"] == 0.6

def test_parse_label_string_bool_and_missing_fields():
    out = parse_label('{"drinking": "yes"}')
    assert out["drinking"] is True and out["confidence"] is None and out["reason"] == ""

def test_parse_label_garbage_raises():
    with pytest.raises(ValueError):
        parse_label("完全不是 json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ai_labeler.py -q`
Expected: FAIL（`ModuleNotFoundError: catcam.ai_labeler`）。

- [ ] **Step 3: Write minimal implementation**

创建 `catcam/ai_labeler.py`：

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ai_labeler.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add catcam/ai_labeler.py tests/test_ai_labeler.py
git commit -m "feat(ai): ai_labeler 纯函数——抽帧编码/提示组装/回复解析"
```

---

## Task 4: AILabeler 类（抽帧→调模型→写库）

**Files:**
- Modify: `catcam/ai_labeler.py`（追加 `AILabeler` 类）
- Test: `tests/test_ai_labeler.py`（追加）

- [ ] **Step 1: Write the failing test**

在 `tests/test_ai_labeler.py` 追加：

```python
import numpy as np, cv2
from catcam.ai_labeler import AILabeler
from catcam.feedback import FeedbackStore

def _make_clip(path, frames=6):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(frames):
        w.write(np.full((16, 16, 3), 120, np.uint8))
    w.release()

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

class _FakeClient:
    def __init__(self, content, fail_times=0):
        self._content = content; self._fail = fail_times; self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()
    def create(self, **kw):  # 充当 client.chat.completions.create
        self.calls += 1
        if self.calls <= self._fail:
            raise RuntimeError("429 rate limit")
        return _FakeResp(self._content)

def _store(tmp_path):
    return FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")

def test_label_writes_ai_label_and_training_frames(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _FakeClient('{"drinking": true, "confidence": 0.77, "reason": "舔水"}')
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client, sleep=lambda s: None)
    out = lab.label(clip)
    assert out["drinking"] is True
    assert store.label_source("a.mp4") == "ai"
    assert store.label_meta("a.mp4")["confidence"] == 0.77
    assert list((tmp_path / "training" / "drinking").glob("*.jpg"))

def test_label_retries_once_on_error(tmp_path):
    clip = tmp_path / "b.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _FakeClient('{"drinking": false}', fail_times=1)
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client, sleep=lambda s: None)
    out = lab.label(clip)
    assert client.calls == 2 and out["drinking"] is False

def test_label_skips_human_labeled(tmp_path):
    clip = tmp_path / "c.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    store.label_clip(clip, True)  # 人工标过
    client = _FakeClient('{"drinking": false}')
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client, sleep=lambda s: None)
    assert lab.label(clip) is None
    assert client.calls == 0  # 没调模型
    assert store.label_source("c.mp4") == "human"  # 未被覆盖
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ai_labeler.py -q`
Expected: FAIL（无 `AILabeler`）。

- [ ] **Step 3: Write minimal implementation**

在 `catcam/ai_labeler.py` 末尾追加：

```python
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
        from openai import OpenAI
        client = OpenAI(base_url=cfg.ai_base_url, api_key=cfg.ai_api_key)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ai_labeler.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add catcam/ai_labeler.py tests/test_ai_labeler.py
git commit -m "feat(ai): AILabeler——录一段调模型标一段，含重试与人工优先"
```

---

## Task 5: 训练类别平衡

**Files:**
- Modify: `catcam/trainer.py:23-39`（`prepare_dataset`），新增纯函数 `balance_target`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_trainer.py` 追加（顶部 import 增补 `prepare_dataset` 已有；新增 `balance_target`）：

```python
from catcam.trainer import balance_target

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
```

（`_seed` 已存在于该文件；若签名不同，沿用文件里现成的造图辅助。）

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_trainer.py -q`
Expected: FAIL（无 `balance_target` / `prepare_dataset` 不接受 `balance`）。

- [ ] **Step 3: Write minimal implementation**

在 `catcam/trainer.py` 的 `count_images` 之后新增纯函数：

```python
def balance_target(train_counts: dict[str, int]) -> int:
    """类别平衡目标张数 = 各类训练张数的最小值（多数类降采样到它）。"""
    return min(train_counts.values()) if train_counts else 0
```

把 `prepare_dataset` 整体替换为：

```python
def prepare_dataset(
    training_dir: Path, dataset_dir: Path, val_ratio: float = 0.25, balance: bool = True
) -> dict[str, int]:
    """把 data/training/{cls}/ 整理成 YOLO 分类要求的 train/ val/ 两份目录。

    balance=True：对 train 划分做类别平衡——多数类降采样到少数类张数，避免失衡训练
    把模型带歪（自动标注会持续堆负样本，不平衡会越来越严重）。val 保持真实分布。
    """
    training_dir = Path(training_dir)
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts: dict[str, int] = {}
    val_imgs: dict[str, list] = {}
    train_imgs: dict[str, list] = {}
    for c in CLASSES:
        imgs = sorted((training_dir / c).glob("*.jpg"))
        counts[c] = len(imgs)
        n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) > 1 else 0
        val_imgs[c] = imgs[:n_val]
        train_imgs[c] = imgs[n_val:]
        for split in ("train", "val"):
            (dataset_dir / split / c).mkdir(parents=True, exist_ok=True)
    if balance:
        target = balance_target({c: len(train_imgs[c]) for c in CLASSES})
        for c in CLASSES:
            train_imgs[c] = train_imgs[c][:target]  # 降采样到最小类（确定性取前 target 张）
    for c in CLASSES:
        for p in val_imgs[c]:
            shutil.copy(p, dataset_dir / "val" / c / p.name)
        for p in train_imgs[c]:
            shutil.copy(p, dataset_dir / "train" / c / p.name)
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_trainer.py -q`
Expected: PASS（含原有 trainer 测试）。

- [ ] **Step 5: Commit**

```bash
git add catcam/trainer.py tests/test_trainer.py
git commit -m "feat(trainer): 训练 train 划分做类别平衡，val 保持真实分布"
```

---

## Task 6: app.py 装配与后台标注

**Files:**
- Modify: `catcam/app.py`（import、装配、`_ai_label_async`、调用点）

> 本任务为线程装配，难以纯单测；以「导入无误 + 关闭时零副作用 + 手动验证」把关。

- [ ] **Step 1: 加 import**

在 `catcam/app.py:11` 的 `from catcam.classifier import ...` 下方加：

```python
from catcam.ai_labeler import AILabeler
```

- [ ] **Step 2: 装配 AILabeler**

在 `catcam/app.py` 创建 `feedback` 之后（`feedback = FeedbackStore(...)` 那行下面）加：

```python
    ai_labeler = AILabeler.from_config(feedback, cfg)  # 未启用/缺 key 返回 None
    if ai_labeler is not None:
        print(f"AI 自动标注已开启 → {cfg.ai_model}（⚠️ 画面帧会上传外部服务器）")
```

- [ ] **Step 3: 定义 `_ai_label_async`**

在 `_email_async` 函数定义之后插入：

```python
    def _ai_label_async(clip_name: str) -> None:
        # 调外部 API，放后台线程绝不阻塞采集；失败只记日志、该段保持未标注。
        if ai_labeler is None:
            return
        def _run():
            try:
                ai_labeler.label(cfg.clips_dir / clip_name)
            except Exception as e:  # noqa: BLE001
                print(f"AI 标注失败（{clip_name}）：{e}")
        threading.Thread(target=_run, daemon=True).start()
```

- [ ] **Step 4: 在会话录完处调用**

在 `catcam/app.py` 的 `_email_async(res.timestamp, res.photo)` 这一行下方加：

```python
                        _ai_label_async(res.clip_name)
```

- [ ] **Step 5: 验证导入与默认关闭无副作用**

Run: `.venv/bin/python -c "import catcam.app, catcam.ai_labeler; print('ok')"`
Expected: 打印 `ok`，无报错。

Run: `.venv/bin/pytest -q`
Expected: 全绿（既有测试不受影响）。

- [ ] **Step 6: Commit**

```bash
git add catcam/app.py
git commit -m "feat(app): 会话录完后台调 AILabeler 自动标注（默认关）"
```

---

## Task 7: 视频页来源徽标

**Files:**
- Modify: `catcam/web.py:729-736`（`/api/clips`）
- Modify: `catcam/web.py`（前端 `clipCard` + 一点 CSS）
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_web.py` 追加（沿用文件里现成的 app/client fixture；下方示意自带最小构造，按文件已有写法调整）：

```python
def test_clips_includes_label_meta(client, feedback, tmp_path_clip):
    # tmp_path_clip: 一段已存在的 clip 名；feedback 为注入的 FeedbackStore
    name = tmp_path_clip
    feedback.label_clip__by_name(name, True, source="ai", confidence=0.8, reason="舔水") \
        if hasattr(feedback, "label_clip__by_name") else None
    r = client.get("/api/clips").json()
    assert "meta" in r
```

> 说明：`test_web.py` 已有 app/store fixture，请按其现成风格断言 `/api/clips` 返回里含 `meta` 键，且对一段 AI 标注的 clip，`meta[name]["source"] == "ai"`、`reason` 非空。若没有现成 fixture，最小化：构造 `FeedbackStore`+`ClipRecorder` 指向 tmp 目录，造一段 mp4，`label_clip(..., source="ai", confidence=0.8, reason="舔水")`，`create_app(...)` 起 `TestClient`，断言 `meta`。

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: FAIL（返回无 `meta`）。

- [ ] **Step 3: 后端加 meta**

把 `catcam/web.py` 的 `/api/clips` 处理函数替换为：

```python
    @app.get("/api/clips")
    def clips():
        names = [p.name for p in recorder.list_clips()]
        labels = {n: feedback.get_label(n) for n in names}
        durations = {n: clip_duration(clips_dir / n) for n in names}
        preds = stats.clip_predictions()
        predictions = {n: preds[n] for n in names if n in preds}  # 测试模型对该段的判断
        meta = {n: feedback.label_meta(n) for n in names}         # 标注来源/置信度/理由
        return {"clips": names, "labels": labels, "durations": durations,
                "predictions": predictions, "meta": meta}
```

- [ ] **Step 4: 前端显示徽标**

在 `catcam/web.py` 的 `clipCard(n)` 里，把取数据那行改为带上 `meta`，并加一段徽标 HTML。
将：

```javascript
  const lab=clipsData.labels||{},dur=clipsData.durations||{},pr=clipsData.predictions||{};
  const v=lab[n],d=dur[n],en=esc(n),jn=JSON.stringify(n),pred=pr[n],t=clipTime(n);
```

替换为：

```javascript
  const lab=clipsData.labels||{},dur=clipsData.durations||{},pr=clipsData.predictions||{},mt=clipsData.meta||{};
  const v=lab[n],d=dur[n],en=esc(n),jn=JSON.stringify(n),pred=pr[n],t=clipTime(n),m=mt[n];
  const srcBadge=!m?'':(m.source==='ai'
    ? `<span class="src-badge ai" title="${esc(m.reason||'')}">🤖 AI ${m.confidence!=null?m.confidence.toFixed(2):''}</span>`
    : `<span class="src-badge human">✋ 人工</span>`);
```

并把该函数返回模板里 `${statusHtml(v)}` 改为 `${statusHtml(v)}${srcBadge}`。

在 CSS 区（任选一处样式块附近，如 `.dur-badge` 定义之后）加：

```css
.src-badge{display:inline-block;margin-left:6px;font-size:11px;padding:1px 7px;border-radius:8px;vertical-align:middle}
.src-badge.ai{background:rgba(10,132,255,.12);color:var(--accent)}
.src-badge.human{background:var(--line);color:var(--muted)}
```

- [ ] **Step 5: 人工翻转后更新来源（前端）**

在 `fb()` 成功后更新本地缓存的那段（`clipsData.labels[clip]=is;` 附近）加：

```javascript
    if(clipsData.meta)clipsData.meta[clip]={is_drinking:is,source:'human',confidence:null,reason:null};
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add catcam/web.py tests/test_web.py
git commit -m "feat(web): 视频页显示标注来源徽标（AI/人工）"
```

---

## Task 8: 依赖、运行配置与文档

**Files:**
- Modify: `pyproject.toml` / `requirements.txt`（加 `openai`）
- Modify: `config.json`（运行值，gitignore）
- Modify: `CLAUDE.md`

- [ ] **Step 1: 加依赖**

在 `requirements.txt` 增加一行 `openai`（若用 `pyproject.toml` 的 dependencies，则在其列表加 `"openai"`）。

Run: `.venv/bin/pip install openai`
Expected: 安装成功。

- [ ] **Step 2: 验证真实可用（可选、需联网）**

Run: `.venv/bin/python -c "import catcam.app; print('import ok')"`
Expected: `import ok`。

- [ ] **Step 3: 填运行配置**

编辑 `config.json`，加入/设置以下键（此文件已 gitignore，不进库）：

```json
  "ai_label_enabled": true,
  "ai_base_url": "http://localhost:3001/proxy/openrouter/v1",
  "ai_api_key": "sk-MZYyWBsmKi4A5CO_Vfj7XBarU9_TgX6H1P0rDV875ZNwhzt8",
  "ai_model": "google/gemma-4-31b-it:free",
  "ai_label_frames": 3,
```

注意：`ai_base_url` 走用户本机代理 `http://localhost:3001/proxy/openrouter/v1`（OpenAI SDK 会在其后接
`/chat/completions`）。若代理路径不带 `/v1`，去掉 `/v1` 即可。运行前需确保该代理已启动（探测时为 000=未启动）。

- [ ] **Step 4: 文档**

在 `CLAUDE.md` 的「Architecture」末尾或「Conventions / gotchas」加一段：

```markdown
- **AI 自动标注（可选、默认关）**：`ai_labeler.py` 的 `AILabeler` 在会话录完后台调外部视觉大模型
  （OpenRouter / OpenAI 兼容，`ai_base_url`/`ai_api_key`/`ai_model` 可配）判「喝/没喝」，直接写
  `labels`（`source='ai'` + 置信度 + 理由）并抽帧进 `data/training`。**⚠️ 开启后画面帧会上传外部服务器**，
  与「画面只在本机/局域网」原则冲突——`ai_label_enabled` 默认 False，需显式开。失败 fail-open（只记日志、
  该段保持未标注，不影响录制）。人工 > AI：已人工标注的段不被覆盖。训练侧 `prepare_dataset(balance=True)`
  对 train 划分做类别平衡（多数类降采样），val 保持真实分布。
```

- [ ] **Step 5: 全量测试 + 提交**

Run: `.venv/bin/pytest -q`
Expected: 全绿。

```bash
git add pyproject.toml requirements.txt CLAUDE.md
git commit -m "chore: 加 openai 依赖 + CLAUDE.md 记录 AI 自动标注"
```

---

## Self-Review

- **Spec coverage：** 数据流(Task6) / ai_labeler 模块(Task3,4) / 存储三列(Task2) / 配置(Task1) / 网页徽标(Task7) / 类别平衡(Task5) / 依赖(Task8) / 隐私默认关(Task1,6,8) / 测试(各任务) — 全覆盖。
- **Placeholder：** 无 TBD/TODO；除 Task7 的 web 测试因依赖既有 fixture 给了「按现成风格」说明（已附最小构造兜底），其余均含完整代码。
- **类型一致：** `label_clip(..., source, confidence, reason)`、`label_source`、`label_meta`、`AILabeler(store,training_dir,model,frames,client,...)`、`from_config(store,cfg)`、`balance_target(dict)->int`、`prepare_dataset(...,balance=True)` 在各任务间签名一致。
