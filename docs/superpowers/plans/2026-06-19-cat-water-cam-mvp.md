# 猫咪饮水摄像头 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Mac Mini 上跑一个纯本地程序，用普通 USB 摄像头看着流动水碗，YOLO 认出猫并停留时录下 2–5 秒 mp4、只留最近 10 段、记录每日喝水次数，并提供本地网页看/下载视频、看今日统计、给每段视频点 👍/👎 反馈攒训练数据。

**Architecture:** 单进程 Python 程序。一个采集-检测循环（camera → detector → recorder → stats）持续运行；同时用一个线程跑 FastAPI 本地网页。核心判定逻辑（停留状态机、几何重叠、环形裁剪）写成无 I/O 的纯函数/纯类便于单测；摄像头、YOLO、mp4 编码、网页是薄适配层。全程不向外部网络发送任何画面。

**Tech Stack:** Python 3.10+，OpenCV（采集/编码）、ultralytics YOLO（预训练认猫）、FastAPI + uvicorn（本地网页）、SQLite（统计与反馈标签）、pytest（测试）。

## Global Constraints

- Python 版本：**3.10+**（用到 `X | None` 联合类型语法）。
- **纯本地**：运行期绝不把画面/视频/帧发往任何外部网络或第三方 API。唯一允许的外网访问是**首次安装时** ultralytics 自动下载一次 YOLO 预训练权重。
- 网页只绑定局域网/本机，**不做端口转发/内网穿透**。
- 检测触发**只能**用「YOLO 认猫 + 水碗区域重叠」，**禁止**用动静/帧差检测（水碗是流动水会持续误触发）。
- 视频环形**最多保留 10 段**；训练帧存到独立永久目录，不随 10 段轮换被删。
- 所有路径用项目内相对 `data/` 目录派生，不写死绝对路径。
- 包名统一 `catcam`；坐标矩形统一用 `(x1, y1, x2, y2)` 四元组；ROI 在配置里用 0–1 比例表示（与分辨率无关）。
- 每个任务最后必须 `git commit`。

---

### Task 0: 项目脚手架与依赖

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `catcam/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: 可导入的 `catcam` 包；`pytest` 可运行；`data/` 被 git 忽略。

- [ ] **Step 1: 写 `.gitignore`**

Create `.gitignore`:

```gitignore
__pycache__/
*.pyc
.venv/
data/
*.pt
.DS_Store
```

- [ ] **Step 2: 写 `requirements.txt`**

Create `requirements.txt`:

```text
opencv-python>=4.8
ultralytics>=8.0
fastapi>=0.110
uvicorn>=0.27
numpy>=1.24
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 3: 写 `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "catcam"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["catcam*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 建包文件**

Create `catcam/__init__.py` (empty file). Create `tests/__init__.py` (empty file).

- [ ] **Step 5: 写冒烟测试**

Create `tests/test_smoke.py`:

```python
def test_can_import_catcam():
    import catcam
    assert catcam is not None
```

- [ ] **Step 6: 建虚拟环境并安装**

Run:
```bash
cd /Users/tght/develop/project/2025/have-try/cat-water-cam
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
```
Expected: 安装成功，结尾出现 `Successfully installed ...`（首次会下载较多包，耐心等）。

- [ ] **Step 7: 跑冒烟测试确认通过**

Run: `.venv/bin/pytest tests/test_smoke.py -v`
Expected: PASS（1 passed）。

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt pyproject.toml catcam/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: 项目脚手架与依赖"
```

---

### Task 1: 配置模块 `config`

**Files:**
- Create: `catcam/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Config` dataclass，字段见下（其它任务都按这些名字读配置）。
  - 属性 `clips_dir: Path`、`training_dir: Path`、`db_path: Path`。
  - `load_config(path: str | Path) -> Config`：文件不存在则写入默认值再返回。

- [ ] **Step 1: 写失败测试**

Create `tests/test_config.py`:

```python
import json
from pathlib import Path
from catcam.config import Config, load_config


def test_defaults_have_expected_values():
    c = Config()
    assert c.max_clips == 10
    assert c.bowl_roi == (0.3, 0.3, 0.7, 0.7)
    assert c.fps == 10


def test_derived_paths(tmp_path):
    c = Config(data_dir=str(tmp_path))
    assert c.clips_dir == tmp_path / "clips"
    assert c.training_dir == tmp_path / "training"
    assert c.db_path == tmp_path / "catcam.db"


def test_load_config_creates_default_file_when_missing(tmp_path):
    cfg_path = tmp_path / "config.json"
    c = load_config(cfg_path)
    assert cfg_path.exists()
    assert c.max_clips == 10
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["max_clips"] == 10


def test_load_config_reads_existing_overrides(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"max_clips": 5, "dwell_seconds": 2.0}))
    c = load_config(cfg_path)
    assert c.max_clips == 5
    assert c.dwell_seconds == 2.0
    # 未指定字段用默认
    assert c.fps == 10
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.config'`）。

- [ ] **Step 3: 写实现**

Create `catcam/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    camera_index: int = 0
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
    web_host: str = "0.0.0.0"
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/config.py tests/test_config.py
git commit -m "feat(config): 配置 dataclass 与加载"
```

---

### Task 2: 几何重叠 `geometry`（纯函数）

**Files:**
- Create: `catcam/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Rect = tuple[float, float, float, float]`，含义 `(x1, y1, x2, y2)`。
  - `area(rect: Rect) -> float`
  - `intersection_area(a: Rect, b: Rect) -> float`
  - `ratio_rect_to_pixels(rect: Rect, width: int, height: int) -> Rect`
  - `cat_overlaps_bowl(cat: Rect, bowl: Rect, min_ratio: float) -> bool`（交集 / 水碗面积 ≥ min_ratio）

- [ ] **Step 1: 写失败测试**

Create `tests/test_geometry.py`:

```python
from catcam.geometry import (
    area,
    intersection_area,
    ratio_rect_to_pixels,
    cat_overlaps_bowl,
)


def test_area():
    assert area((0, 0, 10, 4)) == 40


def test_intersection_partial():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert intersection_area(a, b) == 25


def test_intersection_none():
    a = (0, 0, 5, 5)
    b = (10, 10, 20, 20)
    assert intersection_area(a, b) == 0


def test_ratio_rect_to_pixels():
    assert ratio_rect_to_pixels((0.5, 0.5, 1.0, 1.0), 640, 480) == (320.0, 240.0, 640.0, 480.0)


def test_cat_overlaps_bowl_true_when_enough():
    bowl = (0, 0, 10, 10)          # 面积 100
    cat = (0, 0, 5, 10)            # 交集 50 -> 比例 0.5
    assert cat_overlaps_bowl(cat, bowl, min_ratio=0.15) is True


def test_cat_overlaps_bowl_false_when_too_little():
    bowl = (0, 0, 10, 10)          # 面积 100
    cat = (0, 0, 1, 10)            # 交集 10 -> 比例 0.1
    assert cat_overlaps_bowl(cat, bowl, min_ratio=0.15) is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_geometry.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.geometry'`）。

- [ ] **Step 3: 写实现**

Create `catcam/geometry.py`:

```python
from __future__ import annotations

Rect = tuple[float, float, float, float]


def area(rect: Rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def ratio_rect_to_pixels(rect: Rect, width: int, height: int) -> Rect:
    x1, y1, x2, y2 = rect
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def cat_overlaps_bowl(cat: Rect, bowl: Rect, min_ratio: float) -> bool:
    bowl_area = area(bowl)
    if bowl_area <= 0:
        return False
    return intersection_area(cat, bowl) / bowl_area >= min_ratio
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_geometry.py -v`
Expected: PASS（6 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/geometry.py tests/test_geometry.py
git commit -m "feat(geometry): 矩形重叠与水碗判定纯函数"
```

---

### Task 3: 停留状态机 `detector`（纯类，喝水判定核心）

**Files:**
- Create: `catcam/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass DrinkingEvent` 含字段 `timestamp: float`。
  - `class DrinkingDetector(dwell_seconds: float, cooldown_seconds: float)`，方法
    `update(now: float, cat_in_roi: bool) -> DrinkingEvent | None`：猫在水碗区域连续停留 ≥ dwell_seconds 触发一次事件，触发后 cooldown_seconds 内不再触发；猫离开即重置计时。

- [ ] **Step 1: 写失败测试**

Create `tests/test_detector.py`:

```python
from catcam.detector import DrinkingDetector, DrinkingEvent


def test_no_event_before_dwell_threshold():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    assert d.update(now=0.0, cat_in_roi=True) is None
    assert d.update(now=2.9, cat_in_roi=True) is None


def test_event_fires_after_continuous_dwell():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    evt = d.update(now=3.0, cat_in_roi=True)
    assert isinstance(evt, DrinkingEvent)
    assert evt.timestamp == 3.0


def test_leaving_resets_dwell():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    d.update(now=1.0, cat_in_roi=False)   # 离开，重置
    d.update(now=2.0, cat_in_roi=True)    # 重新开始计时
    assert d.update(now=4.0, cat_in_roi=True) is None   # 仅停留 2s
    assert isinstance(d.update(now=5.0, cat_in_roi=True), DrinkingEvent)


def test_cooldown_blocks_second_event():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    assert isinstance(d.update(now=3.0, cat_in_roi=True), DrinkingEvent)
    # 冷却期内即使一直停留也不再触发
    assert d.update(now=10.0, cat_in_roi=True) is None
    assert d.update(now=62.0, cat_in_roi=True) is None


def test_event_again_after_cooldown():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    d.update(now=3.0, cat_in_roi=True)              # 第一次事件，冷却到 63
    d.update(now=64.0, cat_in_roi=True)             # 冷却结束，重新计时起点
    evt = d.update(now=67.0, cat_in_roi=True)
    assert isinstance(evt, DrinkingEvent)
    assert evt.timestamp == 67.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_detector.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.detector'`）。

- [ ] **Step 3: 写实现**

Create `catcam/detector.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrinkingEvent:
    timestamp: float


class DrinkingDetector:
    def __init__(self, dwell_seconds: float, cooldown_seconds: float):
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._dwell_start: float | None = None
        self._cooldown_until: float = 0.0

    def update(self, now: float, cat_in_roi: bool) -> DrinkingEvent | None:
        if not cat_in_roi:
            self._dwell_start = None
            return None
        if now < self._cooldown_until:
            # 冷却期内：不计时、不触发
            self._dwell_start = None
            return None
        if self._dwell_start is None:
            self._dwell_start = now
            return None
        if now - self._dwell_start >= self.dwell_seconds:
            self._dwell_start = None
            self._cooldown_until = now + self.cooldown_seconds
            return DrinkingEvent(timestamp=now)
        return None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_detector.py -v`
Expected: PASS（5 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/detector.py tests/test_detector.py
git commit -m "feat(detector): 停留+冷却喝水判定状态机"
```

---

### Task 4: 认猫适配层 `vision`（YOLO 解析纯函数 + 薄适配）

**Files:**
- Create: `catcam/vision.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: `catcam.geometry.Rect`
- Produces:
  - `filter_cat_boxes(detections: list[tuple[str, float, Rect]], confidence: float) -> list[Rect]`：从 `(类别名, 置信度, 框)` 列表里筛出置信度达标的 `cat` 框（纯函数）。
  - `class CatDetector(model)`：`detect_cats(frame) -> list[Rect]`，内部跑 YOLO 并调用 `filter_cat_boxes`。
  - `CatDetector.from_path(model_path: str, confidence: float) -> CatDetector`。

- [ ] **Step 1: 写失败测试**

Create `tests/test_vision.py`:

```python
from catcam.vision import filter_cat_boxes, CatDetector


def test_filter_keeps_only_confident_cats():
    detections = [
        ("cat", 0.9, (0, 0, 10, 10)),
        ("dog", 0.95, (1, 1, 2, 2)),
        ("cat", 0.2, (3, 3, 4, 4)),     # 置信度不够
    ]
    boxes = filter_cat_boxes(detections, confidence=0.4)
    assert boxes == [(0, 0, 10, 10)]


def test_filter_empty():
    assert filter_cat_boxes([], confidence=0.4) == []


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: "cat", 1: "dog"}

    def __call__(self, frame, verbose=False):
        return [
            _FakeResult([
                _FakeBox(0, 0.9, (0.0, 0.0, 10.0, 10.0)),
                _FakeBox(1, 0.99, (1.0, 1.0, 2.0, 2.0)),
            ])
        ]


def test_cat_detector_with_fake_model():
    det = CatDetector(_FakeModel(), confidence=0.4)
    boxes = det.detect_cats(frame=None)
    assert boxes == [(0.0, 0.0, 10.0, 10.0)]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_vision.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.vision'`）。

- [ ] **Step 3: 写实现**

Create `catcam/vision.py`:

```python
from __future__ import annotations

from catcam.geometry import Rect


def filter_cat_boxes(
    detections: list[tuple[str, float, Rect]], confidence: float
) -> list[Rect]:
    return [box for name, conf, box in detections if name == "cat" and conf >= confidence]


class CatDetector:
    def __init__(self, model, confidence: float):
        self.model = model
        self.confidence = confidence

    @classmethod
    def from_path(cls, model_path: str, confidence: float) -> "CatDetector":
        from ultralytics import YOLO

        return cls(YOLO(model_path), confidence)

    def detect_cats(self, frame) -> list[Rect]:
        results = self.model(frame, verbose=False)
        detections: list[tuple[str, float, Rect]] = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                detections.append((self.model.names[cls], conf, (x1, y1, x2, y2)))
        return filter_cat_boxes(detections, self.confidence)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_vision.py -v`
Expected: PASS（3 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/vision.py tests/test_vision.py
git commit -m "feat(vision): YOLO 认猫适配与筛选纯函数"
```

---

### Task 5: 滚动帧缓存 `framebuffer`

**Files:**
- Create: `catcam/framebuffer.py`
- Test: `tests/test_framebuffer.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `class FrameBuffer(seconds: float, fps: int)`：环形缓存最近 `seconds` 秒的帧。
    - `add(timestamp: float, frame)`：加入一帧。
    - `all_frames() -> list`：按时间顺序返回当前缓存的所有帧（不含时间戳）。
    - 属性 `maxlen: int`（= `max(1, int(seconds*fps))`）。

- [ ] **Step 1: 写失败测试**

Create `tests/test_framebuffer.py`:

```python
from catcam.framebuffer import FrameBuffer


def test_maxlen_from_seconds_and_fps():
    fb = FrameBuffer(seconds=4.0, fps=10)
    assert fb.maxlen == 40


def test_keeps_only_latest_maxlen_frames():
    fb = FrameBuffer(seconds=1.0, fps=3)   # maxlen = 3
    for i in range(5):
        fb.add(timestamp=float(i), frame=f"f{i}")
    assert fb.all_frames() == ["f2", "f3", "f4"]


def test_maxlen_never_zero():
    fb = FrameBuffer(seconds=0.0, fps=0)
    assert fb.maxlen == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_framebuffer.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.framebuffer'`）。

- [ ] **Step 3: 写实现**

Create `catcam/framebuffer.py`:

```python
from __future__ import annotations

from collections import deque


class FrameBuffer:
    def __init__(self, seconds: float, fps: int):
        self.maxlen = max(1, int(seconds * fps))
        self._buf: deque = deque(maxlen=self.maxlen)

    def add(self, timestamp: float, frame) -> None:
        self._buf.append((timestamp, frame))

    def all_frames(self) -> list:
        return [frame for _, frame in self._buf]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_framebuffer.py -v`
Expected: PASS（3 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/framebuffer.py tests/test_framebuffer.py
git commit -m "feat(framebuffer): 滚动帧缓存"
```

---

### Task 6: 录制与环形保留 `recorder`

**Files:**
- Create: `catcam/recorder.py`
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `catcam.config.Config`（用 `fps`）
- Produces:
  - `prune_dir(clips_dir: Path, max_clips: int) -> None`：删掉除最新 `max_clips` 个 `clip_*.mp4` 外的旧文件（按文件名排序，文件名内含时间戳保证新旧序）。
  - `clip_filename(timestamp: float) -> str`：返回 `clip_{ms}.mp4`（`ms = int(timestamp*1000)`）。
  - `class ClipRecorder(clips_dir: Path, max_clips: int, fps: int)`：
    - `save_clip(frames: list, timestamp: float) -> Path`：把帧写成 mp4，命名 `clip_{ms}.mp4`，写后调用 `prune_dir`，返回文件路径。
    - `list_clips() -> list[Path]`：现存 clip，**最新在前**。

- [ ] **Step 1: 写失败测试（prune 与命名，用假文件，不依赖编码器）**

Create `tests/test_recorder.py`:

```python
import numpy as np
from catcam.recorder import prune_dir, clip_filename, ClipRecorder


def _touch_clip(d, ms):
    (d / f"clip_{ms}.mp4").write_bytes(b"x")


def test_clip_filename_uses_milliseconds():
    assert clip_filename(1.5) == "clip_1500.mp4"


def test_prune_keeps_only_newest(tmp_path):
    for ms in [1000, 2000, 3000, 4000]:
        _touch_clip(tmp_path, ms)
    prune_dir(tmp_path, max_clips=2)
    remaining = sorted(p.name for p in tmp_path.glob("clip_*.mp4"))
    assert remaining == ["clip_3000.mp4", "clip_4000.mp4"]


def test_prune_noop_when_under_limit(tmp_path):
    _touch_clip(tmp_path, 1000)
    prune_dir(tmp_path, max_clips=10)
    assert len(list(tmp_path.glob("clip_*.mp4"))) == 1


def test_save_clip_writes_file_and_enforces_limit(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frames = [frame for _ in range(10)]
    path = rec.save_clip(frames, timestamp=12.0)
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.name == "clip_12000.mp4"


def test_save_clip_ring_buffer_drops_oldest(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=3, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for ts in [1.0, 2.0, 3.0, 4.0]:
        rec.save_clip([frame, frame], timestamp=ts)
    names = sorted(p.name for p in rec.list_clips())
    assert names == ["clip_2000.mp4", "clip_3000.mp4", "clip_4000.mp4"]


def test_list_clips_newest_first(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for ts in [1.0, 3.0, 2.0]:
        rec.save_clip([frame], timestamp=ts)
    names = [p.name for p in rec.list_clips()]
    assert names == ["clip_3000.mp4", "clip_2000.mp4", "clip_1000.mp4"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_recorder.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.recorder'`）。

- [ ] **Step 3: 写实现**

Create `catcam/recorder.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2


def clip_filename(timestamp: float) -> str:
    return f"clip_{int(timestamp * 1000)}.mp4"


def prune_dir(clips_dir: Path, max_clips: int) -> None:
    clips = sorted(clips_dir.glob("clip_*.mp4"))
    for old in clips[:-max_clips] if max_clips > 0 else clips:
        old.unlink()


class ClipRecorder:
    def __init__(self, clips_dir: Path, max_clips: int, fps: int):
        self.clips_dir = Path(clips_dir)
        self.max_clips = max_clips
        self.fps = fps
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_clip(self, frames: list, timestamp: float) -> Path:
        if not frames:
            raise ValueError("frames 为空，无法录制")
        path = self.clips_dir / clip_filename(timestamp)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (width, height))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        prune_dir(self.clips_dir, self.max_clips)
        return path

    def list_clips(self) -> list[Path]:
        return sorted(self.clips_dir.glob("clip_*.mp4"), reverse=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_recorder.py -v`
Expected: PASS（6 passed）。若 `save_clip` 相关用例因本机缺 mp4 编码器失败，改 fourcc 为 `"avc1"` 重试；仍不行则在该机器 `brew install opencv` 后重装 `opencv-python`。prune/命名用例不受编码器影响，必须先全绿。

- [ ] **Step 5: Commit**

```bash
git add catcam/recorder.py tests/test_recorder.py
git commit -m "feat(recorder): mp4 录制与最近10段环形保留"
```

---

### Task 7: 统计存储 `stats`（SQLite）

**Files:**
- Create: `catcam/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `day_bounds(dt: datetime) -> tuple[float, float]`：该自然日的 `[起, 止)` 时间戳。
  - `class StatsStore(db_path: Path)`：建表 `events(id, ts, clip_name)`。
    - `record_event(timestamp: float, clip_name: str | None = None) -> int`：返回新行 id。
    - `count_between(start_ts: float, end_ts: float) -> int`。
    - `events_between(start_ts: float, end_ts: float) -> list[dict]`：每项 `{"ts": float, "clip_name": str | None}`，按时间升序。

- [ ] **Step 1: 写失败测试**

Create `tests/test_stats.py`:

```python
from datetime import datetime
from catcam.stats import StatsStore, day_bounds


def test_day_bounds():
    dt = datetime(2026, 6, 19, 14, 30, 0)
    start, end = day_bounds(dt)
    assert datetime.fromtimestamp(start) == datetime(2026, 6, 19, 0, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 6, 20, 0, 0, 0)


def test_record_and_count(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    store.record_event(100.0, "clip_100000.mp4")
    store.record_event(150.0, "clip_150000.mp4")
    store.record_event(500.0, None)
    assert store.count_between(0.0, 200.0) == 2
    assert store.count_between(0.0, 1000.0) == 3


def test_events_between_sorted_with_fields(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    store.record_event(300.0, "b.mp4")
    store.record_event(100.0, "a.mp4")
    events = store.events_between(0.0, 1000.0)
    assert [e["ts"] for e in events] == [100.0, 300.0]
    assert events[0]["clip_name"] == "a.mp4"


def test_record_returns_id(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    first = store.record_event(1.0)
    second = store.record_event(2.0)
    assert second == first + 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.stats'`）。

- [ ] **Step 3: 写实现**

Create `catcam/stats.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def day_bounds(dt: datetime) -> tuple[float, float]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


class StatsStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts REAL NOT NULL, "
                "clip_name TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record_event(self, timestamp: float, clip_name: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, clip_name) VALUES (?, ?)",
                (timestamp, clip_name),
            )
            return int(cur.lastrowid)

    def count_between(self, start_ts: float, end_ts: float) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events WHERE ts >= ? AND ts < ?",
                (start_ts, end_ts),
            )
            return int(cur.fetchone()[0])

    def events_between(self, start_ts: float, end_ts: float) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT ts, clip_name FROM events "
                "WHERE ts >= ? AND ts < ? ORDER BY ts ASC",
                (start_ts, end_ts),
            )
            return [{"ts": row[0], "clip_name": row[1]} for row in cur.fetchall()]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/stats.py tests/test_stats.py
git commit -m "feat(stats): SQLite 喝水事件记录与今日统计"
```

---

### Task 8: 反馈标注与训练帧入库 `feedback`

**Files:**
- Create: `catcam/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `catcam.recorder.ClipRecorder`（测试里用它造真 mp4）
- Produces:
  - `extract_frames(clip_path: Path, out_dir: Path, max_frames: int) -> list[Path]`：从 mp4 均匀抽最多 `max_frames` 帧存成 jpg，返回写出的路径。
  - `class FeedbackStore(db_path: Path, training_dir: Path)`：建表 `labels(clip_name UNIQUE, is_drinking, ts)`。
    - `label_clip(clip_path: Path, is_drinking: bool, max_frames: int = 5) -> None`：写/覆盖该 clip 标签，并把抽出的帧存到 `training_dir/<drinking|not_drinking>/`。
    - `get_label(clip_name: str) -> bool | None`。

- [ ] **Step 1: 写失败测试**

Create `tests/test_feedback.py`:

```python
import numpy as np
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore, extract_frames


def _make_clip(tmp_path, ts=1.0, n=6):
    rec = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return rec.save_clip([frame for _ in range(n)], timestamp=ts)


def test_extract_frames_writes_jpgs(tmp_path):
    clip = _make_clip(tmp_path)
    out = tmp_path / "frames"
    paths = extract_frames(clip, out, max_frames=3)
    assert 1 <= len(paths) <= 3
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"


def test_label_clip_stores_label_and_frames(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=True, max_frames=3)
    assert store.get_label(clip.name) is True
    drinking_dir = tmp_path / "train" / "drinking"
    assert drinking_dir.exists()
    assert len(list(drinking_dir.glob("*.jpg"))) >= 1


def test_label_clip_negative_goes_to_not_drinking(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=False, max_frames=2)
    assert store.get_label(clip.name) is False
    assert (tmp_path / "train" / "not_drinking").exists()


def test_relabel_overwrites(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=True, max_frames=2)
    store.label_clip(clip, is_drinking=False, max_frames=2)
    assert store.get_label(clip.name) is False


def test_get_label_none_when_unlabeled(tmp_path):
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    assert store.get_label("nope.mp4") is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_feedback.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.feedback'`）。

- [ ] **Step 3: 写实现**

Create `catcam/feedback.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2


def extract_frames(clip_path: Path, out_dir: Path, max_frames: int) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return []
    step = max(1, len(frames) // max_frames)
    chosen = frames[::step][:max_frames]
    stem = Path(clip_path).stem
    written: list[Path] = []
    for i, frame in enumerate(chosen):
        p = out_dir / f"{stem}_{i}.jpg"
        cv2.imwrite(str(p), frame)
        written.append(p)
    return written


class FeedbackStore:
    def __init__(self, db_path: Path, training_dir: Path):
        self.db_path = Path(db_path)
        self.training_dir = Path(training_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS labels ("
                "clip_name TEXT PRIMARY KEY, "
                "is_drinking INTEGER NOT NULL, "
                "ts REAL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def label_clip(self, clip_path: Path, is_drinking: bool, max_frames: int = 5) -> None:
        clip_path = Path(clip_path)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, ?, NULL) "
                "ON CONFLICT(clip_name) DO UPDATE SET is_drinking=excluded.is_drinking",
                (clip_path.name, 1 if is_drinking else 0),
            )
        sub = "drinking" if is_drinking else "not_drinking"
        extract_frames(clip_path, self.training_dir / sub, max_frames)

    def get_label(self, clip_name: str) -> bool | None:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT is_drinking FROM labels WHERE clip_name = ?", (clip_name,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return bool(row[0])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_feedback.py -v`
Expected: PASS（5 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): 反馈标注与训练帧入库"
```

---

### Task 9: 本地网页 `web`（FastAPI，依赖注入便于测试）

**Files:**
- Create: `catcam/web.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `catcam.stats.StatsStore`、`catcam.recorder.ClipRecorder`、`catcam.feedback.FeedbackStore`
- Produces:
  - `create_app(stats: StatsStore, recorder: ClipRecorder, feedback: FeedbackStore, frame_provider, clips_dir: Path) -> FastAPI`
    - `frame_provider` 是一个返回「最新一帧 numpy 数组或 None」的可调用对象。
    - 路由：
      - `GET /` → HTML 页面（含自动刷新的快照 `<img>`、今日统计、视频列表与 👍/👎 按钮）。
      - `GET /api/stats/today` → `{"count": int, "times": [ISO字符串,...]}`。
      - `GET /api/clips` → `{"clips": ["clip_xxx.mp4", ...]}`（最新在前）。
      - `GET /clips/{name}` → 该 mp4 文件（下载/播放）。
      - `GET /snapshot.jpg` → 最新帧 JPEG；无帧时 503。
      - `POST /api/feedback` → body `{"clip": str, "is_drinking": bool}`，调用 `feedback.label_clip`。

- [ ] **Step 1: 写失败测试**

Create `tests/test_web.py`:

```python
import time
import numpy as np
from fastapi.testclient import TestClient

from catcam.stats import StatsStore
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.web import create_app


def _build(tmp_path, frame_provider=lambda: None):
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    app = create_app(stats, recorder, feedback, frame_provider, recorder.clips_dir)
    return app, stats, recorder, feedback


def test_index_serves_html(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_today_stats_counts_recent_event(tmp_path):
    app, stats, *_ = _build(tmp_path)
    stats.record_event(time.time(), "clip_x.mp4")
    client = TestClient(app)
    r = client.get("/api/stats/today")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert len(body["times"]) == body["count"]


def test_clips_list_and_download(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame, frame], timestamp=1.0)
    client = TestClient(app)
    listing = client.get("/api/clips").json()["clips"]
    assert listing == ["clip_1000.mp4"]
    dl = client.get("/clips/clip_1000.mp4")
    assert dl.status_code == 200
    assert len(dl.content) > 0


def test_snapshot_503_without_frame(tmp_path):
    app, *_ = _build(tmp_path, frame_provider=lambda: None)
    client = TestClient(app)
    assert client.get("/snapshot.jpg").status_code == 503


def test_snapshot_returns_jpeg_with_frame(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    app, *_ = _build(tmp_path, frame_provider=lambda: frame)
    client = TestClient(app)
    r = client.get("/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 0


def test_post_feedback_persists_label(tmp_path):
    app, _, recorder, feedback = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame, frame], timestamp=2.0)
    client = TestClient(app)
    r = client.post("/api/feedback", json={"clip": "clip_2000.mp4", "is_drinking": True})
    assert r.status_code == 200
    assert feedback.get_label("clip_2000.mp4") is True


def test_download_rejects_path_traversal(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    assert client.get("/clips/..%2F..%2Fsecret.txt").status_code in (400, 404)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_web.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.web'`）。

- [ ] **Step 3: 写实现**

Create `catcam/web.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.stats import StatsStore, day_bounds

INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>猫咪饮水监控</title>
<style>body{font-family:sans-serif;margin:1rem;max-width:760px}
img{max-width:100%;border:1px solid #ccc}.clip{margin:.5rem 0;padding:.5rem;border:1px solid #eee}
button{font-size:1rem;margin-right:.5rem}</style></head>
<body>
<h2>实时画面</h2>
<img id="live" src="/snapshot.jpg" alt="live">
<h2>今日喝水：<span id="count">-</span> 次</h2>
<ul id="times"></ul>
<h2>最近视频</h2>
<div id="clips"></div>
<script>
setInterval(()=>{document.getElementById('live').src='/snapshot.jpg?t='+Date.now()},1000);
async function refresh(){
  const s=await (await fetch('/api/stats/today')).json();
  document.getElementById('count').textContent=s.count;
  document.getElementById('times').innerHTML=s.times.map(t=>`<li>${t}</li>`).join('');
  const c=await (await fetch('/api/clips')).json();
  document.getElementById('clips').innerHTML=c.clips.map(n=>`
    <div class="clip"><video src="/clips/${n}" controls width="320"></video><br>
    <a href="/clips/${n}" download>下载 ${n}</a><br>
    <button onclick="fb('${n}',true)">👍 真喝水</button>
    <button onclick="fb('${n}',false)">👎 没喝</button></div>`).join('');
}
async function fb(clip,is){
  await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip,is_drinking:is})});
  alert('已记录反馈：'+clip);
}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class FeedbackBody(BaseModel):
    clip: str
    is_drinking: bool


def create_app(
    stats: StatsStore,
    recorder: ClipRecorder,
    feedback: FeedbackStore,
    frame_provider,
    clips_dir: Path,
) -> FastAPI:
    app = FastAPI()
    clips_dir = Path(clips_dir)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/stats/today")
    def today():
        start, end = day_bounds(datetime.now())
        events = stats.events_between(start, end)
        times = [datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S") for e in events]
        return {"count": len(events), "times": times}

    @app.get("/api/clips")
    def clips():
        return {"clips": [p.name for p in recorder.list_clips()]}

    @app.get("/clips/{name}")
    def get_clip(name: str):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/snapshot.jpg")
    def snapshot():
        frame = frame_provider()
        if frame is None:
            raise HTTPException(status_code=503, detail="no frame yet")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.post("/api/feedback")
    def post_feedback(body: FeedbackBody):
        if "/" in body.clip or ".." in body.clip:
            raise HTTPException(status_code=400, detail="bad clip")
        feedback.label_clip(clips_dir / body.clip, body.is_drinking)
        return JSONResponse({"ok": True})

    return app
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_web.py -v`
Expected: PASS（7 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/web.py tests/test_web.py
git commit -m "feat(web): 本地网页 实时快照/视频列表下载/今日统计/反馈"
```

---

### Task 10: 编排步进 `pipeline`（采集→检测→录制→统计 单步，可测）

**Files:**
- Create: `catcam/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `catcam.detector.DrinkingDetector`、`catcam.vision.CatDetector`、`catcam.framebuffer.FrameBuffer`、`catcam.recorder.ClipRecorder`、`catcam.stats.StatsStore`、`catcam.geometry`
- Produces:
  - `class Pipeline(cat_detector, drinking_detector, frame_buffer, recorder, stats, bowl_roi_ratio, min_overlap_ratio)`：
    - `process(now: float, frame) -> str | None`：处理一帧。把帧入缓存→检测猫→算是否在水碗→喂状态机；若产生事件则录 clip、写统计，返回 clip 文件名；否则返回 None。

- [ ] **Step 1: 写失败测试（用假 CatDetector，不依赖真 YOLO）**

Create `tests/test_pipeline.py`:

```python
import numpy as np
from catcam.detector import DrinkingDetector
from catcam.framebuffer import FrameBuffer
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.pipeline import Pipeline


class _CatAlwaysInBowl:
    """假认猫器：永远返回一个铺满整张图的猫框。"""
    def detect_cats(self, frame):
        h, w = frame.shape[:2]
        return [(0.0, 0.0, float(w), float(h))]


class _NoCat:
    def detect_cats(self, frame):
        return []


def _build(tmp_path, cat_detector):
    return Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0),
        frame_buffer=FrameBuffer(seconds=4.0, fps=5),
        recorder=ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5),
        stats=StatsStore(tmp_path / "s.db"),
        bowl_roi_ratio=(0.0, 0.0, 1.0, 1.0),
        min_overlap_ratio=0.15,
    )


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_no_event_without_cat(tmp_path):
    pipe = _build(tmp_path, _NoCat())
    assert pipe.process(now=0.0, frame=_frame()) is None
    assert pipe.process(now=5.0, frame=_frame()) is None


def test_event_records_clip_and_stat(tmp_path):
    pipe = _build(tmp_path, _CatAlwaysInBowl())
    assert pipe.process(now=0.0, frame=_frame()) is None     # 开始计时
    clip_name = pipe.process(now=3.0, frame=_frame())        # 满 3s 触发
    assert clip_name == "clip_3000.mp4"
    start, end = 0.0, 1e12
    assert pipe.stats.count_between(start, end) == 1
    assert (tmp_path / "clips" / clip_name).exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'catcam.pipeline'`）。

- [ ] **Step 3: 写实现**

Create `catcam/pipeline.py`:

```python
from __future__ import annotations

from catcam.geometry import ratio_rect_to_pixels, cat_overlaps_bowl


class Pipeline:
    def __init__(
        self,
        cat_detector,
        drinking_detector,
        frame_buffer,
        recorder,
        stats,
        bowl_roi_ratio,
        min_overlap_ratio: float,
    ):
        self.cat_detector = cat_detector
        self.drinking_detector = drinking_detector
        self.frame_buffer = frame_buffer
        self.recorder = recorder
        self.stats = stats
        self.bowl_roi_ratio = bowl_roi_ratio
        self.min_overlap_ratio = min_overlap_ratio

    def process(self, now: float, frame) -> str | None:
        self.frame_buffer.add(now, frame)
        height, width = frame.shape[:2]
        bowl = ratio_rect_to_pixels(self.bowl_roi_ratio, width, height)
        cats = self.cat_detector.detect_cats(frame)
        cat_in_roi = any(
            cat_overlaps_bowl(c, bowl, self.min_overlap_ratio) for c in cats
        )
        event = self.drinking_detector.update(now, cat_in_roi)
        if event is None:
            return None
        clip_path = self.recorder.save_clip(self.frame_buffer.all_frames(), event.timestamp)
        self.stats.record_event(event.timestamp, clip_path.name)
        return clip_path.name
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS（2 passed）。

- [ ] **Step 5: Commit**

```bash
git add catcam/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): 单帧编排 采集→认猫→判定→录制→统计"
```

---

### Task 11: 主程序 `app`（接真摄像头 + 跑网页 + 全绿回归）

**Files:**
- Create: `catcam/app.py`
- Create: `catcam/__main__.py`
- Create: `README.md`
- Test: 复用既有全部测试做回归（本任务不新增单测；`main()` 是薄装配层，靠前面已测的纯逻辑保证正确性）。

**Interfaces:**
- Consumes: 以上所有模块
- Produces:
  - `class LatestFrame`：线程间共享最新帧；`set(frame)` / `get()`。
  - `main(config_path: str = "config.json") -> None`：装配依赖、起网页线程、跑采集循环。
  - `python -m catcam` 可启动。

- [ ] **Step 1: 写 `LatestFrame` 与 `app.py`**

Create `catcam/app.py`:

```python
from __future__ import annotations

import threading
import time

import cv2
import uvicorn

from catcam.config import load_config
from catcam.detector import DrinkingDetector
from catcam.feedback import FeedbackStore
from catcam.framebuffer import FrameBuffer
from catcam.pipeline import Pipeline
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.vision import CatDetector
from catcam.web import create_app


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _serve_web(app, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)

    cat_detector = CatDetector.from_path(cfg.yolo_model, cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
    pipeline = Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(cfg.dwell_seconds, cfg.cooldown_seconds),
        frame_buffer=FrameBuffer(cfg.clip_seconds, cfg.fps),
        recorder=recorder,
        stats=stats,
        bowl_roi_ratio=cfg.bowl_roi,
        min_overlap_ratio=cfg.min_overlap_ratio,
    )

    latest = LatestFrame()
    app = create_app(stats, recorder, feedback, latest.get, cfg.clips_dir)
    threading.Thread(
        target=_serve_web, args=(app, cfg.web_host, cfg.web_port), daemon=True
    ).start()
    print(f"网页已启动： http://127.0.0.1:{cfg.web_port}")

    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 index={cfg.camera_index}")

    interval = 1.0 / max(1, cfg.fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(interval)
                continue
            now = time.time()
            latest.set(frame)
            clip = pipeline.process(now, frame)
            if clip:
                print(f"记录一次喝水： {clip}")
            time.sleep(interval)
    finally:
        cap.release()
```

- [ ] **Step 2: 写 `__main__.py`**

Create `catcam/__main__.py`:

```python
from catcam.app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证 `LatestFrame` 线程安全行为（临时内联测试）**

Create `tests/test_app_latestframe.py`:

```python
import numpy as np
from catcam.app import LatestFrame


def test_latest_frame_roundtrip():
    lf = LatestFrame()
    assert lf.get() is None
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    lf.set(frame)
    got = lf.get()
    assert got is not None
    # get() 返回拷贝，改动原帧不影响已取出的
    frame[0, 0, 0] = 99
    assert got[0, 0, 0] == 0
```

- [ ] **Step 4: 跑该测试确认通过**

Run: `.venv/bin/pytest tests/test_app_latestframe.py -v`
Expected: PASS（1 passed）。

- [ ] **Step 5: 写 `README.md`**

Create `README.md`:

```markdown
# 猫咪饮水摄像头（MVP）

纯本地的猫咪饮水监控：普通 USB 摄像头 + Mac Mini，YOLO 认猫在流动水碗停留即录 2–5 秒视频，只留最近 10 段，记录每日喝水次数，本地网页可看/下载并对每段点 👍/👎 攒训练数据。画面不出本机。

## 安装
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
```
首次运行会自动下载一次 YOLO 权重（`yolov8n.pt`）。

## 运行
```bash
.venv/bin/python -m catcam
```
然后浏览器打开 `http://127.0.0.1:8000`（远程时先远程进本机再打开）。

## 配置
首次运行生成 `config.json`，可改：
- `camera_index`：摄像头编号（多摄像头时调 0/1/2…）
- `bowl_roi`：水碗区域，0–1 比例 `[x1, y1, x2, y2]`（先用默认中心框，对不准就按画面比例改）
- `dwell_seconds` / `cooldown_seconds`：停留判定秒数 / 两次事件最小间隔
- `clip_seconds` / `max_clips`：每段时长 / 最多保留段数

## 测试
```bash
.venv/bin/pytest -q
```

## 已知范围（MVP）
仅白天/有光时工作（无红外）；会把"好奇凑近没喝"误计（后续训练舌头模型修正）；隐私时段自动关闭、网页画框设区域、开机自启为后续阶段。
```

- [ ] **Step 6: 全量回归，确认所有测试通过**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS（约 36 passed），无 fail。

- [ ] **Step 7: 真机冒烟（手动，一次性）**

Run:
```bash
.venv/bin/python -m catcam
```
Expected: 控制台打印「网页已启动」；浏览器开 `http://127.0.0.1:8000` 能看到实时画面；人为在镜头前模拟"停留"超过 `dwell_seconds` 后控制台打印「记录一次喝水」，网页今日次数 +1 且出现一段可播放/下载的视频；点 👍/👎 出现"已记录反馈"。按 Ctrl+C 退出。

- [ ] **Step 8: Commit**

```bash
git add catcam/app.py catcam/__main__.py tests/test_app_latestframe.py README.md
git commit -m "feat(app): 主程序装配 摄像头采集循环+网页线程"
```

---

## Self-Review（作者自检结果）

**1. Spec 覆盖**
- 纯本地认猫认水碗触发 → Task 4(vision) + Task 2(geometry) + Task 10(pipeline)；明确不用帧差。✅
- 停留 N 秒判喝水 + 冷却 → Task 3(detector)。✅
- 录 2–5 秒 mp4 + 只留最近 10 段 → Task 6(recorder) + Task 5(framebuffer)，`clip_seconds`/`max_clips` 可配。✅
- 每日喝水时间+次数 → Task 7(stats) + 网页今日统计。✅
- 本地网页 看/下载视频 + 今日统计 → Task 9(web)。✅
- 👍/👎 反馈 + 训练帧入永久库（与 10 段视频分离）→ Task 8(feedback) + Task 9 路由；帧存 `training_dir`，视频存 `clips_dir`，互不影响。✅
- 远程靠用户远程进本机、不做穿透 → 网页绑定 `web_host`，文档说明，无穿透代码。✅
- 仅白天/有光 → 不依赖红外，无夜视代码（符合范围）。✅
- 隐私自动关闭/网页画框/开机自启 → 明确列为后续阶段，MVP 不实现（YAGNI）。✅

**2. 占位符扫描**：无 TBD/TODO；每个代码步骤均含完整可运行代码。✅

**3. 类型/命名一致性**：
- `Rect = (x1,y1,x2,y2)` 全程一致；`DrinkingDetector.update(now, cat_in_roi)`、`DrinkingEvent.timestamp` 在 Task 3 定义、Task 10 使用一致。
- `ClipRecorder.save_clip/list_clips`、`clip_filename`/`prune_dir` 在 Task 6 定义、Task 9/10 使用一致。
- `StatsStore.record_event/count_between/events_between`、`day_bounds` 在 Task 7 定义、Task 9/10 使用一致。
- `FeedbackStore.label_clip(clip_path, is_drinking)`/`get_label` 在 Task 8 定义、Task 9 使用一致（web 传 `clips_dir / clip` 路径，签名匹配）。
- `CatDetector.detect_cats`/`from_path`、`filter_cat_boxes` 在 Task 4 定义、Task 10/11 使用一致。
- `create_app(stats, recorder, feedback, frame_provider, clips_dir)` 在 Task 9 定义、Task 11 调用一致。

无不一致项。
