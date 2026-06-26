# VLM 整段裁判 + 邮件/计数闸门（第一阶段）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「发邮件 + 记喝水次数」的权威改成「云 VLM 对录完的整段视频判『真喝水』」——只有 AI 判喝水才发邮件、才计入次数；其余只留作素材。

**Architecture:** 会话录完一段后，新增的 `ClipJudge`（第一阶段实现 `VLMClipJudge`，包住现有 `ai_labeler`）对整段判喝/没喝并写标注；判「喝水」才发邮件。计数口径翻转为「只数有 `is_drinking=1` 标注的事件」。本阶段不引入本地视频模型（第二阶段做）。

**Tech Stack:** Python、现有 `catcam` 包、`ai_labeler`（OpenAI 兼容 VLM 调用，已具模型池轮换/兜底）、SQLite、pytest。

设计依据：`docs/superpowers/specs/2026-06-26-video-action-judge-design.md`（第一阶段部分）。

---

## File Structure

- **新建** `catcam/judge.py`：`Verdict` 数据类 + `VLMClipJudge`（包 `ai_labeler`）+ `judge_and_notify`（判后按结果发邮件，纯逻辑、可注入假对象单测）。
- **新建** `tests/test_judge.py`：覆盖上面三者。
- **改** `catcam/stats.py`：`count_between` / `events_between` 的过滤条件翻转为 `l.is_drinking = 1`。
- **改** `tests/test_stats.py`：按新口径更新断言（加 `labels` 行才计数）。
- **改** `catcam/app.py`：`main` 里用 `VLMClipJudge` 包 `ai_labeler`，会话录完走 `judge_and_notify`（替掉原「无条件发邮件 + fire-and-forget AI 标注」）；无裁判时打印告警；旧 `detect` 路径同样改为判后发。
- **改** `catcam/config.py`：`ai_label_enabled` 默认 `True`（仍受 `ai_api_key` 把关）。
- **改** `tests/test_config.py`：`test_ai_label_defaults` 断言改为 `True`。
- **改** `README.md` + `CLAUDE.md`：说明「AI 当裁判 + 闸门」「计数口径」「默认开 AI 的隐私权衡」。

---

## Task 1: 计数口径翻转（stats.py）

**Files:**
- Modify: `catcam/stats.py:81-117`（`count_between` 与 `events_between`）
- Test: `tests/test_stats.py`

> 口径：从 `COALESCE(l.is_drinking, 1) <> 0`（未标注默认算 1）改为 `l.is_drinking = 1`（必须有「喝水」标注才计）。未标注 / 标「没喝」/ 无 clip 的事件都不计入。

- [ ] **Step 1: 改写 stats 测试为新口径**

替换 `tests/test_stats.py` 中 `test_record_and_count` 与 `test_events_between_sorted_with_fields` 两个函数，并在文件顶部加一个写 `labels` 行的小helper（直接写同库的 `labels` 表，正是被 JOIN 的表）：

```python
import sqlite3
from datetime import datetime
from catcam.stats import StatsStore, day_bounds


def _label(db_path, clip_name, is_drinking):
    """直接往同一个库的 labels 表写一行（StatsStore 也会 CREATE 这张表）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, ?, 0) "
            "ON CONFLICT(clip_name) DO UPDATE SET is_drinking=excluded.is_drinking",
            (clip_name, 1 if is_drinking else 0),
        )


def test_only_confirmed_drinking_counts(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(100.0, "a.mp4")   # 标「喝水」→ 计
    store.record_event(150.0, "b.mp4")   # 标「没喝」→ 不计
    store.record_event(180.0, "c.mp4")   # 未标注 → 不计（口径翻转）
    _label(db, "a.mp4", True)
    _label(db, "b.mp4", False)
    assert store.count_between(0.0, 1000.0) == 1
    assert store.count_between(0.0, 120.0) == 1
    assert store.count_between(120.0, 1000.0) == 0


def test_events_between_only_confirmed_sorted(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(300.0, "b.mp4")
    store.record_event(100.0, "a.mp4")
    store.record_event(200.0, "u.mp4")   # 未标注 → 不出现
    _label(db, "a.mp4", True)
    _label(db, "b.mp4", True)
    events = store.events_between(0.0, 1000.0)
    assert [e["ts"] for e in events] == [100.0, 300.0]
    assert events[0]["clip_name"] == "a.mp4"
```

（保留 `test_day_bounds` 与 `test_record_returns_id` 不动。）

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv/bin/pytest tests/test_stats.py -q`
Expected: FAIL —— 旧 SQL 把未标注事件也算进去，`count`/`events` 数目对不上新断言。

- [ ] **Step 3: 翻转 `count_between` 的过滤**

把 `catcam/stats.py` 的 `count_between` 改为：

```python
    def count_between(self, start_ts: float, end_ts: float) -> int:
        # 「真实喝水」= 被 AI/人工明确标注为「喝水」(is_drinking=1) 的事件。
        # 口径：只数确认喝水的——未标注 / 标「没喝」/ 无对应 clip 的事件都不计。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND l.is_drinking = 1",
                (start_ts, end_ts),
            )
            return int(cur.fetchone()[0])
```

- [ ] **Step 4: 翻转 `events_between` 的过滤**

把 `catcam/stats.py` 的 `events_between` 改为：

```python
    def events_between(self, start_ts: float, end_ts: float) -> list[dict]:
        # 同 count_between：只列被确认「喝水」的事件，让时间点列表与计数一致。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT e.ts, e.clip_name FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND l.is_drinking = 1 "
                "ORDER BY e.ts ASC",
                (start_ts, end_ts),
            )
            return [{"ts": row[0], "clip_name": row[1]} for row in cur.fetchall()]
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `.venv/bin/pytest tests/test_stats.py tests/test_stats_trend.py -q`
Expected: PASS。若 `test_stats_trend.py` 因新口径失败，按其测法补 `labels` 行（趋势同样只数确认喝水的）。

- [ ] **Step 6: 提交**

```bash
git add catcam/stats.py tests/test_stats.py
git commit -m "feat(stats): 计数口径翻转——只数 AI/人工确认喝水的事件

未标注/标没喝的事件不再计入今日次数与趋势。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `judge.py` —— Verdict + VLMClipJudge + judge_and_notify

**Files:**
- Create: `catcam/judge.py`
- Test: `tests/test_judge.py`

> `ai_labeler.label(clip)` 已经做了：跳过人工已标段、抽帧、模型池调用、解析、写 `labels`、抽训练帧，
> 返回 `{"drinking","confidence","reason"}` 或 `None`（人工已标/抽帧空），全模型失败时 raise。
> `VLMClipJudge` 把它适配成 `Verdict`，并把 raise 吞成 `None`（fail-open）。

- [ ] **Step 1: 写失败测试 `tests/test_judge.py`**

```python
from datetime import datetime
from catcam.judge import Verdict, VLMClipJudge, judge_and_notify


class _FakeLabeler:
    """假的 ai_labeler：按预设返回 dict / None / 抛错，并记录被调的 clip。"""
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.labeled = []

    def label(self, clip_path):
        self.labeled.append(str(clip_path))
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeEmailer:
    def __init__(self):
        self.sent = []

    def notify_drinking(self, stats, frame, now_dt):
        self.sent.append((frame, now_dt))
        return True


def test_judge_returns_verdict_on_drinking():
    lab = _FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"})
    v = VLMClipJudge(lab).judge("a.mp4")
    assert v == Verdict(drinking=True, confidence=0.9, reason="舔水", by="vlm")
    assert lab.labeled == ["a.mp4"]


def test_judge_returns_notdrinking_verdict():
    lab = _FakeLabeler({"drinking": False, "confidence": 0.7, "reason": "只凑近"})
    v = VLMClipJudge(lab).judge("b.mp4")
    assert v.drinking is False and v.confidence == 0.7


def test_judge_none_when_labeler_skips():
    # ai_labeler 返回 None（人工已标 / 抽帧空）→ 裁判无判定
    assert VLMClipJudge(_FakeLabeler(None)).judge("c.mp4") is None


def test_judge_fail_open_on_exception():
    # 全模型失败 ai_labeler 会 raise → 裁判吞成 None（不发不计、不崩）
    logs = []
    j = VLMClipJudge(_FakeLabeler(raises=RuntimeError("429")), log=logs.append)
    assert j.judge("d.mp4") is None
    assert logs and "d.mp4" in logs[0]


def test_judge_and_notify_emails_only_on_drinking():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"}))
    v = judge_and_notify(j, em, stats="S", clip_path="a.mp4",
                         start_ts=1700000000.0, photo="PHOTO")
    assert v.drinking is True
    assert len(em.sent) == 1
    frame, now_dt = em.sent[0]
    assert frame == "PHOTO" and isinstance(now_dt, datetime)


def test_judge_and_notify_silent_on_notdrinking():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": False, "confidence": 0.8, "reason": "路过"}))
    judge_and_notify(j, em, "S", "b.mp4", 1700000000.0, "PHOTO")
    assert em.sent == []


def test_judge_and_notify_silent_when_no_verdict():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler(None))
    judge_and_notify(j, em, "S", "c.mp4", 1700000000.0, "PHOTO")
    assert em.sent == []


def test_judge_and_notify_no_email_without_photo():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"}))
    judge_and_notify(j, em, "S", "a.mp4", 1700000000.0, None)
    assert em.sent == []
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv/bin/pytest tests/test_judge.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'catcam.judge'`。

- [ ] **Step 3: 写 `catcam/judge.py`**

```python
"""整段视频裁判：录完一段后判「猫是否真喝水」，作为「发邮件 + 记次数」的唯一权威。

第一阶段实现 VLMClipJudge——包住 ai_labeler（外部视觉大模型，已具模型池轮换/兜底）。
第二阶段会再加 LocalVideoClipJudge（本地 VideoMAE 冻结特征 + 小头），同一 judge() 接口。
判定 fail-open：拿不到判定 → 该段不发邮件、不计数、保持未标注（绝不污染训练集 / 误发）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Verdict:
    drinking: bool
    confidence: float | None
    reason: str
    by: str            # 哪个裁判判的：'vlm' / 本地模型版本号等


class VLMClipJudge:
    """用外部视觉大模型当裁判：调 ai_labeler 判一段并写标注，适配成 Verdict。

    ai_labeler.label(clip) 返回 {"drinking","confidence","reason"} 或 None（人工已标/抽帧空），
    全模型失败时 raise——这里吞成 None（fail-open）。标注写入由 ai_labeler 内部完成。
    """

    def __init__(self, ai_labeler, log=print):
        self.ai_labeler = ai_labeler
        self.log = log

    def judge(self, clip_path) -> Verdict | None:
        try:
            result = self.ai_labeler.label(clip_path)
        except Exception as e:  # noqa: BLE001 —— 裁判失败绝不能崩采集/误发
            self.log(f"AI 裁判失败（{clip_path}）：{e}")
            return None
        if result is None:
            return None
        return Verdict(
            drinking=bool(result["drinking"]),
            confidence=result.get("confidence"),
            reason=str(result.get("reason", "") or ""),
            by="vlm",
        )


def judge_and_notify(judge, emailer, stats, clip_path, start_ts: float, photo, log=print):
    """对一段跑裁判；判「真喝水」且有照片才发邮件。返回 Verdict 或 None。

    纯流程、无线程——调用方负责放后台线程。写标注在 judge.judge() 内部已完成。
    """
    verdict = judge.judge(clip_path)
    if verdict is not None and verdict.drinking and photo is not None:
        emailer.notify_drinking(stats, photo, datetime.fromtimestamp(start_ts))
    return verdict
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv/bin/pytest tests/test_judge.py -q`
Expected: PASS（8 passed）。

- [ ] **Step 5: 提交**

```bash
git add catcam/judge.py tests/test_judge.py
git commit -m "feat(judge): ClipJudge 抽象 + VLMClipJudge——整段裁判当邮件/计数权威

判后只在「真喝水」时发邮件；裁判失败 fail-open（不发不计不崩）。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 接线 app.py —— 会话录完走裁判，判喝水才发邮件

**Files:**
- Modify: `catcam/app.py`（import；`main` 里造 judge；`_capture` 会话分支；旧 `detect` 分支；启动告警）

> 关键改动：把原来「录完即无条件 `_email_async` + fire-and-forget `_ai_label_async`」改成
> 「录完 → 后台线程跑 `judge_and_notify`」。`record_event` + 影子 `predict` 保持不变（事件历史/评估照旧）。

- [ ] **Step 1: 加 import**

在 `catcam/app.py` 顶部 import 区（`from catcam.ai_labeler import AILabeler` 附近）加：

```python
from catcam.judge import VLMClipJudge, judge_and_notify
```

- [ ] **Step 2: 在 main 里造裁判，并在无裁判时告警**

在 `catcam/app.py` 里 `ai_labeler = AILabeler.from_config(feedback, cfg)` 之后那段，改成：

```python
    ai_labeler = AILabeler.from_config(feedback, cfg)  # 未启用/缺 key 返回 None
    if ai_labeler is not None:
        print(f"AI 自动标注已开启 → {cfg.ai_model}（⚠️ 画面帧会上传外部服务器）")
    # 整段裁判：第一阶段用 VLM（包 ai_labeler）。它是「发邮件 + 记次数」的唯一权威——
    # 判「真喝水」才发、才计入；没启用 AI（无 key）则没有裁判，不会发邮件、次数恒 0。
    judge = VLMClipJudge(ai_labeler) if ai_labeler is not None else None
    if judge is None and cfg.record_session:
        print("⚠️ 未启用 AI 裁判（config 里 ai_api_key 为空）："
              "不会发喝水邮件、今日喝水次数为 0。喝水判定依赖 AI——请填 ai_api_key。")
```

- [ ] **Step 3: 替换会话录完分支的发信/标注逻辑**

把 `catcam/app.py` `_capture` 里这段（`res is not None` 分支）：

```python
                    if res is not None:
                        print(f"记录一段喝水： {res.clip_name}")
                        # 测试模型对这段的预测（不影响是否录，仅记下来供评估）。
                        pred = active_model.predict(res.photo)
                        stats.record_event(
                            res.timestamp, res.clip_name,
                            predicted=None if pred is None else int(pred),
                            predicted_by=active_model.active_id,
                        )
                        _email_async(res.timestamp, res.photo)
                        _ai_label_async(res.clip_name)
```

改为：

```python
                    if res is not None:
                        print(f"录到一段候选： {res.clip_name}（等 AI 裁判判是否真喝水）")
                        # 影子模型对这段的预测（不影响判定，仅记下来供评估）。
                        pred = active_model.predict(res.photo)
                        stats.record_event(
                            res.timestamp, res.clip_name,
                            predicted=None if pred is None else int(pred),
                            predicted_by=active_model.active_id,
                        )
                        # 整段裁判 → 判「真喝水」才发邮件 + 计入次数；放后台线程绝不阻塞采集。
                        _judge_async(res.clip_name, res.timestamp, res.photo)
```

- [ ] **Step 4: 用 `_judge_async` 取代 `_email_async` / `_ai_label_async`**

把 `catcam/app.py` 里 `_email_async` 与 `_ai_label_async` 两个内嵌函数整体替换为一个 `_judge_async`：

```python
    def _judge_async(clip_name: str, start_ts: float, photo) -> None:
        # 整段裁判含外部 API 往返，放后台线程绝不阻塞采集。
        # 裁判内部写标注；判「真喝水」且有照片才发邮件（限流仍在 emailer 内）。
        if judge is None:
            return
        def _run():
            try:
                judge_and_notify(judge, emailer, stats, cfg.clips_dir / clip_name,
                                 start_ts, photo)
            except Exception as e:  # noqa: BLE001
                print(f"AI 裁判流程异常（{clip_name}）：{e}")
        threading.Thread(target=_run, daemon=True).start()
```

- [ ] **Step 5: 旧 `detect` 路径也改为「判后发」**

把 `catcam/app.py` 检测循环里旧模式那段：

```python
            elif not blocked:
                clip = pipeline.detect(now, frame, night=night)
                if clip:
                    print(f"记录一次喝水： {clip}")
                    _email_async(now, frame.copy())
```

改为：

```python
            elif not blocked:
                clip = pipeline.detect(now, frame, night=night)
                if clip:
                    print(f"录到一段候选： {clip}（等 AI 裁判）")
                    _judge_async(clip, now, frame.copy())
```

- [ ] **Step 6: 跑冒烟测试，确认 app 仍能 import / 装配**

Run: `.venv/bin/pytest tests/test_smoke.py tests/test_app_latestframe.py -q`
Expected: PASS。（若 `test_smoke` 对 import 名敏感，确认没有残留的 `_email_async` / `_ai_label_async` 引用：`grep -n "_email_async\|_ai_label_async" catcam/app.py` 应为空。）

- [ ] **Step 7: 提交**

```bash
git add catcam/app.py
git commit -m "feat(app): 会话录完走整段裁判——AI 判真喝水才发邮件/计数

替掉「录完即无条件发邮件 + fire-and-forget 标注」；无 AI key 时启动告警。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 默认开启 AI（config）

**Files:**
- Modify: `catcam/config.py:56`
- Test: `tests/test_config.py:41-47`

> 默认 `True` 是安全的：`AILabeler.from_config` 仍要求 `ai_api_key` 非空才真正动作，没填 key = 不上传。

- [ ] **Step 1: 改测试断言**

把 `tests/test_config.py` 的 `test_ai_label_defaults` 第一行断言改为：

```python
def test_ai_label_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.json")
    assert cfg.ai_label_enabled is True   # 默认开（仍受 ai_api_key 把关，无 key 不动作）
    assert cfg.ai_base_url == "https://openrouter.ai/api/v1"
    assert cfg.ai_api_key == ""
    assert cfg.ai_model == "google/gemma-4-31b-it:free"
    assert cfg.ai_label_frames == 3
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `.venv/bin/pytest tests/test_config.py::test_ai_label_defaults -q`
Expected: FAIL —— 当前默认仍是 `False`。

- [ ] **Step 3: 改默认值与注释**

把 `catcam/config.py` 中：

```python
    ai_label_enabled: bool = False
```

改为：

```python
    # 默认开：AI 是「发邮件 + 记次数」的裁判。仍受 ai_api_key 把关——没填 key 则不动作、不上传。
    # ⚠️ 填了 key 开始判定后，猫的画面帧会上传外部视觉大模型，违背「画面只在本机/局域网」原则。
    ai_label_enabled: bool = True
```

（删掉原本那两行 `# ⚠️ 开启后…默认关，需显式开。` 旧注释，避免与新默认矛盾。）

- [ ] **Step 4: 运行测试，确认通过**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add catcam/config.py tests/test_config.py
git commit -m "feat(config): ai_label_enabled 默认开——AI 当裁判（仍受 ai_api_key 把关）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 文档更新（README + CLAUDE.md）

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 CLAUDE.md 的 AI 标注段**

在 `CLAUDE.md` 的「AI 自动标注（可选、默认关）」一节，改成反映新角色（节选要点，按现有行文融入）：

- 标题与口径改为「**AI 自动标注 / 整段裁判（默认开，需填 `ai_api_key`）**」。
- 说明：录完一段后 `catcam/judge.py` 的 `VLMClipJudge` 调外部视觉大模型判「真喝水」，**它是「发邮件 + 记喝水次数」的唯一权威**——判「喝水」才发邮件、才计入次数；判「没喝」/ 失败则该段只留作素材，不发不计。
- 说明计数口径：`stats.py` 的 `count_between`/`events_between` 现在要求 `labels.is_drinking = 1` 才计（未标注/没喝都不计）。
- 保留隐私警示：开启（填 key）后画面帧会上传外部服务器；第二阶段本地视频模型接管后可关停 VLM 回归全本地。
- 指向 spec：`docs/superpowers/specs/2026-06-26-video-action-judge-design.md`。

- [ ] **Step 2: 更新 README 的对应说明**

在 `README.md` 里讲「喝水判定 / 邮件 / 训练」的相应位置，加一段「AI 当裁判」的说明（与 CLAUDE.md 口径一致）：现在准确识别依赖外部视觉大模型当裁判，需在 `config.json` 填 `ai_api_key`（OpenRouter 兼容）；未来计划用本地小视频模型（VideoMAE）接管，画面回归全本地。

- [ ] **Step 3: 提交**

```bash
git add README.md CLAUDE.md
git commit -m "docs: AI 当裁判 + 计数口径 + 默认开 AI 的隐私权衡

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 全量回归

**Files:** 无（只跑测试）

- [ ] **Step 1: 跑完整测试套件**

Run: `.venv/bin/pytest -q`
Expected: 全绿。重点确认没有别的测试依赖旧计数口径（如 `test_web.py` / `test_stats_trend.py` / `test_demo.py`）；若有，按新口径（需 `is_drinking=1` 才计）补 `labels` 行或调整断言，**不要**改回旧 SQL。

- [ ] **Step 2: 若有改动则提交**

```bash
git add -A
git commit -m "test: 适配新计数口径的回归修复

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review 记录

- **Spec 覆盖**：第一阶段四要点——ClipJudge+VLMClipJudge（Task 2）、邮件闸到真喝水（Task 2/3）、计数口径翻转（Task 1）、默认开 AI（Task 4）+ 文档（Task 5）。第二阶段（本地 VideoMAE）明确不在本计划。
- **占位符**：无 TBD；每步含完整代码/命令/期望输出。
- **类型一致**：`Verdict(drinking, confidence, reason, by)` 在 Task 2 定义并在 `judge_and_notify`/测试中一致使用；`VLMClipJudge.judge` / `judge_and_notify` 签名前后一致；app.py 只引用 `VLMClipJudge`、`judge_and_notify`。
- **边界**：无 AI key → 无裁判 → 不发不计（Task 3 Step 2 启动告警提示），与「只数 AI 确认」口径自洽。
