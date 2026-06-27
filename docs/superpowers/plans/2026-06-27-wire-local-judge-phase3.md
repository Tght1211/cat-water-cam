# 把本地视频裁判接进 app（shadow→gate）第三阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 registry 里 `base=s3d+head` 的版本能接进采集链路：`shadow`=影子评估（VLM 仍当权威），`gate`=本地当「发邮件+记次数」权威且 VLM 不再被调（画面不出本机）。**没有 s3d+head 生效时，行为与第一阶段逐字节一致。**

**Architecture:** 新增纯函数 `judge.route_clip` 按模式编排「谁标注 / 谁判邮件计数 / 谁影子预测」，app.py 在会话录完的后台线程里调它。两条新支撑：`stats.set_prediction`（影子预测回填 events）、`feedback.record_machine_label`（gate 模式本地权威写计数标签、不抽训练帧、不覆盖人工/AI）。`video_trainer` 排除 `source='local'` 标签避免自我强化。web activate 认识 `s3d+head`。

**Tech Stack:** 现有 `catcam`（`judge`/`stats`/`feedback`/`videojudge`/`models`/`web`/`app`）、SQLite、pytest。无新依赖。

设计依据：`docs/superpowers/specs/2026-06-26-video-action-judge-design.md`（含环境校正附录）。

---

## 模式语义（实现时的唯一真相）

| registry.active | mode | 标注(写labels) | 邮件+计数权威 | 影子预测 | VLM 是否上传画面 |
|---|---|---|---|---|---|
| 无 / yolo 版本 | — | VLM（source=ai） | VLM | 单帧 active_model（如有） | 是（若开 AI） |
| s3d+head 版本 | shadow | VLM（source=ai） | VLM | **本地视频模型** → events.predicted | 是 |
| s3d+head 版本 | gate | **本地**（source=local，不抽帧、不覆盖人工/AI） | **本地视频模型** | — | **否（不调 VLM）** |

> 计数口径仍是 `labels.is_drinking=1`。gate 模式下本地权威写 `source='local'` 的标签来驱动计数；
> `video_trainer` 排除 `source='local'`，绝不拿自己的判定当训练真值（防自我强化）。
> shadow 模式 VLM 照常写 `source='ai'` 标签、当权威——本地只影子评估，攒「实战命中率」直到够准。

---

## File Structure

- **改** `catcam/stats.py`：加 `set_prediction(clip_name, predicted, predicted_by)`。
- **改** `catcam/feedback.py`：加 `record_machine_label(...)`（写 labels 行、不抽帧、不覆盖 human/ai）。
- **改** `catcam/video_trainer.py`：`_labeled_clips` 排除 `source='local'`。
- **改** `catcam/judge.py`：加 `route_clip(...)` 编排。
- **改** `catcam/app.py`：启动按 registry 加载本地视频裁判；`_judge_async` 改调 `route_clip`。
- **改** `catcam/web.py`：activate 认识 `s3d+head`（不塞进单帧 active_model；提示重启生效）。
- 新增/补测：`tests/test_stats.py`、`tests/test_feedback.py`、`tests/test_video_trainer.py`、`tests/test_judge.py`、`tests/test_web.py`。

---

## Task 1: stats.set_prediction —— 影子预测回填事件

**Files:**
- Modify: `catcam/stats.py`（加方法）
- Test: `tests/test_stats.py`

- [ ] **Step 1: 写失败测试（追加到 tests/test_stats.py）**

```python
def test_set_prediction_updates_event(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(100.0, "clip_p.mp4")          # 初始无预测
    store.set_prediction("clip_p.mp4", 1, "v5")
    assert store.clip_predictions().get("clip_p.mp4") is True
    hr = store.model_hitrate("v5")                    # 还没标注 → total 0
    assert hr["total"] == 0
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_stats.py::test_set_prediction_updates_event -q`
Expected: FAIL —— `AttributeError: 'StatsStore' object has no attribute 'set_prediction'`。

- [ ] **Step 3: 实现 set_prediction**

在 `catcam/stats.py` 的 `record_event` 方法后加：

```python
    def set_prediction(self, clip_name: str, predicted: int, predicted_by: str | None) -> None:
        """回填某段的「测试模型预测」（影子模式：录完后台判完再写）。更新该 clip 的事件行。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET predicted = ?, predicted_by = ? WHERE clip_name = ?",
                (int(predicted), predicted_by, clip_name),
            )
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_stats.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add catcam/stats.py tests/test_stats.py
git commit -m "feat(stats): set_prediction——影子模式录完回填事件预测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: feedback.record_machine_label + video_trainer 排除 source='local'

**Files:**
- Modify: `catcam/feedback.py`（加方法）
- Modify: `catcam/video_trainer.py`（`_labeled_clips` 加过滤）
- Test: `tests/test_feedback.py`、`tests/test_video_trainer.py`

- [ ] **Step 1: 写失败测试（feedback）**

追加到 `tests/test_feedback.py`：

```python
def test_record_machine_label_no_frames_and_respects_human(tmp_path):
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    # 机器标（source=local）：写进 labels 但不抽训练帧
    store.record_machine_label("m.mp4", True, source="local", confidence=0.7, reason="本地判")
    assert store.get_label("m.mp4") is True
    assert store.label_source("m.mp4") == "local"
    assert not (tmp_path / "training" / "drinking").exists()   # 没抽帧
    # 人工标过的不被机器覆盖
    import cv2, numpy as np
    clip = tmp_path / "h.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(4): w.write(np.full((16, 16, 3), 9, np.uint8))
    w.release()
    store.label_clip(clip, True)                               # 人工=喝水
    store.record_machine_label("h.mp4", False, source="local")  # 机器想改成没喝
    assert store.get_label("h.mp4") is True                    # 仍是人工的
    assert store.label_source("h.mp4") == "human"
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_feedback.py::test_record_machine_label_no_frames_and_respects_human -q`
Expected: FAIL —— `AttributeError: ... 'record_machine_label'`。

- [ ] **Step 3: 实现 record_machine_label**

在 `catcam/feedback.py` 的 `label_clip` 后加：

```python
    def record_machine_label(self, clip_name: str, is_drinking: bool, source: str = "local",
                             confidence: float | None = None, reason: str | None = None) -> bool:
        """机器判定写 labels 行（驱动计数/邮件），**不抽训练帧**（判定不是训练真值）。

        不覆盖人工/AI 标注（human/ai 优先）。返回是否写入。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source FROM labels WHERE clip_name = ?", (clip_name,)
            ).fetchone()
            if row is not None and row[0] in ("human", "ai"):
                return False
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts, trained_version, source, confidence, reason) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?) "
                "ON CONFLICT(clip_name) DO UPDATE SET "
                "is_drinking=excluded.is_drinking, ts=excluded.ts, source=excluded.source, "
                "confidence=excluded.confidence, reason=excluded.reason",
                (clip_name, 1 if is_drinking else 0, time.time(), source, confidence, reason),
            )
        return True
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_feedback.py -q`
Expected: PASS。

- [ ] **Step 5: 写失败测试（video_trainer 排除 local）**

追加到 `tests/test_video_trainer.py`：

```python
def test_gather_excludes_source_local(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    _clip(clips / "ai.mp4", 200); store.label_clip(clips / "ai.mp4", True)         # source=human
    _clip(clips / "loc.mp4", 200); store.record_machine_label("loc.mp4", True, source="local")
    X, y, names = gather_dataset(clips, training, store, _FakeExtractor(8), dim=8)
    assert "loc.mp4" not in names and "ai.mp4" in names     # 本地判定不进训练集
```

- [ ] **Step 6: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_video_trainer.py::test_gather_excludes_source_local -q`
Expected: FAIL —— `loc.mp4` 仍混进训练集。

- [ ] **Step 7: 在 _labeled_clips 加过滤**

把 `catcam/video_trainer.py` 的 `_labeled_clips` 改为：

```python
def _labeled_clips(store) -> list[tuple[str, int]]:
    """从 labels 表取所有 (clip_name, is_drinking)；排除 source='local'（机器自身判定，不当训练真值）。"""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT clip_name, is_drinking FROM labels WHERE source IS NULL OR source != 'local'"
        ).fetchall()
    return [(name, int(v)) for name, v in rows]
```

- [ ] **Step 8: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_video_trainer.py -q`
Expected: PASS。

- [ ] **Step 9: 提交**

```bash
git add catcam/feedback.py catcam/video_trainer.py tests/test_feedback.py tests/test_video_trainer.py
git commit -m "feat(feedback): record_machine_label(不抽帧/不覆盖人工AI) + 训练排除 source=local

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: judge.route_clip —— 按模式编排

**Files:**
- Modify: `catcam/judge.py`（加 `route_clip`）
- Test: `tests/test_judge.py`

- [ ] **Step 1: 写失败测试（追加到 tests/test_judge.py）**

```python
from catcam.judge import route_clip


class _FakeLocalJudge:
    def __init__(self, drinking, version="v5"):
        self._v = drinking; self.version = version; self.judged = []
    def judge(self, clip_path):
        from catcam.judge import Verdict
        self.judged.append(str(clip_path))
        return Verdict(drinking=self._v, confidence=0.9, reason="", by=self.version)


class _FakeStats:
    def __init__(self): self.preds = []
    def set_prediction(self, clip, pred, by): self.preds.append((clip, pred, by))


class _FakeFeedback:
    def __init__(self): self.machine = []
    def record_machine_label(self, clip, drink, source="local", confidence=None, reason=None):
        self.machine.append((clip, drink, source)); return True


def test_route_shadow_vlm_authority_local_shadow_predicts():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": True, "confidence": 0.8, "reason": "舔水"})  # VLM 判喝水
    local = _FakeLocalJudge(False, "v5")                                       # 本地判没喝
    r = route_clip(clip_path="a.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
                   local_judge=local, mode="shadow", emailer=em, stats=st, feedback=fb)
    assert r["emailed"] is True            # VLM 是权威、判喝水 → 发
    assert lab.labeled == ["a.mp4"]        # VLM 写了标注
    assert st.preds == [("a.mp4", 0, "v5")]  # 本地影子预测回填(没喝=0)
    assert fb.machine == []                # shadow 不写机器计数标签


def test_route_gate_local_authority_no_vlm():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": False})    # 即便 VLM 会判没喝……
    local = _FakeLocalJudge(True, "v5")        # 本地判喝水（gate 权威）
    r = route_clip(clip_path="b.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
                   local_judge=local, mode="gate", emailer=em, stats=st, feedback=fb)
    assert r["emailed"] is True                       # 本地权威判喝水 → 发
    assert lab.labeled == []                          # gate 不调 VLM（画面不上传）
    assert fb.machine == [("b.mp4", True, "local")]   # 本地写计数标签
    assert st.preds == []                             # gate 不另记影子


def test_route_no_local_is_phase1():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"})
    route_clip(clip_path="c.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
               local_judge=None, mode="shadow", emailer=em, stats=st, feedback=fb)
    assert len(em.sent) == 1 and lab.labeled == ["c.mp4"] and st.preds == []


def test_route_gate_notdrinking_silent():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    local = _FakeLocalJudge(False, "v5")
    route_clip(clip_path="d.mp4", start_ts=1700000000.0, photo="P", ai_labeler=None,
               local_judge=local, mode="gate", emailer=em, stats=st, feedback=fb)
    assert em.sent == [] and fb.machine == [("d.mp4", False, "local")]
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_judge.py -q`
Expected: FAIL —— `ImportError: cannot import name 'route_clip'`。

- [ ] **Step 3: 实现 route_clip**

在 `catcam/judge.py` 末尾加：

```python
def route_clip(*, clip_path, start_ts: float, photo, ai_labeler, local_judge, mode: str,
               emailer, stats, feedback, log=print) -> dict:
    """会话录完后按模式编排：谁标注 / 谁判邮件计数 / 谁影子预测。返回各动作结果（供测试/日志）。

    - gate + 有本地模型：本地当权威，写 source='local' 计数标签，**不调 VLM**（画面不出本机）。
    - 否则（shadow / 无本地）：VLM 当权威+标注（第一阶段行为）；有本地模型则只影子预测、回填 events。
    """
    authority = None
    emailed = False
    shadow_pred = None

    if mode == "gate" and local_judge is not None:
        authority = local_judge.judge(clip_path)
        if authority is not None:
            feedback.record_machine_label(
                Path_name(clip_path), authority.drinking, source="local",
                confidence=authority.confidence, reason=authority.reason,
            )
    else:
        if ai_labeler is not None:
            authority = VLMClipJudge(ai_labeler, log).judge(clip_path)
        if local_judge is not None:
            v = local_judge.judge(clip_path)
            if v is not None:
                stats.set_prediction(Path_name(clip_path), int(v.drinking), local_judge.version)
                shadow_pred = v.drinking

    if authority is not None and authority.drinking and photo is not None:
        emailer.notify_drinking(stats, photo, datetime.fromtimestamp(start_ts))
        emailed = True
    return {"authority": authority, "emailed": emailed, "shadow_pred": shadow_pred}
```

并在 `catcam/judge.py` 顶部 import 区加一个取文件名的小助手（clip_path 可能是 str 或 Path）：

```python
from pathlib import Path as _Path


def Path_name(p) -> str:
    return _Path(p).name
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_judge.py -q`
Expected: PASS（含原 8 个 + 新 4 个）。

- [ ] **Step 5: 提交**

```bash
git add catcam/judge.py tests/test_judge.py
git commit -m "feat(judge): route_clip——按 shadow/gate 编排标注/权威/影子预测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: app.py 接线（启动加载本地裁判 + _judge_async 走 route_clip）

**Files:**
- Modify: `catcam/app.py`

> 安全不变量：`registry.active` 不是 s3d+head 时 `local_video_judge=None`，`route_clip` 走 else 分支 =
> 第一阶段行为逐字节一致。

- [ ] **Step 1: import route_clip**

把 `catcam/app.py` 的 `from catcam.judge import VLMClipJudge, judge_and_notify` 改为：

```python
from catcam.judge import VLMClipJudge, route_clip
```

（`judge_and_notify` 不再直接用；保留在 judge.py 供测试/兼容，不必删。）

- [ ] **Step 2: 启动按 registry 加载本地视频裁判**

在 `catcam/app.py` 里 `judge = VLMClipJudge(...) ...` 那段之后、`emailer = Emailer(cfg)` 之前，加：

```python
    # 本地视频裁判（s3d+head）：仅当 registry 当前生效版本是视频模型时加载。
    # shadow=影子评估（VLM 仍权威）；gate=本地当权威且不再调 VLM。其余情况 = None（第一阶段行为）。
    local_video_judge = None
    local_video_mode = "shadow"
    _active = registry.get(registry.active_id()) if registry.active_id() else None
    if _active and _active.get("base") == "s3d+head":
        try:
            from catcam.videojudge import S3DFeatureExtractor, DrinkingHead, LocalVideoClipJudge
            _head = DrinkingHead.load(_active["path"])
            local_video_judge = LocalVideoClipJudge(S3DFeatureExtractor(), _head, _active["id"])
            local_video_mode = registry.active_mode()
            tip = "本地当权威、不调 VLM" if local_video_mode == "gate" else "影子评估、VLM 仍权威"
            print(f"本地视频裁判已加载：{_active['id']}（{local_video_mode} · {tip}）")
        except Exception as e:  # noqa: BLE001
            print(f"本地视频裁判加载失败，忽略（继续用 VLM）：{e}")
```

- [ ] **Step 3: _judge_async 改调 route_clip**

把 `catcam/app.py` 的 `_judge_async` 整体替换为：

```python
    def _judge_async(clip_name: str, start_ts: float, photo) -> None:
        # 会话录完后台编排：按模式决定谁标注/谁判邮件计数/谁影子预测。绝不阻塞采集。
        if judge is None and local_video_judge is None:
            return
        def _run():
            try:
                route_clip(
                    clip_path=cfg.clips_dir / clip_name, start_ts=start_ts, photo=photo,
                    ai_labeler=ai_labeler, local_judge=local_video_judge,
                    mode=local_video_mode, emailer=emailer, stats=stats, feedback=feedback,
                )
            except Exception as e:  # noqa: BLE001
                print(f"AI 裁判流程异常（{clip_name}）：{e}")
        threading.Thread(target=_run, daemon=True).start()
```

- [ ] **Step 4: 冒烟 + 全量（确认装配、无残留引用）**

Run: `grep -n "judge_and_notify" catcam/app.py` → 应为空。
Run: `.venv/bin/python -c "import catcam.app" && .venv/bin/pytest tests/test_smoke.py tests/test_app_latestframe.py -q`
Expected: import OK；PASS。

- [ ] **Step 5: 提交**

```bash
git add catcam/app.py
git commit -m "feat(app): 接入本地视频裁判——启动按 registry 加载 + 会话录完走 route_clip

无 s3d+head 生效时与第一阶段行为一致。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: web activate 认识 s3d+head

**Files:**
- Modify: `catcam/web.py`（`/api/model/activate`）
- Test: `tests/test_web.py`

> 现状：activate 对任何版本都 `DrinkingClassifier.from_path(path)`（YOLO），对 `.npz` 头会抛错 500。
> 改：`base=s3d+head` 的版本只登记 active+mode、清掉单帧 active_model，并提示「重启后由视频裁判生效」。

- [ ] **Step 1: 写失败测试（追加到 tests/test_web.py）**

```python
def test_activate_s3d_head_version_does_not_500(tmp_path):
    app, stats, recorder, feedback, registry, active_model = _build_with_registry(tmp_path)
    # 登记一个 s3d+head 版本（造个假头文件）
    head_path = tmp_path / "videohead_1.npz"; head_path.write_bytes(b"x")
    registry.add(path=head_path, top1=0.9, image_counts={"drinking": 5, "not_drinking": 5},
                 label_counts=None, base="s3d+head", epochs=300, imgsz=224, created_ts=1.0)
    client = TestClient(app)
    r = client.post("/api/model/activate", json={"id": "v1", "mode": "shadow"})
    assert r.status_code == 200
    assert registry.active_id() == "v1" and registry.active_mode() == "shadow"
```

> 注：若 `tests/test_web.py` 现有 `_build` 不返回 registry/active_model，则加一个 `_build_with_registry`
> 辅助（构造 `ModelRegistry` + `ActiveModel` 注入 `create_app`）。参照文件顶部现有 `_build` 的注入方式照搬，
> 多传 `registry=` 与 `active_model=`。

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_web.py::test_activate_s3d_head_version_does_not_500 -q`
Expected: FAIL —— 500（试图把 .npz 当 YOLO 加载）。

- [ ] **Step 3: 改 activate 分支**

把 `catcam/web.py` 的 activate 里 `else:` 块（`model_id is None` 的否则分支）改为：

```python
        if model_id is None:
            active_model.clear()
        else:
            entry = registry.get(model_id)
            if entry and entry.get("base") == "s3d+head":
                # 视频模型：不塞进单帧 active_model；清掉单帧模型，视频裁判在重启后按 registry 生效。
                active_model.clear()
                return {"active": registry.active_id(), "mode": registry.active_mode(),
                        "note": "视频模型已登记生效，重启采集进程后由本地视频裁判接管"}
            path = registry.active_path()
            if not path or not Path(path).exists():
                raise HTTPException(status_code=404, detail="模型文件丢了")
            try:
                active_model.set(DrinkingClassifier.from_path(path), model_id, mode)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"加载失败：{e}")
        return {"active": registry.active_id(), "mode": registry.active_mode()}
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add catcam/web.py tests/test_web.py
git commit -m "feat(web): activate 认识 s3d+head 版本(登记生效、重启由视频裁判接管，不再 500)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 文档 + 全量回归

**Files:**
- Modify: `CLAUDE.md`、`README.md`

- [ ] **Step 1: CLAUDE.md**

在「本地视频模型」条目末尾把「本轮不改采集/裁判热路径」那句改成已接入说明：

```markdown
  - **已接入裁判（shadow→gate）**：`judge.route_clip` 按模式编排——`shadow`=本地影子评估（回填
    `events.predicted` 供 `model_hitrate`，VLM 仍当邮件/计数权威）；`gate`=本地当权威、写 `source='local'`
    计数标签且**不再调 VLM**（画面不出本机）。`app.py` 启动按 registry 当前版本加载（**切视频模型需重启采集进程**）。
    没有 s3d+head 生效时行为与第一阶段一致。`video_trainer` 排除 `source='local'` 防自我强化。
```

- [ ] **Step 2: README.md**

在「进阶（本地视频模型）」那段末尾加一句：

```markdown
> 训好评估满意后：在「模型版本」把该 `s3d+head` 版本设为生效——先 **测试模式(shadow)** 看实战命中率，够准
> 再切 **过滤模式(gate)** 让本地接管发邮件/计数（gate 下不再调用云 VLM，画面回归全本地）。切视频模型需重启采集进程。
```

- [ ] **Step 3: 全量回归**

Run: `.venv/bin/pytest -q`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md README.md
git commit -m "docs: 本地视频裁判已接入(shadow→gate)说明

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review 记录

- **Spec/语义覆盖**：模式表三行分别由 route_clip 的 gate 分支、else+影子分支、else 分支实现（Task 3）；影子回填（Task 1）、本地计数标签不抽帧不覆盖（Task 2）、排除 source=local（Task 2）、app 启动加载+路由（Task 4）、web 不再 500（Task 5）。
- **安全不变量**：无 s3d+head 生效 → `local_video_judge=None` → route_clip else 分支 → 与第一阶段逐字节一致（Task 4 Step 2 守卫 + Task 3 `test_route_no_local_is_phase1` 验证）。
- **占位符**：无 TBD；每步含完整代码/命令/期望输出。
- **类型一致**：`route_clip(*, clip_path,start_ts,photo,ai_labeler,local_judge,mode,emailer,stats,feedback,log)`；`local_judge.judge(clip)->Verdict`、`.version` 字段；`stats.set_prediction(clip,pred,by)`；`feedback.record_machine_label(clip,drink,source,confidence,reason)->bool`；`Verdict.by/drinking/confidence/reason` 一致；`registry.get(id)['base']`/`active_path()` 与 models.py 一致。
- **边界**：local.judge 失败→None（route 不发不写）；gate 不调 VLM；human/ai 标签不被机器覆盖；切视频模型需重启（已在 web 返回 note + 文档注明）。
