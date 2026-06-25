# AI 自动标注设计（录一段 → 自动判「喝/没喝」→ 直接进训练）

日期：2026-06-26
状态：已通过设计评审，待写实现计划

## 背景与目标

当前模型「很垃圾」，根因不是标注太累，而是 **训练样本严重失衡**：👍35 / 👎350（约 1:10），
真喝水的正样本太少，且训练没做类别平衡——和 CLAUDE.md 记的教训一致（5👍/80👎 训出来的模型
几乎把一切都判「没喝」）。

用户处于「测试期」，目标是 **尽快把模型训好**。手段：接入外部视觉大模型（OpenRouter，
OpenAI 兼容接口）对每段录像 **全自动** 判「猫是否真在喝水」并直接写入训练数据，省掉人工标注
一大堆无用片段。模型训好后这套外部 AI 即可关闭撤掉，因此设计上要 **干净可拆、默认关闭**。

非目标（本 spec 不含，留作后续独立 spec）：趋势页重设计、按时长估喝水量/换水提醒、视频录音。

## 关键决策（已与用户确认）

- **信任度**：AI 直接标注、直接进训练（全自动，不需人工逐段确认）。
- **触发时机**：录一段自动标一段（会话录完即后台调用 AI）。
- **类别平衡**：纳入本次——训练时把多数类降采样/少数类过采样到接近 1:1。
- **默认模型**：`google/gemma-4-31b-it:free`（OpenRouter 上已标视觉输入，免费）。base_url/key/model 全可配。
- **兜底**：AI 标注同时存「理由 + 置信度」，视频页可见、可手动翻转；人工 > AI（已人工标的段 AI 不覆盖）。

## 数据流

在现有采集链路尾部挂一个后台标注（与发邮件同款 fire-and-forget，绝不阻塞采集/录制）：

```
会话 _finish 录完一段 mp4
  → app.py：若 ai_label_enabled 且配了 key，起后台线程 AILabeler.label(clip_path)
       ├─ feedback.extract_frames 从 mp4 均匀抽 N 帧（默认 3）
       ├─ build_messages：N 帧 base64 → OpenAI 兼容 vision 消息，问「猫是不是在真喝水」
       ├─ client.chat.completions.create(...) → content
       ├─ parse_label(content) → {drinking, confidence, reason}
       └─ FeedbackStore.label_clip(clip, drinking, source="ai", confidence, reason)
              内部：写 labels 表 + 抽帧进 data/training/{drinking,not_drinking}
```

失败处理：任何异常（网络/限流/解析失败）只打日志，该段保持「未标注」状态——你仍可手动标，
下次也可重试。**绝不影响录制本身。**

## 组件设计

### 1. 新模块 `catcam/ai_labeler.py`

职责：把一段 clip 的若干帧送给视觉大模型，得到「喝/没喝 + 置信度 + 理由」。

- 客户端：用官方 `openai` SDK 的 `OpenAI(base_url=..., api_key=...)`（与用户给的代码一致）。
- **纯函数（可单测，不碰网络）**：
  - `build_messages(frames_b64: list[str]) -> list[dict]`：组装 vision 提示消息。提示词要求
    模型 **严格输出 JSON** `{"drinking": bool, "confidence": 0-1, "reason": "<简短中文>"}`，并明确判据：
    「舌头/嘴接触水面、低头舔水 = 喝；只是凑近/嗅闻/路过/趴着 = 没喝」。
  - `parse_label(content: str) -> dict`：从模型回复抠出 JSON，容忍 ```json 围栏、前后杂质、
    大小写/真假值变体；解析失败抛 `ValueError`（由调用方吞掉记日志）。
  - `encode_frames(frame_paths) -> list[str]`：读 jpg → base64 data URL。
- **类 `AILabeler`**：
  - `__init__(self, store, training_dir, model, frames, client=None, ...)`——`client` 可注入，
    测试塞假 client。
  - `label(self, clip_path) -> dict | None`：抽帧 → 调模型 → parse → 写库；返回标注结果或 None（失败）。
  - 调模型带 **一次 429 退避重试**（免费模型限流时）。

### 2. 存储变更 `catcam/feedback.py`

`labels` 表加三列（沿用现有 `PRAGMA table_info` + `ALTER TABLE` 迁移写法，老库平滑升级）：
- `source TEXT`——`"human"` / `"ai"`，区分来源。
- `confidence REAL`——AI 的置信度（人工标注为 NULL）。
- `reason TEXT`——AI 的简短理由（人工标注为 NULL）。

`label_clip` 签名扩展为 `label_clip(clip_path, is_drinking, max_frames=5, source="human", confidence=None, reason=None)`：
- 写入时带上来源三列；人工翻转时 `source="human"`（理由/置信度清空）。
- **人工 > AI**：AILabeler 在 label 前先查 `get_label`/来源——若该段已是 `source="human"`，跳过不覆盖。
  （AI 标过的段 AI 可重标；人工标过的段 AI 不动。）
- 维持现有行为：写库即 `trained_version=NULL`（AI 标注同样计入「已标注未训练」，可被训练）。

副作用（正向）：AI 把误触判成 `not_drinking` 后，因 stats 用 `COALESCE(is_drinking,1)<>0` 过滤，
「今日喝水次数」会自动剔除这些假触发——次数更准。

### 3. 接入 `catcam/app.py`

会话 `res = session.update(...)` 返回非 None（录完一段）后，在现有 `_email_async` 旁边加一个
`_ai_label_async(res.clip_name)`：若 `cfg.ai_label_enabled` 且 `cfg.ai_api_key` 非空，起 daemon 线程
调 `ai_labeler.label(clips_dir / clip_name)`。串行（每段一个短任务，冷却 60s 节流，不会并发刷爆）。

### 4. 配置 `catcam/config.py` 新增字段

```python
ai_label_enabled: bool = False                       # 开关，默认关（外发画面，需显式开）
ai_base_url: str = "https://openrouter.ai/api/v1"
ai_api_key: str = ""                                 # OpenRouter key，写在 config.json（已 gitignore）
ai_model: str = "google/gemma-4-31b-it:free"
ai_label_frames: int = 3                             # 每段送几帧给模型
```

### 5. 网页 `catcam/web.py`（视频页）

- `/api/clips` 每段多带 `source` / `confidence` / `ai_reason`。
- 每段显示来源徽标：`🤖 AI：喝水 0.82 · 理由…` 或 `✋ 人工`；未标注的无徽标。
- 原有 👍/👎 翻转按钮不变；翻转即写成 `source="human"`。

### 6. 训练类别平衡 `catcam/trainer.py`

`prepare_dataset` 在拷贝进 train/ 前做类别平衡，避免自动标注把负样本越堆越多导致失衡加剧：
- 取两类训练张数的较小值为基准，多数类 **降采样** 到接近少数类（或少数类过采样到多数类，二选一，
  默认降采样多数类，简单且不引入重复过拟合）。
- 仅作用于 `train/` 划分；`val/` 保持真实分布以反映真实准确率。
- 行为可关（保守起见加一个内部参数，默认开）。纯函数部分（按目标数量挑选文件）单独可单测。

### 7. 依赖

`openai` Python SDK 加入 `pyproject.toml` / `requirements.txt`。

## 错误处理

- AI 调用失败（网络/超时/429/非法 JSON）→ 记日志，该段不写标注，保持未标注。
- 429 限流 → 一次退避重试；仍失败则放弃该段。
- key 未配或开关关 → 整条 AI 链路不启动（等价于功能不存在）。
- 抽帧为空（坏 mp4）→ 跳过。

## 测试（沿用「纯函数 + 可注入依赖」约定，全部不依赖网络/真模型/摄像头）

- `parse_label`：标准 JSON、```json 围栏、前后有杂质、缺字段、真假值变体、彻底非法（抛 ValueError）。
- `build_messages`：消息结构含 N 张图、提示词含判据。
- `encode_frames`：jpg → base64 data URL 前缀正确。
- `AILabeler.label`：注入假 client 返回固定 JSON → 断言 labels 表写入 + 训练目录抽帧 + 来源/置信度/理由落库。
- 人工 > AI：已 `source="human"` 的段，AILabeler 跳过不覆盖。
- 类别平衡：构造 8👍/40👎 → 平衡后 train 两类张数接近、val 保持原分布。

## 隐私 / 成本（重要）

- ⚠️ 开启后 **猫的画面帧会上传到 OpenRouter 外部服务器**，违背项目「视频/画面只在本机与局域网内
  流转」的核心原则。因此 **默认关闭，需用户在 config.json 显式开启**，并在文档/CLAUDE.md 注明。
- key 已明文出现在聊天记录中，建议接通后到 OpenRouter 后台轮换。
- 免费模型有每日额度，超额标注失败（不影响录制）。测试期用，模型训好后关闭即可。

## 验收标准

1. config 开关关闭时，行为与现状完全一致（无外发、无新线程）。
2. 开关开 + 配 key 后，录完一段约数秒内该段在视频页出现 `🤖 AI` 徽标，labels 表有对应来源/置信度/理由。
3. 已人工标注的段，AI 不覆盖。
4. AI 判「没喝」的段，从「今日喝水次数」中剔除。
5. 训练时两类样本接近 1:1（train 划分），验证准确率反映真实分布。
6. 断网/错 key 下录制照常，仅该段未标注，无崩溃。
7. 新增测试全绿，`pytest -q` 通过。
