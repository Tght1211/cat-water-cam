# 视频动作裁判：云 VLM 标注工厂 → 本地小视频模型（VideoMAE 冻结特征 + 小头）

日期：2026-06-26
状态：已和用户确认方向，待 spec 评审

## 背景与动机

现在的分类器弱，根因不是素材少，而是**模型种类选错**：

- `trainer.py` 训的是 `yolov8n-cls`，一个**单帧图片分类器**。`prepare_dataset` 把每段 mp4 抽成一张张独立 jpg 来训。
- 但「喝水」是个**时间上的动作**——舌头一伸一缩地舔。单看一帧，「在舔水」和「只是盯着碗」几乎一样。
- **我们明明录了完整 mp4，却在训练时把时间维度整个扔掉。** 这才是模型弱的真正原因。

用户提出「微调小视频模型来准确识别」。方向对，但需修正一点：OpenRouter 上那些 9B~31B 的通用 VLM
**本地既微调不动、也跑不起来**，只能云端调用（画面要出本机）。真正适合「认出喝水这个动作」的是
**小型视频动作识别模型**，而非 VLM。

部署机器是 **Mac mini / 16G 内存 / 约 30G 磁盘**：跑小视频模型**推理**没问题，但**全量微调不现实**。
因此采用「**冻结主干 + 只训小头**」：主干只前向不反传，内存/算力/磁盘都吃得下。

## 目标 / 非目标

**目标**
1. 把「发邮件 + 记喝水次数」的权威从「兜底检测器」改为「**整段视频裁判的『真喝水』判定**」。
2. 第一阶段：**云 VLM 当裁判**（零训练、立刻能准），同时当**免费标注工厂**自动攒标签。
3. 第二阶段：用 VLM 攒的标签，在本地训「**VideoMAE 冻结特征 + 小头**」视频模型，shadow 评估达标后
   **接管裁判**，画面不再出本机（完全本地、免费、私有）。
4. 复用现有 `registry` / `shadow`·`gate` 版本与模式机制；保持「网页标注 → 一键训练 → 自我迭代」的 UX。

**非目标**
- 不微调大 VLM、不本地跑大 VLM。
- 不做实时逐帧视频识别（裁判只在**会话录完后**对整段跑一次，不卡预览/录制）。
- 兜底检测器（YOLO/简单模型）继续负责「**何时录一段候选**」，本设计不动其召回职责。

## 架构总览

```
摄像头/视频 ─► 兜底检测(YOLO/简单) ─► 会话录制(session.py) ─► 录完一段 clip
                                                              │
                                              ┌───────────────┘
                                              ▼
                                     ClipJudge.judge(clip)  ← 整段裁判（新核心抽象）
                                       │  返回 {drinking, confidence, reason, by}
                                       ├─ 写 labels（source=ai/local，喂训练）
                                       ├─ drinking=True → 发邮件 + 该段计入「喝水次数」
                                       └─ drinking=False/失败 → 只留素材，不发、不计
```

裁判有两种实现，**同一接口 `judge(clip_path) -> Verdict`**：

- `VLMClipJudge`：包住现有 `ai_labeler.py` 的调用（多模型池轮换 + 兜底已实现）。第一阶段的裁判。
- `LocalVideoClipJudge`：VideoMAE 冻结特征 + 小头。第二阶段，本地、私有。

「当前生效的裁判」与模式由 `registry` 管理，沿用 `active` / `active_mode`：
- 无本地模型时 → 裁判 = VLM。
- 有本地模型且 `shadow` → 裁判仍是 VLM；本地模型**只预测、记录**，与 VLM/人工标注对比算命中率。
- 有本地模型且 `gate`（达标后手动切）→ 裁判 = 本地模型；**VLM 可关停**（画面不出本机）。

> 这是把现有「兜底 + 测试模型」的 shadow→gate 思路，从「逐帧分类器是否拦录制」平移到
> 「整段裁判由谁说了算」。recording 永远不被裁判压制（兜底始终录素材）；裁判只决定**发邮件 + 计数**。

## 组件设计

### 1. `ClipJudge` 抽象与裁判选择（新）

新模块 `judge.py`：

- `Verdict = {drinking: bool, confidence: float|None, reason: str, by: str}`（`by` 记录是哪个裁判/版本判的）。
- `ClipJudge.judge(clip_path) -> Verdict | None`（None = 该段无法判，fail-open：不发不计、保持未标注）。
- `VLMClipJudge(ai_labeler)`：调 `ai_labeler` 拿判定（复用其抽帧/模型池/解析）。
- `LocalVideoClipJudge(extractor, head, version)`：见下。
- `select_judge(registry, ...)`：按 registry 的 active/mode 决定「权威裁判」是谁、本地模型是否 shadow 预测。

`app.py` 在会话录完那一处（`_capture` 里 `res is not None` 分支）改为：调权威裁判 → 写标注 → 据
`drinking` 决定 `_email_async` 与计数 → 若有 shadow 本地模型则额外 `predict` 记 `events.predicted`。

### 2. `LocalVideoClipJudge`：VideoMAE 冻结特征 + 小头（新，第二阶段）

新模块 `videojudge.py`：

- **特征提取器** `VideoFeatureExtractor`（懒加载，省常驻内存）：
  - 用 `transformers` 的 `VideoMAEImageProcessor` + `VideoMAEModel`（基座 `MCG-NJU/videomae-base`，约 87M）。
  - 输入：从 clip 用 **opencv**（项目已依赖，免引 decord）均匀抽 16 帧 → 处理器预处理 → 主干前向 →
    取 `last_hidden_state` 池化（mean over tokens）得 **768 维特征向量**。主干权重**冻结**，只前向。
  - 设备：优先 MPS，回退 CPU。每段约 0.5~2s，**会话录完后台跑**，不卡实时。
- **小头** `DrinkingHead`：768 维 → 二分类。
  - 实现：`sklearn.linear_model.LogisticRegression(class_weight="balanced")`（训练几秒、自带类别平衡）。
    *（决策点见下：是否引入 sklearn 依赖；备选是 numpy/torch 手写逻辑回归，免新依赖。）*
  - `predict(feat) -> (drinking: bool, confidence: float)`（confidence 取 `predict_proba`）。
  - 存成小文件（`head_<ts>.pkl` 或 `.npz`，几 KB）登记进 registry，与 VideoMAE 主干名一起记。
- **纯函数拆分**（沿用项目测试惯例）：抽帧、池化、`predict` 逻辑都做成可注入/可单测；
  `VideoMAEModel` 用假 extractor 注入，测试不依赖真权重/网络。

### 3. 特征缓存与「一键训练小头」（改 `trainer.py` 思路 / 新 `video_trainer.py`）

- **特征缓存**：每段 clip 的 768 维特征是**不随标注变化的**（只跟像素有关），抽一次缓存进
  `data/training/features/<clip_stem>.npy`（约 3KB/段）。标注/录完时顺手抽，避免每次训练重跑主干。
- **训练小头** = 读所有缓存特征 + 各自当前 `labels.is_drinking` → 拟合 `LogisticRegression`。
  - 标注可被人工改（人工 > AI），所以训练时**实时 join 当前标签**，不固化。
  - 几秒训完，16G/30G 毫无压力；登记一个 `vN` 版本（top1 用留出验证集算），**不自动生效**，网页手动启用。
  - 复用 `registry.add` / `feedback.mark_trained` / 训练页状态机；`label_states` 的「已标注未训练/已训练」逻辑照旧。
- 训练页区分两类模型基座（`base` 字段已有）：旧的 `yolov8n-cls` 与新的 `videomae+head`，UI 标明。

### 4. 计数与邮件口径改动（据用户确认）

> 用户拍板：计数「**只数 AI/人工确认喝水的**」；邮件只在裁判判「真喝水」时发。

- **`stats.py` 计数口径翻转**：`count_between` / `events_between` / 趋势的过滤从
  `COALESCE(l.is_drinking, 1) <> 0`（未标注默认算 1）改为 **`l.is_drinking = 1`**（必须有「喝水」标注才计）。
  - 影响：未被裁判确认的事件（含历史遗留、裁判还没跑完/失败的）**不计入**喝水次数。这是用户要的「严格只数确认的」。
  - `events.predicted` 仍记 shadow 本地模型预测，用于命中率评估（不变）。
- **邮件闸门**：`app.py` 把 `_email_async` 从「录完即发」移到「**裁判 drinking=True 才发**」分支。
  - 邮件正文照片仍用会话触发照（`res.photo`）；趋势/次数因口径翻转自然变成「确认喝水」口径。
  - `Emailer.should_send` 的最小间隔限流保留（双保险）。

### 5. 配置项（`config.py`）

- `ai_label_enabled` 默认改 **True**（用户确认默认开 AI）。**仍受 `ai_api_key` 把关**：没填 key 时
  `AILabeler.from_config` 返回 None，等于不动作——所以默认 True 是安全的（无 key 不上传）。
  README/CLAUDE.md 需更新此权衡说明（画面会上传外部，换取准确度 + 自动标注）。
- 新增（第二阶段，给默认值即可，先不强开）：
  - `video_judge_enabled: bool = False`（本地视频裁判总开关）
  - `video_base_model: str = "MCG-NJU/videomae-base"`
  - `video_judge_frames: int = 16`
- 沿用 `config.py` 的 dataclass 即 schema 约定，新增项只改 dataclass。

## 数据流（录完一段后）

1. `session.update` 返回 `res`（clip 存盘）。
2. `select_judge` 取权威裁判：VLM 或本地模型。
3. `judge(clip)` → `Verdict`。
4. 写 `labels`（`source='ai'` 或 `'local'`，带 confidence/reason）+ 抽帧/缓存特征进训练目录。
   - 人工 > AI/local：已人工标注的段不覆盖（`label_source == 'human'` 时跳过，逻辑已存在）。
5. `drinking=True` → `record_event` 计入 + `_email_async`；`False` → 只记录为素材，不发不计。
6. 若存在 shadow 本地模型且权威裁判是 VLM → 额外 `LocalVideoClipJudge.predict` 写 `events.predicted` 供评估。

## 错误处理

- 裁判 fail-open：VLM 调用失败 / 抽帧为空 / 主干加载失败 → `judge` 返回 None → **该段保持未标注、不发不计**
  （不污染训练集、不误发邮件）。仅记日志。沿用 `ai_labeler` 现有 fail-open 风格。
- VideoMAE 主干懒加载失败（缺权重/内存不足）→ 本地裁判降级，回退 VLM 或纯兜底；绝不崩采集循环。
- 所有外部调用与重活都在**后台线程**（已有 `_ai_label_async` 模式），不阻塞采集/预览/录制。

## 测试策略（沿用「纯函数 + 可注入依赖」）

- `judge.py`：`select_judge` 用假 registry 测分支；`VLMClipJudge`/`LocalVideoClipJudge` 用假底层注入。
- `videojudge.py`：抽帧/池化/`DrinkingHead.predict` 纯函数单测；`VideoMAEModel` 用假 extractor，不碰真权重。
- `video_trainer.py`：用合成特征 + 标签测小头训练与版本登记；不依赖真模型/GPU。
- `stats.py`：补测新计数口径（未标注不计、仅 `is_drinking=1` 计）。
- 不依赖真摄像头 / YOLO / VideoMAE 权重 / 网络——与现有测试约定一致。

## 风险与权衡

- **磁盘 30G 最紧**：增量 = `transformers`（torch 已装）+ VideoMAE 权重（约 350MB）+ 可选 sklearn（约 30MB）
  ≈ 1~1.5GB，装得下但余量不大。clips 靠现有 `max_clips` 裁剪盯紧；特征缓存极小（~3KB/段）。
- **内存 16G**：运行期可能同时驻留 YOLO + VideoMAE（约 3~4GB）。VideoMAE **懒加载**（判一段才载），降常驻。
- **MPS 兼容**：个别算子可能回退 CPU；只前向、不训练，影响有限。失败回退 CPU。
- **第一阶段隐私权衡**：默认开 AI = 画面帧上传外部。这是用户明确接受的换取项；第二阶段本地模型接管后可关 VLM 回归全本地。
- **sklearn 依赖**：引入则训练最省心；不引入则用 numpy/torch 手写逻辑回归（免依赖、但要自测）。→ 待定，见下。

## 待定决策（实现前定）

1. 小头实现：`sklearn LogisticRegression`（省心、自带类别平衡）vs numpy/torch 手写（免新依赖）。
2. 第二阶段是否本迭代就做，还是先只交付第一阶段（VLM 当裁判 + 闸门 + 计数口径），第二阶段下一轮再做。

## 分阶段交付

- **阶段一（本质是已有 AI 标注的延伸）**：`ClipJudge` 抽象 + `VLMClipJudge`；邮件/计数闸到「真喝水」；
  计数口径翻转；`ai_label_enabled` 默认开。**立刻能准、能自动攒标签。**
- **阶段二**：`LocalVideoClipJudge`（VideoMAE 冻结特征 + 小头）+ 特征缓存 + 一键训小头 + shadow→gate 接管。
  攒够约 100~200 段标注后训练、评估、切换到本地裁判。
