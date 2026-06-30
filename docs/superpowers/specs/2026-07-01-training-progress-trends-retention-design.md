# 训练进度条 + 趋势页扩充 + 保留上限文案 设计

日期：2026-07-01

三个互不依赖的网页改进，一并实现。

## Part 1 — 视频训练进度条

**问题**：网页能触发的训练只有「训练视频模型」（单帧训练按钮早已移除）。`VideoTrainingManager`
只暴露 `state` + 一句静态 `detail`，而真正耗时的是**逐段 s3d 特征抽取**（外加首次 ~30MB s3d 权重下载），
用户几分钟看不到任何变化。

**后端**（`videojudge.gather_dataset` → `video_trainer`）：
- `gather_dataset` 增可选 `progress_cb`。总数 = 已标注 clip 数（循环前已知）。回调阶段：
  - `{"phase":"preparing"}`：构造/下载 s3d 提取器时（抽第一段前）。
  - `{"phase":"extracting","done":k,"total":N}`：每处理完一段（含被跳过的）即报。
  - `{"phase":"training"}`：进入（很快的）logistic 头训练。
- `train_video_head` 把 `progress_cb` 透传给 `gather_dataset`；提取器构造移到能发 `preparing` 的位置。
- `VideoTrainingManager` 加锁存 `_phase/_done/_total`，`status()` 增 `phase`、`done`、`total`、
  `progress`（0–1，= done/total，training 阶段为 1.0）。回调里抛异常一律吞掉，绝不拖累训练。

**前端**（`pollTrainVideo`）：running 时渲染真进度条（填充轨 + 文案）：
`加载 s3d 模型…` / `抽取特征 12/47 段` / `训练分类器…`。沿用 2s 轮询。

单帧 `TrainingManager` 已有进度字段但无触发入口 → 不在本次范围。

## Part 2 — 趋势页四组件

**一个新端点** `/api/stats/trend?days=N`，一次取齐，页面只发一个请求：
- `days`：每日次数 `[{date,count}]`（沿用 `daily_counts`）。
- `total`、`active_days`（≥1 次喝水的天数）。
- `prev_total`：上一个等长窗口的总次数（环比）。
- `hourly`：长度 24 的计数（窗口内各小时）。
- `weekday`：长度 7（周一→周日）。

分桶逻辑放 `stats.py` 纯函数 `bucket_events(events, ...) -> {"hourly":[...], "weekday":[...]}`（可单测）；
端点薄封装，复用 `events_between`（取窗口内事件）+ `daily_counts`。**全部沿用既有口径**：只数
`labels.is_drinking = 1`（`events_between` 已是该口径）。

**布局**（range 切换下方）：
```
[KPI 卡片排：期间总计 · 日均 · 单日最多 · 活跃天数]
本期 42 次  ↑17% 比上期(36)多 6 次              ← 环比一行
[每日喝水次数（现有 drawChart 柱状图）]
[时段分布(24h)]   [星期分布(周一→日)]            ← 两张新 mini 图并排
```
两张新图用小 SVG 函数 `drawMini`，复用现有渐变/圆角柱样式，视觉与主图一致。

## Part 3 — 视频页保留上限文案（纯显示修正）

后端保留逻辑**已正确**（`config.max_clips=1000`；`app.py` 注入
`is_deletable=lambda n: feedback.get_label(n) is False`；`recorder.prune_dir` 最旧优先、只删「没喝」，
喝水/未判定永不删）——本次**不动逻辑**。

只修过期硬编码文案 `web.py` 的「最多保留 100 段」：
- 后端：`/api/clips` 响应加 `max_clips`（取 `recorder.max_clips`）。
- 前端：`renderClips` 用该值动态渲染，并写明策略：
  「最多保留 **1000** 段 · 超量先删最旧的『没喝』，**喝水/未判定永不删**」。

## 测试

沿用仓库约定（纯函数 + 可注入依赖，不依赖摄像头/YOLO 权重）：
- `stats.bucket_events`：构造若干事件 ts → 断言 hourly/weekday 分桶。
- 视频训练进度：注入假 extractor，断言 `progress_cb` 收到 preparing/extracting/training 序列、
  `VideoTrainingManager.status()` 跑完含 progress 字段。
- Part 3 文案：可加端点测试断言 `/api/clips` 含 `max_clips`。

`pytest -q` 全绿，三部分分开提交。
