# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

纯本地的猫咪饮水监控：USB 摄像头 → 认猫（YOLO + 启发式简单模型）→ 猫在水碗区域停留即触发，回放缓冲区里最近几秒帧录成 mp4，记录每日喝水次数。网页平台（可局域网访问）看实时画面/趋势/视频并对每段点 👍/👎 攒训练数据，攒够后一键训练分类器（自我迭代）。猫来喝水时发邮件提醒（照片+今日次数+周/月趋势）。**视频/画面只在本机与同局域网内流转**。注释与面向用户的字符串用中文，保持一致。

## Commands

```bash
# 环境（首次）
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt   # 首次运行会自动下载一次 yolov8n.pt

# 真实采集（需摄像头）
.venv/bin/python -m catcam

# 演示/试标注（无摄像头，自动塞 3 段示例视频，只起网页）
.venv/bin/python -m catcam.demo

# 测试
.venv/bin/pytest -q
.venv/bin/pytest tests/test_pipeline.py -q          # 单文件
.venv/bin/pytest tests/test_pipeline.py::test_name  # 单测
```

打开 `http://127.0.0.1:8000` 看网页。没接摄像头但想跑完整识别/录制链路时，在 `config.json` 的 `video_source` 填一段录好的视频文件路径即可。

## Architecture

数据流是一条单向链，`catcam/app.py:main` 把各组件装配起来后跑采集循环；网页跑在 daemon 线程上，与采集循环之间靠 `LatestFrame`（带锁）和 SQLite 共享状态。

```
摄像头/视频 ──frame──► Pipeline.process(now, frame)
                          │
        FrameBuffer.add ──┤  滚动回放缓冲（deque，长度 = clip_seconds * fps）
   CatDetector.detect ────┤  YOLO 出框 → geometry 算与水碗 ROI 的交集占比
 DrinkingDetector.update ─┤  停留 dwell_seconds 才触发，触发后进 cooldown
                          └─► ClipRecorder.save_clip（写整段缓冲）+ StatsStore.record_event
```

事件发生后，`app.py` 主循环额外起后台线程发邮件（`Emailer.notify_drinking`，内部限流）。

关键点：
- **「猫在水碗」是两路 OR**：`vision.py` 的 YOLO 认猫（框压到水碗 ROI）**或** `simple.py` 的 `MotionGrayDetector`（水碗 ROI 内「画面变化 + 灰蓝色块」启发式，无需训练）。后者是为「模型没训好前先多录候选、之后人工标注」设计的，宁可多录。`pipeline.py` 先看 YOLO，没命中再问简单模型。
- **触发逻辑全在 `detector.py`（`DrinkingDetector`）**，是个纯时间状态机（`_dwell_start` / `_cooldown_until`），不碰图像 —— 改触发行为先看这里。`pipeline.py` 只负责把"猫是否在碗里"这个布尔喂进去。
- **采集与检测解耦**（为了预览流畅）：`app.py` 起一个**采集线程**全速读相机 → 更新 `LatestFrame`（预览用）+ 按 `fps` 节奏喂 `FrameBuffer`；**主线程是检测循环**，按 `detect_interval_seconds` 取最新帧跑 YOLO，把「猫是否在碗」写进 `Presence`。慢的推理不再拖累出图/录制。`FrameBuffer` 因此加了锁。网页实时画面走 **MJPEG**（`/stream.mjpg`，单连接连续推帧），不是刷快照。
- **整段会话录制（默认）**：`session.py` 的 `DrinkSession` 是状态机——猫在碗持续 `dwell_seconds` → 开录（先把 `FrameBuffer` 里「凑近过程 + dwell」写进去做 pre-roll）→ 猫还在就一直写 → 离开持续 `session_end_grace_seconds`（或到 `max_session_seconds` 封顶）→ 收尾存盘。**写帧只在采集线程**（按 fps 节奏，writer fps 一致播放速度才对），检测线程只更新 `Presence`。`record_session=False` 退回旧的「固定 `clip_seconds` 缓冲 dump」（`Pipeline.detect`）。会话录制时缓冲开到 `preroll+dwell` 秒。
- **编码必须 H.264(`avc1`)**——`recorder.open_writer` 先试 avc1 再退回 mp4v；mp4v 浏览器 `<video>` 播不了（黑屏 0:00），网页要播就得 avc1。`prune_dir` 只留最近 `max_clips` 段。
- **几何判定与 YOLO 解耦**：`geometry.py` 用 0–1 比例的 `bowl_roi`（与分辨率无关），`ratio_rect_to_pixels` 按当前帧尺寸换算；`cat_overlaps_bowl` 用 `交集 / 水碗面积 >= min_overlap_ratio` 判猫是否在碗里。
- **两个 SQLite 表，同一个 `data/catcam.db`**：`StatsStore`→`events`（喝水事件），`FeedbackStore`→`labels`（人工标注）。每次连接都新开 `sqlite3.connect`，无连接池。
- **标注即产训练数据**：`FeedbackStore.label_clip` 写 `labels` 表的同时，把该段 mp4 抽帧到 `data/training/{drinking,not_drinking}/`。
- **配置即 schema**：`config.py` 的 `Config` dataclass 是唯一真相，`load_config` 首次运行生成 `config.json` 并写默认值；读取时按 dataclass 字段过滤未知键，`bowl_roi` 强转 tuple。加配置项就改这个 dataclass。SMTP 凭据（`smtp_password` 是邮箱授权码）写在 `config.json`，已 gitignore。
- **邮件提醒**：`mailer.py` 的 `Emailer` 走 QQ SMTP SSL（465）。`should_send` 做最小间隔限流（`mail_min_interval_seconds`，防误判刷屏）；`build_drinking_email`（纯函数、可单测）组装 HTML + 三张内嵌图（触发照片 + 周/月趋势）。趋势图在 `charts.py` 用 matplotlib Agg 渲染 PNG（标题用英文避开中文字体乱码）。邮件里的平台链接靠 `netutil.lan_ip()` 取真实内网 IP（会避开 VPN/TUN 的隧道地址，优先 RFC1918）。
- **一键训练 / 模型版本 / 自我迭代**：`trainer.py` 的 `TrainingManager`（后台线程 + 状态）把 `data/training/{drinking,not_drinking}` 整理成 train/val，训 `yolov8n-cls`，回报验证集准确率。**训练 project 必须传绝对路径**（`Path(work_dir).resolve()`），否则 ultralytics 会把产物塞进 `runs/classify/<相对project>/`，按 `work_dir/cls/weights/best.pt` 找不到 `best.pt`。
- **模型版本登记**：`models.py` 的 `ModelRegistry`（`data/models/registry.json`）每次训练登记一个版本 `vN`（准确率 + 抽帧数 + 标注快照 + 时间），`active` 指向当前生效版本。训练**不自动生效**，网页手动「设为生效」。
- **兜底 + 测试模型（关键设计）**：简单模型/YOLO 是**兜底**，永远负责多录、保证召回；分类器是**测试模型**，两种模式（`classifier.py`）：
  - `shadow`（测试，**默认**）：`ActiveModel.gate()` 恒放行 → **绝不拦截录制**，兜底全录；只用 `predict()` 给每段打个预测，记到 `events.predicted/predicted_by`，和人工标注一比就是「实战命中率」（`stats.model_hitrate`）。
  - `gate`（过滤，opt-in）：`gate()` 用模型判「真喝水」才放行，过滤误触——**只在模型够准时用**。
  - **教训**：早先把分类器当 `AND` 门（`cat_in_roi AND confirm()`），一个偏科模型（5👍/80👎 训出来、几乎全判「没喝」）直接把录制全掐死。模型还在迭代时**绝不能让它否决兜底**。`Pipeline.cat_in_bowl` 现在调 `active_model.gate(frame)`；shadow 恒 True。出错 fail-open。
  - 模式持久化在 registry（`active_mode`），网页 `/api/model/activate {id,mode}` 热切换；视频列表显示模型对每段的判断（`/api/clips` 的 `predictions`）。
- **标注状态 / 避免重复训练**：`labels` 表加了 `trained_version` 列（NULL=已标注未训练）。`FeedbackStore.label_states()` 给出 待标注（靠 web 比对当前 clips）/ 已标注未训练 / 已标注已训练 + 👍👎 计数；训练完 `mark_trained(version)` 把当前标注全标为已训练。`/api/train` 在「无新标注」时拒绝，避免重复无效训练。改标注会把 `trained_version` 清回 NULL（数据变了 = 又得练）。
- **AI 整段裁判 / 自动标注（默认开，需填 `ai_api_key`）**：`ai_labeler.py` 的 `AILabeler` 在会话录完后台调
  外部视觉大模型（OpenRouter / OpenAI 兼容，`ai_base_url`/`ai_api_key`/`ai_model` 可配）判「喝/没喝」，写
  `labels`（`source='ai'` + 置信度 + 理由）并抽帧进 `data/training`。
  - **`judge.py` 的 `VLMClipJudge` 把它包成「整段裁判」，是「发邮件 + 记喝水次数」的唯一权威**（`app.py` 的
    `_judge_async` → `judge_and_notify`）：判「真喝水」才发邮件、才计入次数；判「没喝」/ 失败则该段只留作素材，
    **不发、不计**。`record_event`（事件历史 + 影子模型预测）照旧无条件记，计数靠下面的口径过滤。
  - **计数口径**：`stats.py` 的 `count_between`/`events_between` 现在要求 `labels.is_drinking = 1` 才计——
    未标注 / 标「没喝」/ 无 clip 的事件都不计入今日次数与趋势（即「只数 AI/人工确认喝水的」）。
  - **默认开但受 key 把关**：`ai_label_enabled` 默认 True，`AILabeler.from_config` 仍要求 `ai_api_key` 非空才
    动作——没填 key = 无裁判，不发邮件、次数恒 0（`app.py` 启动会告警）。**⚠️ 填了 key 开始判定后，画面帧会上传
    外部服务器**，与「画面只在本机/局域网」原则冲突；规划中第二阶段用本地小视频模型接管后可关停 VLM 回归全本地。
  - fail-open（裁判失败只记日志、该段保持未标注，不影响录制）。人工 > AI：已人工标注的段不被覆盖。训练侧
    `prepare_dataset(balance=True)` 对 train 划分做类别平衡（多数类降采样），val 保持真实分布。
  - 设计/路线：`docs/superpowers/specs/2026-06-26-video-action-judge-design.md`（含 2026-06-27 环境校正附录）。
- **本地视频模型（第二阶段，离线已就绪、未接裁判）**：`videojudge.py`（`s3d` 冻结特征 + `DrinkingHead`
  torch 小头 + `LocalVideoClipJudge`）+ `video_trainer.py` + `python -m catcam.video_train`。用 VLM/人工攒的
  `labels` 离线训一个本地视频「真喝水/没喝」小头（**看动作、非单帧**——这才是单帧 `yolov8n-cls` 弱的根因），
  登记成 `base=s3d+head` 的版本（不自动生效）。主干用 torchvision 自带 `s3d`（8.3M、权重 ~30MB，免
  transformers）；小头一层 logistic（免 sklearn）；特征按 clip 缓存进 `data/training/features/*.npy`。
  - **类别不平衡下别看 top1**：训练报告同时给「喝水召回/精确」+「全猜没喝」基线——喝水样本少时 top1 会被多数类
    带高（实测 10 喝/90 没喝时 top1 84% 却低于 96% 基线、喝水召回 0%）。**要等 VLM 攒够足量「喝水」正样本**。
  - **本轮不改采集/裁判热路径**——`LocalVideoClipJudge` 与 `VLMClipJudge` 同接口，评估满意后再接进 `app.py`
    的裁判位 + 复用 registry 的 shadow→gate（后续一步）。

## Conventions / gotchas

- **`data/` 与 `config.json` 都被 gitignore**（连同 `*.pt`、`.venv/`）。运行产物（视频、db、下载的权重、用户本地配置）不进库。
- `catcam.demo` 与 `catcam.app` 共用同一个 `create_app`：web 层只依赖注入进来的 store / recorder / `frame_provider` 回调，不知道帧从哪来 —— 加网页功能改 `web.py`，两条入口都受益。
- **网页是单文件标签页应用**（`web.py` 的 `INDEX_HTML`，无框架、纯 vanilla JS）：四个标签 总览/趋势/视频/训练，切到哪个才加载哪个的数据。**视频列表懒加载**：每段先显示封面缩略图（`/clips/{name}/thumb.jpg`，只解码首帧）+ 播放按钮，点了才把该段换成 `<video>`，避免一进来所有视频一起 `preload` 转圈。内联 `onclick` 里嵌 `JSON.stringify(name)` 时属性要用**单引号**（值含双引号），否则 handler 被截断。`今日喝水次数 / 时间点` 已**按裁判口径过滤**——只数 `labels.is_drinking = 1` 的事件（未标注/没喝都不计，见 `stats.py` 的 `JOIN labels`）。
- 测试覆盖每个模块且不依赖真实摄像头或 YOLO 权重：`vision.py` 把 YOLO 的 box 解析逻辑拆成纯函数 `filter_cat_boxes` 单测，`CatDetector` 用 `_FakeBox` 注入。新增逻辑沿用"纯函数 + 可注入依赖"以便单测。
- web 文件名参数（`/clips/{name}`、feedback 的 `clip`）都手动挡 `/ \ ..` 防目录穿越，新增涉及路径的接口照做。
- `web_host` 默认 `127.0.0.1`（画面不出本机）；改成 `0.0.0.0` 会暴露给整个局域网 —— 别擅自改默认。

## 夜间/弱光

摄像头无红外。`nightvision.py`：`is_dark` 按整体亮度判暗 → `enhance_lowlight`（LAB 的 L 通道 CLAHE）增强后用于显示/录制/识别；夜间识别走「画面变化」（`MotionGrayDetector.present(..., night=True)` 跳过颜色门、只看运动）。`record_at_night=False` 则天黑直接不记录。极暗环境（房间全黑）增强也只能拉出轮廓，检测靠运动——可靠性有限，碗边放个小夜灯会显著改善。

## Scope (MVP 已知边界)

会把"好奇凑近没喝"误计 —— 这是**有意为之**：先用简单模型多录候选，靠网页标注 + 一键训练分类器逐步修正。隐私时段自动关闭、网页画框选区域、开机自启均为后续阶段。设计与计划见 `docs/superpowers/`。
