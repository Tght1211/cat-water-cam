# 录制带声音 + 有声/无声下载 设计

日期：2026-07-04

给会话录制加音轨；下载支持「有声（原始）」与「无声（去音轨）」。

## 前提与约束（实测）

- 相机自带麦克风：avfoundation 音频设备 `[1] UGREEN Camera`；ffmpeg 8.1 已装。
- **⚠️ macOS 麦克风权限**：实测 `ffmpeg -f avfoundation -i ":1"` 在无权限时**卡在设备打开**（12s 零输出）。
  跑 catcam 的终端/Python 必须在「系统设置 → 隐私与安全性 → 麦克风」里获授权，否则采音会 hang。
  ⇒ **采音必须超时保护 + fail-open**：起不来就当无音频，照常录**无声**视频，绝不因麦克风拖垮录制。
- 隐私：音频与画面一样**只在本机/局域网**，不外传（AI 裁判只送画面帧，不送音频）。
- OpenCV 的 `VideoWriter` **不支持音轨**——视频链路完全不动，音频靠**旁路环形缓冲 + 结束后 mux**。

## 决策（已和用户确认）

- **默认关**：`record_audio` 默认 `False`，config.json 手动开。
- **下载 UX**：单个「下载」链接 + 视频页工具栏一个全局「☑ 下载含声音」勾选框切换有声/无声。

## 架构

视频链路（OpenCV 出帧→检测/预览/`DrinkSession` 写帧）**原样不变**。新增一条音频旁路：

```
持续采音(ffmpeg avfoundation → s16le PCM 管道)
     │ 读线程按 wall-clock 锚定时间轴，写入 AudioRing（有界 bytearray）
     ▼
会话 _finish：拿到 clip 的 [audio_start, audio_end]（含 preroll 回溯）
     ▼
app 后台线程：AudioRing.slice(start,end) → mux_audio_into(clip.mp4, pcm)
     （ffmpeg -c:v copy -c:a aac -shortest → 临时文件 → os.replace 原子换入）
```

### `catcam/audio.py`（新）

- `AudioRing`：
  - `start()`：起持续 ffmpeg：`ffmpeg -nostdin -f {fmt} -i {device} -ac 1 -ar {sr} -f s16le -`，
    读线程把 stdout 定长块喂 `_ingest(now, chunk)`。**看门狗**：启动后 N 秒（默认 5）没收到任何 PCM →
    判定不可用（`available=False`）、杀 ffmpeg、记日志。整个在线程里，绝不阻塞调用方。
  - `_ingest(now, chunk)`（**可单测**，不碰 ffmpeg）：首块时锚定 `_t0 = now - len(chunk)/bytes_per_sec`；
    追加进 `_buf`；超过 `max_seconds` 从头部丢弃并推进 `_base_sample`。
  - `slice(start_ts, end_ts) -> bytes|None`（**纯逻辑、可单测**）：按 `_t0`+采样率把时间换成样本区间，
    夹到现有范围，返回 PCM；越界/空 → None。
  - `available`、`stop()`。
- 纯函数（可单测，只拼参数不执行）：
  - `capture_cmd(fmt, device, sr) -> list[str]`
  - `mux_cmd(video, pcm_raw, out, sr) -> list[str]`（`-i video -f s16le -ar sr -ac 1 -i pcm -c:v copy -c:a aac -shortest`）
  - `muted_cmd(video) -> list[str]`（`-i video -an -c:v copy -movflags frag_keyframe+empty_moov -f mp4 pipe:1`）
- `mux_audio_into(video_path, pcm, sr) -> bool`：写临时 raw → 跑 `mux_cmd` → 成功则原子替换、返回 True；
  任何失败（无 ffmpeg/pcm 空/返回码非 0）→ 保留原无声视频、返回 False（**fail-open**）。

内存：16kHz·mono·16bit = 32KB/s；环 ≈ max_session+preroll+余量（~90s）→ <3MB。

### config.py

```
record_audio: bool = False
audio_input_format: str = "avfoundation"   # 平台相关（mac）
audio_device: str = ":1"                     # ffmpeg -i 值；":1" = 仅音频设备 1（本机相机麦）
audio_sample_rate: int = 16000
```

### session.py

`_begin` 记 `self._audio_start = now - len(frame_buffer.all_frames())/fps`（回溯 preroll）；
`_finish` 里 `SessionResult` 新增 `audio_start`、`audio_end=now`。旧字段不变。

### app.py

- 若 `cfg.record_audio`：构造并 `start()` 一个 `AudioRing`（失败仅告警）。
- 会话出结果后，若 ring 可用：起**后台线程** `mux_audio_into(clip, ring.slice(audio_start,audio_end), sr)`，
  与现有 `_judge_async` 并行。mux 用临时文件 + 原子替换，读者（缩略图/裁判/下载）永远读到完整文件；
  音轨不影响裁判抽帧（`-c:v copy`，视频帧不变）。非会话定长模式仍无声（本次不覆盖，边界已记）。

### web.py（下载 UX）

- `/clips/{name}` 加查询 `audio: int = 1`。`audio=0` → 用 `muted_cmd` 起 ffmpeg 把去音轨 mp4 流式吐出
  （`StreamingResponse`，`Content-Disposition: attachment`）；ffmpeg 不可用则回退原文件。有声=原文件。
- 前端：视频页工具栏加全局 `☑ 下载含声音`（默认勾选）；每段的「下载」链接 `onclick` 时按勾选态设 `href`
  （勾=原始，不勾=`?audio=0`）。**播放**已是 `controls` 未静音，录了音就自动有声，无需改。

## 测试（不依赖真麦克风/摄像头，遵循仓库「纯函数+可注入」约定）

- `audio.py`：`_ingest`/`slice` 用合成 PCM 断言时间轴与切片字节区间（含越界夹取、环丢弃）；
  `capture_cmd/mux_cmd/muted_cmd` 断言参数；`mux_audio_into` 在无 ffmpeg / 空 pcm 时返回 False 不抛。
- `session.py`：断言 `SessionResult` 带 `audio_start/audio_end` 且 preroll 回溯正确。
- `web.py`：`/clips/{name}?audio=0` 路由存在；ffmpeg 不可用时回退原文件（monkeypatch/跳过真跑）。

`pytest -q` 全绿。麦克风权限相关只能在用户授权后手测（本设计保证未授权也只是无声、不崩）。
