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
