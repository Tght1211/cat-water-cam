from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from catcam.charts import trend_png
from catcam.classifier import DrinkingClassifier
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.stats import StatsStore, day_bounds


def clip_duration(path: Path) -> float:
    """读视频头拿时长（秒），不解码全片。供网页展示「总共几秒」。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return round(frames / fps, 1) if fps > 0 else 0.0


INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>猫咪饮水监控</title>
<style>
:root{
  --bg:#f5f5f7; --surface:#ffffff; --surface2:#fbfbfd; --nav:rgba(245,245,247,.72);
  --ink:#1d1d1f; --muted:#86868b; --line:rgba(0,0,0,.08);
  --accent:#0071e3; --accent2:#0a84ff; --green:#34c759; --red:#ff3b30; --amber:#ff9f0a;
  --shadow:0 8px 30px rgba(0,0,0,.06); --shadow-h:0 16px 44px rgba(0,0,0,.12);
  --radius:20px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#000; --surface:#1c1c1e; --surface2:#161618; --nav:rgba(28,28,30,.7);
  --ink:#f5f5f7; --muted:#98989d; --line:rgba(255,255,255,.12);
  --accent:#0a84ff; --accent2:#409cff; --green:#30d158; --red:#ff453a; --amber:#ffd60a;
  --shadow:0 8px 30px rgba(0,0,0,.5); --shadow-h:0 18px 48px rgba(0,0,0,.65);
}}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","PingFang SC","Microsoft YaHei",sans-serif;
  letter-spacing:-.01em}
.wrap{max-width:1280px;margin:0 auto;padding:0 24px}
a{color:inherit}

/* 顶部导航 + 标签页 */
header{position:sticky;top:0;z-index:20;background:var(--nav);
  backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid var(--line)}
.nav{display:flex;align-items:center;gap:16px;height:60px}
.brand{display:flex;align-items:center;gap:9px;font-size:17px;font-weight:600;white-space:nowrap}
.brand svg{width:22px;height:22px;color:var(--accent)}
.tabs{display:flex;gap:3px;background:var(--bg);border:1px solid var(--line);
  border-radius:980px;padding:4px;margin:0 auto}
.tabs button{border:0;background:transparent;color:var(--muted);font-family:inherit;
  font-size:14px;font-weight:600;padding:8px 18px;border-radius:980px;cursor:pointer;transition:.2s}
.tabs button:hover{color:var(--ink)}
.tabs button.on{background:var(--surface);color:var(--ink);box-shadow:0 1px 4px rgba(0,0,0,.14)}
.pill{display:inline-flex;align-items:center;gap:7px;background:var(--accent);color:#fff;
  border-radius:980px;padding:7px 15px;font-size:13px;font-weight:600;white-space:nowrap}
.pill b{font-variant-numeric:tabular-nums}
@media (max-width:720px){
  .nav{flex-wrap:wrap;height:auto;padding:10px 0;gap:10px}
  .tabs{order:3;width:100%;margin:0;justify-content:space-between}
  .tabs button{flex:1;padding:8px 0}
}

main{padding:30px 0 90px}
.tab{display:none}
.tab.on{display:block;animation:fade .35s cubic-bezier(.22,1,.36,1)}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:0 2px 22px}
.head h2{font-size:27px;font-weight:700;letter-spacing:-.02em;margin:0}
.head p{color:var(--muted);font-size:14px;margin:5px 0 0}
.head .right{color:var(--muted);font-size:13px}

.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden}
.card-h{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:16px 20px 0;
  font-size:12px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
.card-b{padding:16px 20px 20px}

/* KPI 指标卡 */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:18px;
  padding:18px 20px;box-shadow:var(--shadow)}
.kpi .k-top{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:13px;margin-bottom:12px}
.kpi .k-ico{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;
  background:rgba(0,113,227,.12);color:var(--accent)}
.kpi .k-ico svg{width:17px;height:17px}
.kpi .k-val{font-size:36px;font-weight:700;letter-spacing:-.02em;line-height:1;font-variant-numeric:tabular-nums}
.kpi .k-val small{font-size:15px;color:var(--muted);font-weight:600;margin-left:4px;letter-spacing:0}

/* 总览：实时画面 + 今日 */
.ov{display:grid;grid-template-columns:1.85fr 1fr;gap:20px}
@media (max-width:900px){.ov{grid-template-columns:1fr}}
.live-frame{position:relative;border-radius:14px;overflow:hidden;background:#000;aspect-ratio:4/3}
#live{display:block;width:100%;height:100%;object-fit:cover}
.live-tag{position:absolute;top:12px;left:12px;display:flex;align-items:center;gap:6px;
  background:rgba(0,0,0,.55);color:#fff;border-radius:980px;padding:6px 12px;font-size:11px;font-weight:600;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,199,89,.5)}70%{box-shadow:0 0 0 7px rgba(52,199,89,0)}100%{box-shadow:0 0 0 0 rgba(52,199,89,0)}}
.today .card-b{display:flex;flex-direction:column;justify-content:center;height:100%;gap:4px}
.big-num{font-size:76px;font-weight:700;line-height:1;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.big-num small{font-size:22px;font-weight:600;color:var(--muted);margin-left:6px}
.lbl{color:var(--muted);font-size:13px;margin:14px 0 0}
.times{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
.chip{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:5px 11px;
  font-size:12px;font-variant-numeric:tabular-nums}
.times-empty{color:var(--muted);font-size:13px;margin-top:12px}
/* 今日时间线（24 小时横轴） */
.timeline{position:relative;height:76px;margin:4px 7px 0}
.tl-track{position:absolute;left:0;right:0;top:30px;height:5px;background:var(--bg);border:1px solid var(--line);border-radius:4px}
.tl-tick{position:absolute;top:24px;transform:translateX(-50%)}
.tl-tick i{display:block;width:1px;height:11px;background:var(--line);margin:0 auto}
.tl-tick span{position:absolute;top:14px;left:50%;transform:translateX(-50%);font-size:10px;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
.tl-now{position:absolute;top:22px;height:21px;width:2px;background:var(--red);border-radius:2px;transform:translateX(-50%)}
.tl-now::after{content:"";position:absolute;top:-3px;left:50%;width:6px;height:6px;border-radius:50%;background:var(--red);transform:translateX(-50%)}
.tl-dot{position:absolute;top:24px;width:15px;height:15px;border-radius:50%;background:var(--accent);border:3px solid var(--surface);transform:translateX(-50%);box-shadow:0 1px 5px rgba(0,0,0,.28);cursor:pointer;transition:transform .15s;z-index:2}
.tl-dot:hover{transform:translateX(-50%) scale(1.3)}
.tl-empty{color:var(--muted);font-size:13px;display:flex;align-items:center;height:100%}
.today-foot{border-top:1px solid var(--line);padding-top:16px;margin-top:18px;color:var(--muted);font-size:13.5px}
.today-foot b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;font-size:15px}

/* 分段控件 */
.seg-ctl{display:inline-flex;background:var(--bg);border:1px solid var(--line);border-radius:980px;padding:3px}
.seg-ctl button{border:0;background:transparent;color:var(--muted);font-family:inherit;font-size:13px;
  font-weight:600;padding:6px 15px;border-radius:980px;cursor:pointer;transition:.2s}
.seg-ctl button.on{background:var(--surface);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.14)}

/* 趋势图 */
.trend-sum{display:flex;flex-wrap:wrap;gap:8px 26px;margin:2px 2px 18px;color:var(--muted);font-size:13px}
.trend-sum b{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:600;font-size:15px}
.chartwrap{width:100%}
.chartwrap svg{width:100%;height:auto;display:block;overflow:visible;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC",sans-serif}
.chartwrap:hover .bar{opacity:.4}
.chartwrap .bar{opacity:1;transform-box:fill-box;transform-origin:bottom;
  animation:grow .55s cubic-bezier(.22,1,.36,1) both;transition:opacity .15s}
.chartwrap .bar:hover{opacity:1}
@keyframes grow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.cval{fill:var(--ink);font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.cxlab{fill:var(--muted);font-size:11px}
.cbase{stroke:var(--line);stroke-width:1}

/* 视频：筛选 + 懒加载缩略图 */
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.fchip{border:1px solid var(--line);background:var(--surface);color:var(--muted);border-radius:980px;
  padding:8px 16px;font-size:13px;font-weight:600;cursor:pointer;transition:.18s;font-family:inherit}
.fchip:hover{color:var(--ink)}
.fchip.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(248px,1fr));gap:18px}
.clip{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden;display:flex;flex-direction:column;
  transition:transform .25s cubic-bezier(.22,1,.36,1),box-shadow .25s}
.clip:hover{transform:translateY(-3px);box-shadow:var(--shadow-h)}
.thumb{position:relative;aspect-ratio:4/3;background:#000;cursor:pointer;overflow:hidden}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s,opacity .3s}
.thumb:hover img{transform:scale(1.05);opacity:.82}
.thumb .play{position:absolute;inset:0;margin:auto;width:56px;height:56px;border:0;border-radius:50%;
  background:rgba(255,255,255,.92);color:#111;display:grid;place-items:center;cursor:pointer;
  box-shadow:0 6px 22px rgba(0,0,0,.34);transition:transform .2s}
.thumb .play svg{width:22px;height:22px;margin-left:3px}
.thumb:hover .play{transform:scale(1.1)}
.thumb .dur-badge{position:absolute;right:9px;bottom:9px;background:rgba(0,0,0,.62);color:#fff;
  border-radius:7px;padding:3px 8px;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px)}
.src-badge{display:inline-block;margin-left:6px;font-size:11px;padding:1px 7px;border-radius:8px;vertical-align:middle}
.src-badge.ai{background:rgba(10,132,255,.12);color:var(--accent)}
.src-badge.human{background:var(--line);color:var(--muted)}
.thumb .time-badge{position:absolute;left:9px;bottom:9px;background:rgba(0,0,0,.62);color:#fff;
  border-radius:7px;padding:3px 8px;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px)}
.thumb video{width:100%;height:100%;display:block;background:#000;object-fit:cover}
/* 视频时间线分组 + 滚动分页 */
.day-head{grid-column:1/-1;display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;color:var(--ink);margin:16px 2px 0}
.day-head:first-child{margin-top:0}
.day-dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px rgba(0,113,227,.15)}
.day-n{font-size:12px;font-weight:600;color:var(--muted);background:var(--bg);border:1px solid var(--line);padding:2px 9px;border-radius:980px}
.clip-more{grid-column:1/-1;text-align:center;color:var(--muted);font-size:13px;font-weight:600;padding:18px;border:1px dashed var(--line);border-radius:14px;cursor:pointer;margin-top:8px;transition:.15s}
.clip-more:hover{color:var(--accent);border-color:var(--accent)}
.meta{padding:13px 15px 15px;display:flex;flex-direction:column;gap:11px}
.meta .top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.fname{font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.status{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
  padding:4px 9px;border-radius:980px;white-space:nowrap}
.status i{width:6px;height:6px;border-radius:50%}
.s-none{background:var(--bg);color:var(--muted)} .s-none i{background:var(--muted)}
.s-yes{background:rgba(52,199,89,.14);color:var(--green)} .s-yes i{background:var(--green)}
.s-no{background:rgba(255,59,48,.14);color:var(--red)} .s-no i{background:var(--red)}
.seg{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.seg button{display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);
  border-radius:11px;padding:9px 0;font-size:13px;font-weight:500;cursor:pointer;
  transition:transform .12s,background .15s,border-color .15s,color .15s;font-family:inherit}
.seg button svg{width:15px;height:15px}
.seg button:hover{background:var(--bg)}
.seg button:active{transform:scale(.96)}
.seg .yes.on{background:var(--green);border-color:var(--green);color:#fff}
.seg .no.on{background:var(--red);border-color:var(--red);color:#fff}
.dl{display:inline-flex;align-items:center;gap:5px;color:var(--accent2);
  font-size:12px;text-decoration:none;font-weight:500}
.dl svg{width:13px;height:13px}
.empty{grid-column:1/-1;color:var(--muted);background:var(--surface);
  border:1px solid var(--line);border-radius:var(--radius);padding:56px 24px;text-align:center}
.empty svg{width:34px;height:34px;color:var(--muted);margin-bottom:12px}
.empty code{background:var(--bg);border:1px solid var(--line);padding:2px 7px;border-radius:6px;font-size:12px}

/* 训练 */
.train-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:20px}
@media (max-width:860px){.train-grid{grid-template-columns:1fr}}
.train-grid p{margin:0 0 16px;color:var(--muted);font-size:14px;line-height:1.7}
.btn{display:inline-flex;align-items:center;gap:8px;border:0;background:var(--accent);color:#fff;
  border-radius:12px;padding:12px 22px;font-size:15px;font-weight:600;cursor:pointer;font-family:inherit;
  transition:transform .12s,opacity .2s}
.btn:hover{opacity:.92} .btn:active{transform:scale(.97)}
.btn:disabled{opacity:.5;cursor:default}
.t-status{margin-top:14px;color:var(--muted);font-size:13.5px;min-height:18px}
.t-prog{margin-top:14px}
.t-prog .t-line{display:flex;justify-content:space-between;align-items:baseline;
  font-size:13px;color:var(--muted);margin-bottom:7px}
.t-prog .t-line b{color:var(--ink);font-size:14px;font-variant-numeric:tabular-nums}
.t-bar{height:8px;border-radius:6px;background:var(--line);overflow:hidden}
.t-bar>i{display:block;height:100%;border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width .5s ease}
.t-bar.indet>i{width:35%!important;animation:tIndet 1.2s ease-in-out infinite}
@keyframes tIndet{0%{margin-left:-35%}100%{margin-left:100%}}
.lab-stat{display:flex;gap:14px;margin-bottom:14px}
.lab-stat .box{flex:1;background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.lab-stat .n{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
.lab-stat .t{color:var(--muted);font-size:12px;margin-top:6px}
.bar-track{height:9px;background:var(--bg);border:1px solid var(--line);border-radius:7px;overflow:hidden}
.bar-fill{height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--green));
  border-radius:7px;transition:width .6s cubic-bezier(.22,1,.36,1)}
.bar-cap{color:var(--muted);font-size:12.5px;margin-top:9px}
.amini{color:var(--muted);font-size:12.5px;line-height:1.65;margin:14px 0 0}
.amini b{color:var(--ink)}
.mpredline{margin-top:2px}
.mpred{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:3px 9px;border-radius:980px}
.mpred.y{background:rgba(52,199,89,.14);color:var(--green)}
.mpred.n{background:rgba(255,159,10,.16);color:var(--amber)}
.active-box{display:flex;align-items:center;gap:13px;flex-wrap:wrap}
.active-tag{display:inline-flex;align-items:center;font-size:13px;font-weight:700;padding:7px 14px;border-radius:980px}
.active-tag.on{background:rgba(52,199,89,.16);color:var(--green)}
.active-tag.off{background:var(--bg);color:var(--muted);border:1px solid var(--line)}
.mlist{display:flex;flex-direction:column;gap:12px}
.mrow{display:flex;align-items:center;gap:16px;background:var(--surface);border:1px solid var(--line);
  border-radius:16px;padding:15px 20px;box-shadow:var(--shadow);transition:border-color .2s,box-shadow .2s}
.mrow.on{border-color:var(--accent);box-shadow:0 0 0 2px rgba(0,113,227,.22),var(--shadow)}
.mrow .mv{font-size:16px;font-weight:700;display:flex;align-items:center;gap:9px}
.mrow .macc{font-size:12px;font-weight:600;color:var(--accent);background:rgba(0,113,227,.1);padding:2px 9px;border-radius:980px}
.mrow .mmeta{color:var(--muted);font-size:12.5px;margin-top:5px;font-variant-numeric:tabular-nums}
.mrow .grow{flex:1}
.mbtn{border:1px solid var(--line);background:var(--surface);color:var(--accent);border-radius:10px;
  padding:9px 18px;font-weight:600;font-size:13px;cursor:pointer;font-family:inherit;transition:.15s;white-space:nowrap}
.mbtn:hover:not(:disabled){background:var(--bg)}
.mbtn.cur{background:var(--green);border-color:var(--green);color:#fff;cursor:default}
.mbtn.off{color:var(--muted)}
.mbtn:disabled{opacity:.5}
</style></head>
<body>
<header><div class="wrap nav">
<div class="brand">
<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg>
猫咪饮水监控</div>
<nav class="tabs">
<button data-tab="home" class="on">总览</button>
<button data-tab="trend">趋势</button>
<button data-tab="clips">视频</button>
<button data-tab="train">训练</button>
</nav>
<span class="pill">今日 <b id="count">–</b> 次</span>
</div></header>

<main class="wrap">

<!-- 总览 -->
<section id="tab-home" class="tab on">
<div class="head"><div><h2>总览</h2><p>实时画面与今日饮水概览</p></div></div>
<div class="kpis">
<div class="kpi"><div class="k-top"><span class="k-ico" id="ic1"></span>今日喝水</div><div class="k-val"><span id="kToday">–</span><small>次</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ic2"></span>近 7 天</div><div class="k-val"><span id="kWeek">–</span><small>次</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ic3"></span>日均</div><div class="k-val"><span id="kAvg">–</span><small>次</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ic4"></span>最近一次</div><div class="k-val" style="font-size:30px" id="kLast">–</div></div>
</div>
<div class="ov">
<div class="card"><div class="card-h">实时画面</div><div class="card-b">
<div class="live-frame"><img id="live" src="/stream.mjpg" alt="实时画面" onerror="this.src='/snapshot.jpg?t='+Date.now()">
<span class="live-tag"><span class="dot"></span>在线</span></div></div></div>
<div class="card today"><div class="card-h">今日喝水</div><div class="card-b">
<div><div class="big-num"><span id="count2">–</span><small>次</small></div>
<div class="lbl">各次时间点</div><div class="times" id="times"></div></div>
<div class="today-foot" id="todayFoot"></div>
</div></div>
</div>
<div class="card" style="margin-top:20px"><div class="card-h">今日时间线 <span id="tlNow" style="font-weight:400;text-transform:none;letter-spacing:0"></span></div>
<div class="card-b"><div id="timeline" class="timeline"></div></div></div>
</section>

<!-- 趋势 -->
<section id="tab-trend" class="tab">
<div class="head"><div><h2>喝水趋势</h2><p>每日饮水次数变化</p></div>
<div class="seg-ctl" id="rangeCtl">
<button class="on" onclick="setRange(7,this)">近 7 天</button>
<button onclick="setRange(30,this)">近 30 天</button>
</div></div>
<div class="card"><div class="card-b">
<div class="trend-sum" id="trendSum"></div>
<div class="chartwrap" id="chart"></div>
</div></div>
</section>

<!-- 视频 -->
<section id="tab-clips" class="tab">
<div class="head"><div><h2>喝水视频</h2><p>点缩略图播放 · 点 👍/👎 标注攒训练数据</p></div>
<div class="right">最多保留 100 段</div></div>
<div class="toolbar" id="filters">
<button class="fchip on" onclick="setFilter('all',this)">全部</button>
<button class="fchip" onclick="setFilter('none',this)">未标注</button>
<button class="fchip" onclick="setFilter('yes',this)">真喝水</button>
<button class="fchip" onclick="setFilter('no',this)">没喝</button>
</div>
<div class="clips" id="clips"></div>
</section>

<!-- 训练 -->
<section id="tab-train" class="tab">
<div class="head"><div><h2>模型训练</h2><p>越标越准 · 自我迭代 · 训练产出新版本但不自动生效，需手动启用</p></div></div>
<div class="kpis">
<div class="kpi"><div class="k-top"><span class="k-ico" id="ict1"></span>待标注</div><div class="k-val"><span id="dsUn">–</span><small>段</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ict2"></span>已标注·未训练</div><div class="k-val"><span id="dsNew">–</span><small>段</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ict3"></span>已标注·已训练</div><div class="k-val"><span id="dsTr">–</span><small>段</small></div></div>
<div class="kpi"><div class="k-top"><span class="k-ico" id="ict4"></span>标注 👍/👎</div><div class="k-val" style="font-size:30px" id="dsBal">–</div><div class="mmeta" style="margin-top:6px">段（每类需 ≥4）</div></div>
</div>
<div class="train-grid">
<div class="card"><div class="card-b">
<p>对「视频」里每段点 👍真喝水 / 👎没喝攒样本；这里训一个<b>看动作</b>的视频模型
（s3d 冻结特征 + 小头）——喝水是「舔水」这个<b>动作</b>，单帧图片看不出，所以用整段视频。
首次要为每段抽特征，可能几分钟；看报告里的<b>喝水召回</b>（样本不平衡时 top1 会骗人）。
训完产出 vN 版本、<b>不自动生效</b>，去下面设为生效（切视频模型需重启采集进程）。每类需 ≥4 段。</p>
<button id="trainVideoBtn" class="btn" onclick="trainVideo()">训练视频模型</button>
<div class="t-status" id="trainVideoStatus"></div>
</div></div>
<div class="card"><div class="card-h">当前生效模型</div><div class="card-b">
<div id="activeBox"></div>
</div></div>
</div>
<div class="head" style="margin-top:34px"><div><h2 style="font-size:21px">模型版本</h2><p>每次训练产出一个版本，可随时切换生效</p></div></div>
<div id="modelList" class="mlist"></div>
</section>

</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const I={
  check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>',
  x:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  dl:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg>',
  play:'<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  cam:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L19 6h0a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><circle cx="12" cy="12.5" r="3.2"/></svg>',
  drop:'<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg>',
  cal:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4M16 3v4"/></svg>',
  avg:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18 9 9l4 5 3-4 4 5"/></svg>',
  clock:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'
};
$('#ic1').innerHTML=I.drop; $('#ic2').innerHTML=I.cal; $('#ic3').innerHTML=I.avg; $('#ic4').innerHTML=I.clock;
$('#ict1').innerHTML=I.cam; $('#ict2').innerHTML=I.drop; $('#ict3').innerHTML=I.check; $('#ict4').innerHTML=I.avg;
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

/* 标签页切换 —— 视频只在打开「视频」页时才加载，避免一进来全部转圈 */
function show(t){
  $$('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.tab===t));
  $$('.tab').forEach(s=>s.classList.toggle('on',s.id==='tab-'+t));
  history.replaceState(null,'','#'+t);
  if(t==='trend')renderTrend();
  if(t==='clips')loadClips();
  if(t==='train'){pollTrain();pollTrainVideo();}
}
$$('.tabs button').forEach(b=>b.onclick=()=>show(b.dataset.tab));

/* 概览统计（轮询） */
async function loadStats(){
  const s=await (await fetch('/api/stats/today')).json();
  $('#count').textContent=s.count; $('#count2').textContent=s.count; $('#kToday').textContent=s.count;
  $('#times').innerHTML=s.times.length?s.times.map(x=>`<span class="chip">${esc(x)}</span>`).join('')
    :'<div class="times-empty">今天还没记录到喝水</div>';
  $('#kLast').textContent=s.times.length?s.times[s.times.length-1]:'—';
  renderTimeline(s.times);
  try{
    const w=await (await fetch('/api/stats/range?days=7')).json();
    const wv=(w.days||[]).map(d=>d.count),wt=wv.reduce((a,b)=>a+b,0);
    $('#kWeek').textContent=wt; $('#kAvg').textContent=(wt/7).toFixed(1);
    $('#todayFoot').innerHTML=`近 7 天共 <b>${wt}</b> 次　·　日均 <b>${(wt/7).toFixed(1)}</b> 次`;
  }catch(e){}
}

/* 今日时间线：24 小时横轴，标出每次喝水 + 当前时刻 */
function hms2frac(t){const [h,m,s]=t.split(':').map(Number);return (h+(m||0)/60+(s||0)/3600)/24;}
function renderTimeline(times){
  const tl=$('#timeline'),now=new Date();
  const p=n=>String(n).padStart(2,'0');
  $('#tlNow').textContent=`现在 ${p(now.getHours())}:${p(now.getMinutes())}`;
  if(!times||!times.length){tl.innerHTML='<div class="tl-empty">今天还没记录到喝水</div>';return;}
  const nowFrac=(now.getHours()+now.getMinutes()/60+now.getSeconds()/3600)/24;
  const ticks=[0,3,6,9,12,15,18,21,24].map(h=>
    `<div class="tl-tick" style="left:${(h/24*100).toFixed(2)}%"><i></i><span>${h}:00</span></div>`).join('');
  const dots=times.map(t=>`<div class="tl-dot" style="left:${(hms2frac(t)*100).toFixed(2)}%" title="${esc(t)}"></div>`).join('');
  const nowLine=`<div class="tl-now" style="left:${(nowFrac*100).toFixed(2)}%"></div>`;
  tl.innerHTML=`<div class="tl-track"></div>${ticks}${nowLine}${dots}`;
}
/* 趋势图（原生 SVG，Apple Health 风） */
let trendDays=7;
function drawChart(box,pts){
  const W=760,H=240,L=12,R=12,T=26,B=30,ph=H-T-B,pw=W-L-R;
  const vals=pts.map(p=>p.count),maxV=Math.max(1,...vals),n=pts.length;
  const slot=pw/n,bw=Math.min(40,slot*0.62),every=Math.ceil(n/9);
  const bx=i=>L+slot*i+(slot-bw)/2,bh=v=>v/maxV*ph;
  let bars='',vtxt='',xlab='';
  pts.forEach((p,i)=>{const h=bh(p.count),x=bx(i),y=T+ph-h,last=i===n-1;
    bars+=`<rect class="bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(h,.001).toFixed(1)}" rx="${Math.min(bw/2,7).toFixed(1)}" fill="url(#${last?'gT':'gB'})" style="animation-delay:${i*20}ms"><title>${p.date}　${p.count} 次</title></rect>`;
    if(p.count>0)vtxt+=`<text class="cval" x="${(x+bw/2).toFixed(1)}" y="${(y-8).toFixed(1)}" text-anchor="middle">${p.count}</text>`;
    if((i%every===0&&n-1-i>=every)||last)xlab+=`<text class="cxlab" x="${(x+bw/2).toFixed(1)}" y="${H-10}" text-anchor="middle">${p.date}</text>`;});
  box.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="喝水趋势"><defs>
    <linearGradient id="gB" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--accent2)"/><stop offset="1" stop-color="var(--accent)" stop-opacity=".75"/></linearGradient>
    <linearGradient id="gT" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--green)"/><stop offset="1" stop-color="var(--accent2)" stop-opacity=".8"/></linearGradient></defs>
    <line class="cbase" x1="${L}" x2="${W-R}" y1="${T+ph}" y2="${T+ph}"/>${bars}${vtxt}${xlab}</svg>`;
}
async function renderTrend(){
  const r=await (await fetch('/api/stats/range?days='+trendDays)).json();
  const pts=r.days||[];
  drawChart($('#chart'),pts);
  const vals=pts.map(p=>p.count),total=vals.reduce((a,b)=>a+b,0);
  const avg=pts.length?(total/pts.length).toFixed(1):'0',mx=Math.max(0,...vals);
  $('#trendSum').innerHTML=`近 ${pts.length} 天共 <b>${total}</b> 次　·　日均 <b>${avg}</b> 次　·　单日最多 <b>${mx}</b> 次`;
}
function setRange(d,btn){trendDays=d;for(const b of $('#rangeCtl').children)b.classList.toggle('on',b===btn);renderTrend();}

/* 视频：按时间线（今天/昨天/日期）分组 + 下滑滚动分页；懒加载缩略图，点开才载入视频 */
let clipFilter='all', clipsData=null, clipPage=1, clipObserver=null;
const CLIP_PAGE=12;
async function loadClips(){
  clipPage=1;
  clipsData=await (await fetch('/api/clips')).json();
  renderClips();
}
function statusHtml(v){return v===true?`<span class="status s-yes"><i></i>真喝水</span>`
  :v===false?`<span class="status s-no"><i></i>没喝</span>`:`<span class="status s-none"><i></i>未标注</span>`;}
function clipTime(name){const m=String(name).match(/clip_(\\d+)/);return m?new Date(parseInt(m[1],10)):null;}
function dayKey(d){return d?`${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`:'?';}
function dayLabel(d){
  if(!d)return '未知时间';
  const now=new Date(),a=new Date(now.getFullYear(),now.getMonth(),now.getDate());
  const b=new Date(d.getFullYear(),d.getMonth(),d.getDate());
  const diff=Math.round((a-b)/86400000);
  if(diff===0)return '今天'; if(diff===1)return '昨天';
  return `${d.getMonth()+1} 月 ${d.getDate()} 日`;
}
function hhmmss(d){const p=n=>String(n).padStart(2,'0');return d?`${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`:'';}
function filteredClips(){
  const lab=clipsData.labels||{};
  return clipsData.clips.filter(n=>{const v=lab[n];
    return clipFilter==='all'||(clipFilter==='none'&&v==null)||(clipFilter==='yes'&&v===true)||(clipFilter==='no'&&v===false);});
}
function clipCard(n){
  const lab=clipsData.labels||{},dur=clipsData.durations||{},pr=clipsData.predictions||{},mt=clipsData.meta||{};
  const v=lab[n],d=dur[n],en=esc(n),jn=JSON.stringify(n),pred=pr[n],t=clipTime(n),m=mt[n];
  const srcBadge=!m?'':(m.source==='ai'
    ? `<span class="src-badge ai" title="${esc(m.reason||'')}">🤖 AI ${m.confidence!=null?m.confidence.toFixed(2):''}</span>`
    : `<span class="src-badge human">✋ 人工</span>`);
  const dtxt=d?`<span class="dur-badge">${d.toFixed(1)} 秒</span>`:'';
  const tbadge=t?`<span class="time-badge">${hhmmss(t)}</span>`:'';
  const predLine=(pred===undefined)?'':`<div class="mpredline"><span class="mpred ${pred?'y':'n'}">模型：${pred?'真喝水':'没喝'}</span></div>`;
  return `<div class="clip" data-name="${en}">
    <div class="thumb" onclick='playClip(this,${jn})'>
      <img loading="lazy" src="/clips/${encodeURIComponent(n)}/thumb.jpg" alt="">
      <button class="play" aria-label="播放">${I.play}</button>${tbadge}${dtxt}</div>
    <div class="meta">
      <div class="top"><span class="fname">${en}</span>${statusHtml(v)}${srcBadge}</div>${predLine}
      <div class="seg">
        <button class="yes ${v===true?'on':''}" onclick='fb(${jn},true)'>${I.check}喝了</button>
        <button class="no ${v===false?'on':''}" onclick='fb(${jn},false)'>${I.x}没喝</button>
      </div>
      <a class="dl" href="/clips/${encodeURIComponent(n)}" download>${I.dl}下载</a>
    </div></div>`;
}
function renderClips(){
  if(!clipsData)return;
  const box=$('#clips');
  if(!clipsData.clips.length){box.innerHTML=`<div class="empty">${I.cam}<div>还没有视频。接上摄像头跑 <code>python -m catcam</code>，猫在水碗停留就会自动录制。</div></div>`;return;}
  const items=filteredClips();
  if(!items.length){box.innerHTML='<div class="empty">这个筛选下没有视频</div>';return;}
  const dayCounts={}; items.forEach(n=>{const k=dayKey(clipTime(n));dayCounts[k]=(dayCounts[k]||0)+1;});
  const shown=items.slice(0,clipPage*CLIP_PAGE);
  let html='',lastDay=null;
  shown.forEach(n=>{const d=clipTime(n),k=dayKey(d);
    if(k!==lastDay){lastDay=k;html+=`<div class="day-head"><span class="day-dot"></span>${dayLabel(d)}<span class="day-n">${dayCounts[k]} 段</span></div>`;}
    html+=clipCard(n);});
  const remaining=items.length-shown.length;
  if(remaining>0)html+=`<div class="clip-more" id="clipSentinel" onclick="clipPage++;renderClips()">下滑加载更多 · 还有 ${remaining} 段</div>`;
  box.innerHTML=html;
  if(clipObserver)clipObserver.disconnect();
  const s=$('#clipSentinel');
  if(s&&'IntersectionObserver' in window){
    clipObserver=new IntersectionObserver(es=>{if(es[0].isIntersecting){clipPage++;renderClips();}},{rootMargin:'300px'});
    clipObserver.observe(s);
  }
}
function playClip(thumb,name){
  thumb.innerHTML=`<video src="/clips/${encodeURIComponent(name)}" controls autoplay playsinline></video>`;
}
function setFilter(f,btn){clipFilter=f;clipPage=1;$$('.fchip').forEach(b=>b.classList.toggle('on',b===btn));renderClips();}
async function fb(clip,is){
  await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip,is_drinking:is})});
  if(clipsData){clipsData.labels[clip]=is;
    if(clipsData.meta)clipsData.meta[clip]={is_drinking:is,source:'human',confidence:null,reason:null};
    // 局部更新该卡片：不重建整个网格，避免打断其它正在播放的视频
    const card=$(`.clip[data-name="${CSS.escape(clip)}"]`);
    if(card){card.querySelector('.status').outerHTML=statusHtml(is);
      // 人工翻转 → 来源徽标也改成「✋ 人工」（原来可能是 🤖 AI 或没有），别让旧徽标留着
      const human='<span class="src-badge human">✋ 人工</span>';
      const sb=card.querySelector('.src-badge');
      if(sb){sb.outerHTML=human;}else{const st=card.querySelector('.status');if(st)st.insertAdjacentHTML('afterend',human);}
      const y=card.querySelector('.seg .yes'),no=card.querySelector('.seg .no');
      y.classList.toggle('on',is===true);no.classList.toggle('on',is===false);
      if(clipFilter!=='all'){const keep=(clipFilter==='yes'&&is===true)||(clipFilter==='no'&&is===false);
        if(!keep){card.style.transition='opacity .3s';card.style.opacity='0';setTimeout(renderClips,300);}}
    }}
  loadStats();
}

/* 训练 */
function fmtAcc(a){return (typeof a==='number')?(a*100).toFixed(1)+'%':'—';}
function fmtTime(ts){if(!ts)return '';const d=new Date(ts*1000);
  const p=n=>String(n).padStart(2,'0');return `${d.getMonth()+1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;}
/* 载入 KPI + 当前生效模型 + 版本列表（单帧训练已移除，统一用「训练视频模型」）。 */
async function pollTrain(){
  const s=await (await fetch('/api/train/status')).json();
  const ls=s.label_states||{labeled:0,drinking:0,not_drinking:0,untrained:0,trained:0};
  $('#dsUn').textContent=(s.unlabeled??'–');
  $('#dsNew').textContent=ls.untrained; $('#dsTr').textContent=ls.trained;
  $('#dsBal').textContent=`${ls.drinking} / ${ls.not_drinking}`;
  renderActive(s); renderModels(s);
}

/* 训练视频模型（s3d+head，与单帧训练并存） */
let trainVideoTimer=null;
function fmtPct(a){return (typeof a==='number')?(a*100).toFixed(0)+'%':'—';}
async function trainVideo(){
  const r=await (await fetch('/api/train_video',{method:'POST'})).json();
  if(!r.started&&r.error){$('#trainVideoStatus').textContent=r.error;return;}
  pollTrainVideo();
}
async function pollTrainVideo(){
  const s=await (await fetch('/api/train_video/status')).json();
  const btn=$('#trainVideoBtn'),st=$('#trainVideoStatus');
  if(s.state==='disabled'){btn.disabled=true;st.textContent='本入口未启用视频训练';return;}
  if(s.state==='running'){
    btn.disabled=true;st.textContent=s.detail||'训练中…';
    if(!trainVideoTimer)trainVideoTimer=setInterval(pollTrainVideo,2000);
  }else{
    if(trainVideoTimer){clearInterval(trainVideoTimer);trainVideoTimer=null;}
    btn.disabled=false;
    const r=s.result;
    if(s.state==='done'&&r){
      st.innerHTML=`完成 ${r.version} · <b>喝水召回 ${fmtPct(r.drinking_recall)}</b> `+
        `精确 ${fmtPct(r.drinking_precision)} <span style="color:#86868b">`+
        `(top1 ${fmtPct(r.top1)}，全猜没喝基线 ${fmtPct(r.naive_baseline)}；`+
        `样本 👍${r.counts.drinking}/👎${r.counts.not_drinking})</span>`;
    }else{st.textContent=s.detail||'';}
    if(s.models)renderModels(s);
  }
}
function renderActive(s){
  const box=$('#activeBox'),m=(s.models||[]).find(x=>x.id===s.active),mode=s.active_mode||'shadow';
  if(!m){
    box.innerHTML=`<div class="active-box"><span class="active-tag off">未启用</span>
      <span style="font-size:13px;color:var(--muted)">仅用简单模型兜底（宁可多录候选）</span></div>
      <p class="amini">训练出的模型默认<b>测试模式</b>：只预测、不拦截录制，简单模型继续兜底全录；
      等它在真实数据上够准了，再切「过滤模式」让它过滤误触。</p>`;
    return;
  }
  const hr=s.hitrate;
  const hrTxt=hr&&hr.total?`实战命中 <b>${hr.correct}/${hr.total}</b>（${fmtAcc(hr.rate)}）`
    :'实战命中 <b>—</b>（录到新喝水并标注后累计）';
  box.innerHTML=`<div class="active-box"><span class="active-tag on">${m.id} 生效中</span>
    <span style="font-size:13px;color:var(--muted)">验证 <b style="color:var(--ink)">${fmtAcc(m.top1)}</b> · ${hrTxt}</span></div>
    <div class="seg-ctl" style="margin-top:14px">
      <button class="${mode==='shadow'?'on':''}" onclick="activate('${m.id}','shadow')">测试模式</button>
      <button class="${mode==='gate'?'on':''}" onclick="activate('${m.id}','gate')">过滤模式</button>
    </div>
    <p class="amini">${mode==='gate'
      ?'<b>过滤模式</b>：模型判「没喝」就不录——只在它够准时用，否则会漏录真喝水。'
      :'<b>测试模式</b>：只预测打分、<b>不拦截录制</b>，简单模型兜底全录；在「视频」里看模型判得准不准。'}</p>`;
}
function renderModels(s){
  const models=s.models||[];
  let html=`<div class="mrow ${!s.active?'on':''}"><div><div class="mv">不启用任何模型</div>
    <div class="mmeta">只用简单模型，宁可多录候选</div></div><div class="grow"></div>
    <button class="mbtn ${!s.active?'cur':'off'}" ${!s.active?'':"onclick=\\"activate(null)\\""}>${!s.active?'生效中':'停用模型'}</button></div>`;
  if(!models.length){html+=`<div class="empty" style="grid-column:auto">还没有训练过的模型。标注后点上面「训练视频模型」。</div>`;}
  html+=models.map(m=>{const cur=m.id===s.active,ic=m.image_counts||{},lc=m.label_counts||{};
    return `<div class="mrow ${cur?'on':''}"><div><div class="mv">${m.id} <span class="macc">${fmtAcc(m.top1)}</span></div>
      <div class="mmeta">${fmtTime(m.created_ts)} · 抽帧 👍${ic.drinking||0}/👎${ic.not_drinking||0} · 标注 ${lc.labeled||0} 段</div></div>
      <div class="grow"></div>
      <button class="mbtn ${cur?'cur':''}" ${cur?'':`onclick="activate('${m.id}')"`}>${cur?'生效中':'设为生效'}</button></div>`;
  }).join('');
  $('#modelList').innerHTML=html;
}
async function activate(id,mode){
  document.querySelectorAll('.mbtn,#activeBox .seg-ctl button').forEach(b=>b.disabled=true);
  try{await fetch('/api/model/activate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,mode:mode||'shadow'})});}
  catch(e){}
  pollTrain();
}

/* 初始化 */
const start=(location.hash||'#home').slice(1);
show(['home','trend','clips','train'].includes(start)?start:'home');
loadStats();setInterval(loadStats,5000);
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
    trainer=None,
    registry=None,
    active_model=None,
    video_trainer=None,
) -> FastAPI:
    app = FastAPI()
    clips_dir = Path(clips_dir)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/stats/range")
    def stats_range(days: int = 7):
        days = max(1, min(int(days), 90))
        points = stats.daily_counts(datetime.now(), days)
        return {"days": [{"date": d, "count": c} for d, c in points]}

    @app.get("/chart/{span}.png")
    def chart(span: str):
        days = {"week": 7, "month": 30}.get(span)
        if days is None:
            raise HTTPException(status_code=404, detail="unknown span")
        title = "Last 7 days" if days == 7 else "Last 30 days"
        png = trend_png(stats.daily_counts(datetime.now(), days), title)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    def _unlabeled_count() -> int:
        names = [p.name for p in recorder.list_clips()]
        return sum(1 for n in names if feedback.get_label(n) is None)

    @app.post("/api/train")
    def train():
        if trainer is None:
            return JSONResponse({"started": False, "error": "本入口未启用训练"})
        # 避免重复无效训练：没有「已标注未训练」的新数据就不练。
        states = feedback.label_states()
        if states["untrained"] == 0 and states["trained"] > 0:
            return JSONResponse({"started": False, "error": "暂无新标注，无需重复训练"})
        started = trainer.start()
        return JSONResponse({"started": started,
                             "error": None if started else "已经在训练中"})

    @app.get("/api/train/status")
    def train_status():
        if trainer is None:
            return {"state": "disabled", "detail": "本入口未启用训练", "models": [], "active": None}
        s = trainer.status()
        s["unlabeled"] = _unlabeled_count()  # 待标注（当前还在的视频里没标的）
        if registry is not None:
            s["active_mode"] = registry.active_mode()
            s["hitrate"] = stats.model_hitrate(registry.active_id()) if registry.active_id() else None
        return s

    @app.post("/api/train_video")
    def train_video():
        if video_trainer is None:
            return JSONResponse({"started": False, "error": "本入口未启用视频训练"})
        started = video_trainer.start()
        return JSONResponse({"started": started,
                             "error": None if started else "已经在训练中"})

    @app.get("/api/train_video/status")
    def train_video_status():
        if video_trainer is None:
            return {"state": "disabled", "detail": "本入口未启用视频训练"}
        return video_trainer.status()

    @app.post("/api/model/activate")
    def activate(body: dict):
        if registry is None or active_model is None:
            raise HTTPException(status_code=400, detail="未启用模型管理")
        model_id = body.get("id")  # None = 停用，只用简单模型
        mode = body.get("mode") or "shadow"  # 默认测试模式（不拦截录制）
        try:
            registry.set_active(model_id, mode)
        except KeyError:
            raise HTTPException(status_code=404, detail="没有这个版本")
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

    @app.get("/api/stats/today")
    def today():
        start, end = day_bounds(datetime.now())
        events = stats.events_between(start, end)
        times = [datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S") for e in events]
        return {"count": len(events), "times": times}

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

    @app.get("/clips/{name}/thumb.jpg")
    def clip_thumb(name: str):
        # 懒加载用的封面：只解码首帧成 jpg，不拉整段视频。
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="no frame")
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

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

    @app.get("/stream.mjpg")
    def stream():
        # MJPEG：单连接持续推帧，<img> 直接当视频放，告别每秒刷快照的卡顿。
        enc = [cv2.IMWRITE_JPEG_QUALITY, 72]  # 画质够看又压住带宽，局域网更跟手
        def gen():
            while True:
                frame = frame_provider()
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, enc)
                    if ok:
                        chunk = buf.tobytes()
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(chunk)).encode()
                               + b"\r\n\r\n" + chunk + b"\r\n")
                time.sleep(0.05)  # ~20fps 上限；实际跟着采集帧率走
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/api/feedback")
    def post_feedback(body: FeedbackBody):
        if "/" in body.clip or "\\" in body.clip or ".." in body.clip:
            raise HTTPException(status_code=400, detail="bad clip")
        feedback.label_clip(clips_dir / body.clip, body.is_drinking)
        return JSONResponse({"ok": True})

    return app
