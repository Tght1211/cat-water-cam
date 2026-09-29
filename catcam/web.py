from __future__ import annotations

import shutil
import subprocess
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from catcam.audio import muted_cmd
from catcam.charts import trend_png
from catcam.dispenser import estimate_remaining, filter_days_left
from catcam.classifier import DrinkingClassifier
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.stats import StatsStore, bucket_events, day_bounds


def clip_duration(path: Path) -> float:
    """读视频头拿时长（秒），不解码全片。供网页展示「总共几秒」。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return round(frames / fps, 1) if fps > 0 else 0.0


# ============================================================================
# 前端：PC 优先的侧边栏多页应用（vanilla JS、零 CDN、全本地）。
# 信息架构：总览 / 饮水机（核心）/ 喝水记录（只看已确认喝水）/ 趋势 / 实验室（标注+训练+模型）。
# 设计：水主题 aqua glass —— 玻璃拟态卡片 + 双主题 + 手写动画（count-up、波浪水箱、
# 灯箱、toast、骨架屏），全部尊重 prefers-reduced-motion。
# 设计定稿见 docs/superpowers/specs/2026-07-05-web-ui-redesign-design.md
# ============================================================================
INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>喵喵水站 · 控制台</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🐱%3C/text%3E%3C/svg%3E">
<style>
:root{
  --bg:#edf3f8; --bg2:#e3edf5;
  --card:rgba(255,255,255,.72); --card-solid:#ffffff; --glass:rgba(255,255,255,.55);
  --ink:#0e2433; --muted:#63798a; --faint:#93a7b6; --line:rgba(15,70,105,.11);
  --aqua:#0895c8; --aqua2:#22c8b4; --coral:#ff7a68; --grape:#8f7bf1;
  --green:#12b586; --amber:#f09f1f; --red:#ef4e6e;
  --aqua-soft:rgba(9,150,200,.11); --green-soft:rgba(18,181,134,.13);
  --amber-soft:rgba(240,159,31,.14); --red-soft:rgba(239,78,110,.12);
  --shadow:0 8px 30px rgba(20,80,120,.10); --shadow-h:0 16px 44px rgba(20,80,120,.17);
  --r:20px;
  --tk1:#3fd0e8; --tk2:#0d9fd4;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#070d17; --bg2:#0a1322;
  --card:rgba(17,29,46,.66); --card-solid:#101c2e; --glass:rgba(13,23,38,.6);
  --ink:#e9f2f9; --muted:#8ba3b5; --faint:#64798c; --line:rgba(140,195,235,.13);
  --aqua:#35cdec; --aqua2:#3ee6c4; --coral:#ff8a76; --grape:#a894ff;
  --green:#2bd39e; --amber:#ffbe4d; --red:#ff6d8a;
  --aqua-soft:rgba(53,205,236,.13); --green-soft:rgba(43,211,158,.14);
  --amber-soft:rgba(255,190,77,.15); --red-soft:rgba(255,109,138,.14);
  --shadow:0 8px 30px rgba(0,0,0,.42); --shadow-h:0 16px 44px rgba(0,0,0,.55);
  --tk1:#41d8ef; --tk2:#0fa9dd;
}}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{margin:0;background:linear-gradient(160deg,var(--bg) 0%,var(--bg2) 100%);
  background-attachment:fixed;color:var(--ink);min-height:100vh;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC","Microsoft YaHei",sans-serif;
  letter-spacing:-.01em;overflow-x:hidden}
a{color:inherit}
button{font-family:inherit}
::selection{background:rgba(45,200,220,.3)}

/* ---------- 背景柔光 blob ---------- */
.blob{position:fixed;border-radius:50%;filter:blur(90px);pointer-events:none;z-index:0;opacity:.5}
.blob.b1{width:560px;height:560px;left:-140px;top:-160px;
  background:radial-gradient(circle,rgba(45,200,232,.35),transparent 65%);
  animation:drift1 26s ease-in-out infinite alternate}
.blob.b2{width:640px;height:640px;right:-200px;bottom:-240px;
  background:radial-gradient(circle,rgba(255,122,104,.22),transparent 65%);
  animation:drift2 32s ease-in-out infinite alternate}
@keyframes drift1{from{transform:translate(0,0) scale(1)}to{transform:translate(70px,50px) scale(1.12)}}
@keyframes drift2{from{transform:translate(0,0) scale(1.08)}to{transform:translate(-60px,-70px) scale(1)}}
@media (prefers-reduced-motion:reduce){.blob{animation:none}}

/* ---------- 布局：侧边栏 + 主区 ---------- */
.side{position:fixed;z-index:30;inset:0 auto 0 0;width:236px;display:flex;flex-direction:column;
  background:var(--glass);border-right:1px solid var(--line);
  backdrop-filter:blur(22px) saturate(170%);-webkit-backdrop-filter:blur(22px) saturate(170%)}
.logo{display:flex;align-items:center;gap:11px;padding:24px 22px 20px}
.logo .cat{width:42px;height:42px;flex:none;display:grid;place-items:center;border-radius:14px;
  background:linear-gradient(135deg,var(--aqua),var(--aqua2));color:#fff;
  box-shadow:0 6px 18px rgba(20,170,200,.35)}
.logo .cat svg{width:28px;height:28px}
.logo .cat .eye{transform-box:fill-box;transform-origin:center;animation:blink 5.5s infinite}
@keyframes blink{0%,94%,100%{transform:scaleY(1)}96.5%{transform:scaleY(.08)}}
@media (prefers-reduced-motion:reduce){.logo .cat .eye{animation:none}}
.logo b{font-size:16.5px;font-weight:800;letter-spacing:-.02em;display:block;line-height:1.15}
.logo small{font-size:10.5px;color:var(--muted);font-weight:600;letter-spacing:.14em}
nav.menu{position:relative;display:flex;flex-direction:column;gap:3px;padding:8px 12px;flex:1}
.menu .ind{position:absolute;left:4px;width:3.5px;height:26px;border-radius:4px;
  background:linear-gradient(180deg,var(--aqua),var(--aqua2));opacity:0;
  transition:transform .38s cubic-bezier(.3,1.4,.4,1),opacity .2s}
.menu button{position:relative;display:flex;align-items:center;gap:12px;border:0;background:transparent;
  color:var(--muted);font-size:14px;font-weight:650;padding:11px 14px;border-radius:13px;cursor:pointer;
  transition:background .18s,color .18s;text-align:left}
.menu button svg{width:19px;height:19px;flex:none;transition:transform .25s cubic-bezier(.3,1.6,.4,1)}
.menu button:hover{color:var(--ink);background:rgba(128,168,195,.09)}
.menu button.on{color:var(--aqua);background:var(--aqua-soft)}
.menu button.on svg{transform:scale(1.12)}
.menu button .adot{position:absolute;right:13px;top:50%;transform:translateY(-50%);
  width:8px;height:8px;border-radius:50%;background:var(--red);
  box-shadow:0 0 0 0 rgba(239,78,110,.5);animation:ping 2s infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(239,78,110,.45)}70%{box-shadow:0 0 0 8px rgba(239,78,110,0)}100%{box-shadow:0 0 0 0 rgba(239,78,110,0)}}
.menu .gap{flex:1}
.menu .sep{height:1px;background:var(--line);margin:8px 6px}
.menu button.lab-link{font-size:13px;color:var(--faint)}
.menu button.lab-link:hover{color:var(--ink)}
.menu button.lab-link.on{color:var(--grape);background:rgba(143,123,241,.11)}
.side-foot{padding:14px 16px 20px}
.side-pill{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,var(--aqua),var(--aqua2));
  border-radius:15px;padding:13px 16px;color:#fff;box-shadow:0 8px 22px rgba(18,165,200,.32)}
.side-pill .n{font-size:23px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1}
.side-pill .t{font-size:11px;opacity:.9;font-weight:650;margin-top:3px}
.side-pill svg{width:22px;height:22px;margin-left:auto;opacity:.85}

main{position:relative;z-index:1;margin-left:236px;padding:30px 38px 60px}
@media (max-width:1020px){
  .side{width:70px}
  .logo{padding:20px 0;justify-content:center}
  .logo div:last-child,.menu button span,.side-foot{display:none}
  .menu button{justify-content:center;padding:12px 0}
  .menu button .adot{right:8px;top:9px}
  main{margin-left:70px;padding:24px 20px 50px}
}

/* ---------- 页面切换 + 级联入场 ---------- */
.pg{display:none}
.pg.on{display:block}
.pg.on .an{opacity:0;animation:rise .55s cubic-bezier(.22,1,.36,1) both;animation-delay:calc(var(--i,0)*70ms)}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.pg.on .an{animation-duration:.01s}}
.ph{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px}
.ph h1{font-size:24px;font-weight:800;letter-spacing:-.03em;margin:0}
.ph p{color:var(--muted);font-size:13.5px;margin:6px 0 0}
.ph .phr{display:flex;align-items:center;gap:10px}

/* ---------- 通用卡片 ---------- */
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow);backdrop-filter:blur(18px) saturate(160%);
  -webkit-backdrop-filter:blur(18px) saturate(160%);overflow:hidden;
  display:flex;flex-direction:column}
.card-b{flex:1}
.card-h{display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:16px 20px 0;font-size:13.5px;font-weight:750}
.card-h .note{font-weight:600;color:var(--aqua);font-size:12.5px}
.card-b{padding:14px 20px 18px}
.hoverable{transition:transform .28s cubic-bezier(.22,1,.36,1),box-shadow .28s}
.hoverable:hover{transform:translateY(-3px);box-shadow:var(--shadow-h)}

/* ---------- 通用控件 ---------- */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:0;white-space:nowrap;
  background:linear-gradient(135deg,var(--aqua),var(--aqua2));color:#fff;
  border-radius:13px;padding:12px 22px;font-size:14px;font-weight:700;cursor:pointer;
  box-shadow:0 6px 18px rgba(18,165,200,.3);transition:transform .15s,filter .2s,box-shadow .2s}
.btn:hover{filter:brightness(1.06);box-shadow:0 8px 24px rgba(18,165,200,.4)}
.btn:active{transform:scale(.96)}
.btn:disabled{opacity:.45;cursor:default;box-shadow:none}
.btn.ghost{background:transparent;color:var(--ink);border:1px solid var(--line);box-shadow:none}
.btn.ghost:hover{background:rgba(128,168,195,.09);filter:none}
.btn.danger{background:linear-gradient(135deg,var(--red),var(--coral));box-shadow:0 6px 18px rgba(239,78,110,.3)}
.seg-ctl{display:inline-flex;background:rgba(128,168,195,.12);border:1px solid var(--line);
  border-radius:12px;padding:3px;gap:2px}
.seg-ctl button{border:0;background:transparent;color:var(--muted);font-size:13px;
  font-weight:650;padding:7px 16px;border-radius:9px;cursor:pointer;transition:.2s}
.seg-ctl button.on{background:var(--card-solid);color:var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.14)}
.fchip{border:1px solid var(--line);background:var(--card);color:var(--muted);border-radius:99px;
  padding:8px 16px;font-size:13px;font-weight:650;cursor:pointer;transition:.18s}
.fchip:hover{color:var(--ink)}
.fchip.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:18px}
.tag{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:700;
  padding:4px 10px;border-radius:99px;white-space:nowrap}
.tag i{width:6px;height:6px;border-radius:50%;flex:none}
.tag.ok{background:var(--green-soft);color:var(--green)}.tag.ok i{background:var(--green)}
.tag.warn{background:var(--amber-soft);color:var(--amber)}.tag.warn i{background:var(--amber)}
.tag.bad{background:var(--red-soft);color:var(--red)}.tag.bad i{background:var(--red)}
.tag.mute{background:rgba(128,168,195,.13);color:var(--muted)}.tag.mute i{background:var(--muted)}
.tag.ai{background:var(--aqua-soft);color:var(--aqua)}
.empty{color:var(--muted);background:var(--card);border:1px dashed var(--line);
  border-radius:var(--r);padding:52px 26px;text-align:center;font-size:14px;line-height:1.8}
.empty .eico{font-size:38px;display:block;margin-bottom:8px;animation:bob 3s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@media (prefers-reduced-motion:reduce){.empty .eico{animation:none}}
.empty code{background:rgba(128,168,195,.14);border:1px solid var(--line);padding:2px 8px;border-radius:7px;font-size:12px}

/* ---------- 总览 ---------- */
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}
.sp3{grid-column:span 3}.sp4{grid-column:span 4}.sp5{grid-column:span 5}
.sp6{grid-column:span 6}.sp7{grid-column:span 7}.sp8{grid-column:span 8}.sp12{grid-column:span 12}
@media (max-width:1240px){.sp3,.sp4,.sp5{grid-column:span 6}.sp7,.sp8{grid-column:span 12}}
@media (max-width:760px){.grid>*{grid-column:span 12}}
.hero-num{font-size:66px;font-weight:800;letter-spacing:-.04em;line-height:1;
  font-variant-numeric:tabular-nums;background:linear-gradient(135deg,var(--aqua),var(--aqua2));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero-num small{font-size:19px;font-weight:700;-webkit-text-fill-color:var(--muted);margin-left:6px;letter-spacing:0}
.hero-sub{margin-top:10px}
.mini-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:20px;
  border-top:1px solid var(--line);padding-top:16px}
.mini-stats .ms .k{font-size:11.5px;color:var(--muted);font-weight:650;margin-bottom:5px}
.mini-stats .ms .v{font-size:19px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.mini-stats .ms .v small{font-size:11px;color:var(--muted);font-weight:650;margin-left:2px}
/* 首页 hero：近 7 天迷你柱状 */
.spark{margin-top:auto;padding-top:16px}
.spark .sp-l{font-size:11.5px;color:var(--muted);font-weight:650;margin-bottom:8px}
.spark .sp-bars{display:flex;align-items:flex-end;gap:7px;height:54px}
.spark .sb{flex:1;border-radius:6px 6px 3px 3px;min-height:4px;position:relative;
  background:linear-gradient(180deg,var(--aqua2),var(--aqua));opacity:.85;
  transform-origin:bottom;animation:sgrow .6s cubic-bezier(.22,1,.36,1) both;
  animation-delay:calc(var(--i,0)*60ms);transition:opacity .15s}
.spark .sb:hover{opacity:1}
.spark .sb.zero{background:rgba(128,168,195,.2)}
@keyframes sgrow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.spark .sp-days{display:flex;gap:7px;margin-top:6px}
.spark .sp-days span{flex:1;text-align:center;font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion:reduce){.spark .sb{animation-duration:.01s}}
.live-frame{position:relative;border-radius:14px;overflow:hidden;background:#05090f;aspect-ratio:4/3;cursor:zoom-in}
.live-frame img{display:block;width:100%;height:100%;object-fit:cover}
.live-tag{position:absolute;top:12px;left:12px;display:flex;align-items:center;gap:7px;
  background:rgba(4,10,16,.55);color:#fff;border-radius:99px;padding:6px 13px;
  font-size:11px;font-weight:700;letter-spacing:.04em;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:ping2 2s infinite}
@keyframes ping2{0%{box-shadow:0 0 0 0 rgba(18,181,134,.55)}70%{box-shadow:0 0 0 7px rgba(18,181,134,0)}100%{box-shadow:0 0 0 0 rgba(18,181,134,0)}}
.live-zoom{position:absolute;right:12px;top:12px;width:34px;height:34px;border:0;border-radius:11px;
  background:rgba(4,10,16,.55);color:#fff;display:grid;place-items:center;cursor:pointer;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);transition:transform .15s}
.live-zoom:hover{transform:scale(1.1)}
.live-zoom svg{width:16px;height:16px}
/* 今日时间线 */
.timeline{position:relative;height:78px;margin:6px 8px 0}
.tl-track{position:absolute;left:0;right:0;top:32px;height:6px;border-radius:5px;
  background:linear-gradient(90deg,rgba(128,168,195,.13),rgba(128,168,195,.2),rgba(128,168,195,.13));
  border:1px solid var(--line)}
.tl-tick{position:absolute;top:26px;transform:translateX(-50%)}
.tl-tick i{display:block;width:1px;height:12px;background:var(--line);margin:0 auto}
.tl-tick span{position:absolute;top:16px;left:50%;transform:translateX(-50%);font-size:10px;
  color:var(--faint);white-space:nowrap;font-variant-numeric:tabular-nums}
.tl-now{position:absolute;top:24px;height:22px;width:2px;background:var(--coral);border-radius:2px;transform:translateX(-50%)}
.tl-now::after{content:"";position:absolute;top:-4px;left:50%;width:7px;height:7px;border-radius:50%;
  background:var(--coral);transform:translateX(-50%);box-shadow:0 0 0 3px rgba(255,122,104,.22)}
.tl-dot{position:absolute;top:26px;width:16px;height:16px;border-radius:50%;
  background:linear-gradient(135deg,var(--aqua),var(--aqua2));border:3px solid var(--card-solid);
  transform:translateX(-50%);box-shadow:0 2px 8px rgba(10,140,190,.4);cursor:pointer;
  transition:transform .18s cubic-bezier(.3,1.6,.4,1);z-index:2;animation:plop .4s cubic-bezier(.3,1.6,.4,1) both;
  animation-delay:calc(var(--i,0)*60ms)}
@keyframes plop{from{transform:translateX(-50%) scale(0)}to{transform:translateX(-50%) scale(1)}}
.tl-dot:hover{transform:translateX(-50%) scale(1.35)}
.tl-empty{position:absolute;top:-4px;left:2px;color:var(--muted);font-size:12.5px;font-weight:600}

/* ---------- 饮水机（霍曼三代 Pro 的二次元 Q 版：顶盆喷泉 + 水位窗 + 表情） ---------- */
.tankbox{position:relative;flex:none}
.tank{display:block;overflow:visible}
.tank .lvl{transition:transform 1s cubic-bezier(.25,1,.35,1)}
/* 水位窗里的小波浪 */
.tank .wv{animation:waveXs 1.9s linear infinite}
.tank .wv2{animation-duration:2.8s;animation-direction:reverse}
@keyframes waveXs{from{transform:translateX(0)}to{transform:translateX(-24px)}}
/* 水位窗气泡 */
.tank .bub{animation:bubW 3.2s ease-in infinite;opacity:0}
@keyframes bubW{0%{transform:translateY(0);opacity:0}20%{opacity:.7}100%{transform:translateY(-58px);opacity:0}}
/* 顶盆涟漪（从出水舌扩散） */
.tank .rip{transform-box:fill-box;transform-origin:center;animation:rip 2.8s ease-out infinite}
.tank .rip.r2{animation-delay:1.4s}
@keyframes rip{0%{transform:scale(.22);opacity:.8}70%{opacity:.25}100%{transform:scale(1.12);opacity:0}}
/* 喷泉水流 + 水珠 */
.tank .jet{animation:jetflow .65s linear infinite}
@keyframes jetflow{to{stroke-dashoffset:-10}}
.tank .jd{animation:jdrop 1.15s ease-in infinite}
.tank .jd.d2{animation-delay:.55s}
@keyframes jdrop{0%{transform:translateY(-3px);opacity:0}25%{opacity:.9}100%{transform:translateY(8px);opacity:0}}
/* 表情：眨眼 / 低水位换担忧脸 + 汗滴 */
.tank .eye{transform-box:fill-box;transform-origin:center;animation:blink 4.8s infinite}
.tank .f-low{display:none}
.tank.low .f-low,.tank.empty .f-low{display:block}
.tank.low .f-normal,.tank.empty .f-normal{display:none}
.tank .sweat{animation:sweat 1.7s ease-in-out infinite}
@keyframes sweat{0%,100%{transform:translateY(0);opacity:.85}50%{transform:translateY(3.5px);opacity:.55}}
/* 星星闪烁 */
.tank .spk{transform-box:fill-box;transform-origin:center;animation:twinkle 3.4s ease-in-out infinite}
.tank .spk.s2{animation-delay:1.7s}
@keyframes twinkle{0%,100%{opacity:.12;transform:scale(.65)}50%{opacity:.9;transform:scale(1.08)}}
/* LED 呼吸 */
.tank .led{animation:ledpulse 2.6s ease-in-out infinite}
@keyframes ledpulse{0%,100%{opacity:.35}50%{opacity:.95}}
/* 状态：水色随水位变（覆盖渐变 stop 的 CSS 变量） */
.tank.warn{--tk1:#ffc255;--tk2:#f08b2d}
.tank.low{--tk1:#ff7d90;--tk2:#ff6a4d}
.tank.empty .basin-w,.tank.empty .fount,.tank.empty .rip{display:none}
.tankbox.glow-low{filter:drop-shadow(0 12px 28px rgba(239,78,110,.32))}
.tankbox.glow-ok{filter:drop-shadow(0 12px 28px rgba(28,175,215,.28))}
/* 加水：连续落水滴 */
.tank .pourdrop{opacity:0}
.tank.pouring .pourdrop{animation:dropfall .48s ease-in 5}
@keyframes dropfall{0%{opacity:0;transform:translateY(-4px)}18%{opacity:1}88%{opacity:.9;transform:translateY(30px)}100%{opacity:0;transform:translateY(33px)}}
@media (prefers-reduced-motion:reduce){
  .tank .wv,.tank .bub,.tank .rip,.tank .jet,.tank .jd,.tank .eye,.tank .sweat,.tank .spk,.tank .led{animation:none}}
/* 气泡对话框（报水量，二次元味） */
.tank-bub{position:absolute;top:-2px;right:-16px;background:var(--card-solid);border:1.5px solid var(--line);
  border-radius:15px;padding:8px 14px;font-weight:800;font-size:19px;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;box-shadow:var(--shadow);white-space:nowrap;z-index:2;
  animation:bobble 3.6s ease-in-out infinite}
.tank-bub::after{content:"";position:absolute;left:16px;bottom:-7.5px;width:12px;height:12px;
  background:inherit;border-right:1.5px solid var(--line);border-bottom:1.5px solid var(--line);
  transform:rotate(45deg)}
.tank-bub small{display:block;font-size:10.5px;color:var(--muted);font-weight:700;margin-top:1px;letter-spacing:0}
.tank-bub.low{border-color:rgba(239,78,110,.55);color:var(--red)}
.tank-bub.low::after{border-color:rgba(239,78,110,.55)}
.tank-bub.warn{border-color:rgba(240,159,31,.55);color:var(--amber)}
.tank-bub.warn::after{border-color:rgba(240,159,31,.55)}
.tank-bub.mini{font-size:13.5px;padding:5px 10px;top:0;right:-10px;border-radius:11px}
@keyframes bobble{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@media (prefers-reduced-motion:reduce){.tank-bub{animation:none}}

/* ---------- 饮水机页 ---------- */
.disp-hero{display:flex;align-items:center;gap:40px;flex-wrap:wrap;padding:10px 8px}
.disp-main{flex:1;min-width:280px}
.disp-remain-line{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.disp-remain-line .big{font-size:40px;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.disp-remain-line .big small{font-size:15px;color:var(--muted);font-weight:650}
.disp-alert{display:inline-flex;align-items:center;gap:8px;margin-top:12px;
  background:var(--red-soft);color:var(--red);border-radius:12px;padding:9px 15px;
  font-size:13px;font-weight:750;animation:alertPulse 2.4s ease-in-out infinite}
.disp-alert svg{width:16px;height:16px;flex:none}
.disp-alert.amber{background:var(--amber-soft);color:var(--amber)}
@keyframes alertPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.025)}}
@media (prefers-reduced-motion:reduce){.disp-alert{animation:none}}
.disp-cells{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px}
@media (max-width:1100px){.disp-cells{grid-template-columns:repeat(2,1fr)}}
.dcell{background:rgba(128,168,195,.08);border:1px solid var(--line);border-radius:16px;
  padding:15px 17px;transition:background .2s}
.dcell:hover{background:rgba(128,168,195,.13)}
.dcell .k{font-size:11.5px;color:var(--muted);font-weight:650;margin-bottom:8px}
.dcell .v{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.dcell .v small{font-size:12px;color:var(--muted);font-weight:650;margin-left:3px}
.dcell.warn{border-color:rgba(239,78,110,.4);background:var(--red-soft)}
.dcell.warn .v{color:var(--red)}
.dcell .sub{font-size:11px;color:var(--faint);margin-top:5px}
.ringwrap{display:flex;align-items:center;gap:10px}
.ring{transform:rotate(-90deg)}
.ring .bgc{stroke:rgba(128,168,195,.18);fill:none}
.ring .fgc{stroke:url(#ringGrad);fill:none;stroke-linecap:round;
  transition:stroke-dashoffset 1s cubic-bezier(.25,1,.35,1)}
.ring.bad .fgc{stroke:var(--red)}
.disp-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:26px}
.calib-badge{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;font-weight:750;
  padding:2px 8px;border-radius:99px;background:var(--green-soft);color:var(--green);vertical-align:2px}
.calib-badge.default{background:rgba(128,168,195,.13);color:var(--muted)}
/* 总览页速览小水箱 */
.dmini{display:flex;align-items:center;gap:20px;cursor:pointer;height:100%;padding:6px 2px}
.dmini-info .dm-sub{font-size:11.5px;color:var(--faint);margin-top:7px}
.dmini-info .dm-l{font-size:12px;color:var(--muted);font-weight:650}
.dmini-info .dm-v{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums;margin-top:3px;letter-spacing:-.02em}
.dmini-info .dm-v small{font-size:11.5px;color:var(--muted);font-weight:650}
.dmini-info .go{display:inline-flex;align-items:center;gap:4px;margin-top:10px;
  color:var(--aqua);font-size:12.5px;font-weight:700}
.dmini-info .go svg{width:13px;height:13px;transition:transform .2s}
.dmini:hover .go svg{transform:translateX(3px)}

/* ---------- 视频卡（记录 + 标注共用底座） ---------- */
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;perspective:1200px}
.clip{position:relative;background:var(--card);border:1px solid var(--line);border-radius:18px;overflow:hidden;
  display:flex;flex-direction:column;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  transition:transform .2s ease-out,box-shadow .28s,border-color .28s;box-shadow:var(--shadow);
  transform:rotateX(calc(var(--ry,0)*1deg)) rotateY(calc(var(--rx,0)*1deg)) translateY(var(--lift,0px));
  opacity:0;animation:rise .5s cubic-bezier(.22,1,.36,1) both;animation-delay:calc(var(--i,0)*45ms)}
.clip:hover{--lift:-4px;box-shadow:var(--shadow-h);border-color:rgba(45,190,225,.4)}
.clip::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;z-index:1;
  background:radial-gradient(340px circle at var(--mx,50%) var(--my,20%),rgba(80,200,235,.13),transparent 55%);
  opacity:0;transition:opacity .3s}
.clip:hover::before{opacity:1}
@media (prefers-reduced-motion:reduce){.clip{animation-duration:.01s;transform:none!important}}
.thumb{position:relative;aspect-ratio:4/3;background:#05090f;overflow:hidden}
.thumb .inline-clip{display:block;width:100%;height:100%;object-fit:contain;background:#05090f}
.thumb .dur{position:absolute;right:9px;top:9px;background:rgba(4,10,16,.66);color:#fff;
  border-radius:8px;padding:3px 9px;font-size:11px;font-weight:650;font-variant-numeric:tabular-nums;
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px)}
.cmeta{padding:12px 14px 14px;display:flex;flex-direction:column;gap:10px}
.cmeta .row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.ctime{display:flex;align-items:center;gap:6px;font-size:14px;font-weight:750;font-variant-numeric:tabular-nums}
.ctime svg{width:14px;height:14px;color:var(--aqua)}
.fname{font-size:10.5px;color:var(--faint);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:130px}
.dlbtn{display:inline-flex;align-items:center;gap:5px;color:var(--aqua);font-size:12px;
  text-decoration:none;font-weight:650;transition:opacity .15s}
.dlbtn:hover{opacity:.75}
.dlbtn svg{width:13px;height:13px}
.seg2{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.seg2 button{display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border:1px solid var(--line);background:transparent;color:var(--ink);
  border-radius:11px;padding:9px 0;font-size:13px;font-weight:600;cursor:pointer;
  transition:transform .12s,background .15s,border-color .15s,color .15s}
.seg2 button svg{width:14px;height:14px}
.seg2 button:hover{background:rgba(128,168,195,.09)}
.seg2 button:active{transform:scale(.94)}
.seg2 .yes.on{background:var(--green);border-color:var(--green);color:#fff}
.seg2 .no.on{background:var(--red);border-color:var(--red);color:#fff}
/* 人工审核：紧凑网格，同屏快速判断多段候选。 */
#labelClips{grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
#labelClips .review-card{opacity:1;transform:none;animation:none;border-radius:14px}
#labelClips .review-card:hover{--lift:-2px}
#labelClips .review-card .thumb{aspect-ratio:4/3}
#labelClips .review-side{padding:11px 12px 12px;display:flex;flex-direction:column;gap:9px}
#labelClips .review-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
#labelClips .review-title{font-size:14px;font-weight:800;font-variant-numeric:tabular-nums}
#labelClips .review-side .row{min-width:0}
#labelClips .review-side .seg2{margin-top:1px}
#labelClips .review-side .seg2 button{min-height:38px}
#labelClips .review-foot{display:flex;align-items:center;justify-content:space-between;gap:8px}
@media (min-width:1700px){#labelClips{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media (max-width:720px){#labelClips{grid-template-columns:1fr}}
.mpred{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;padding:3px 9px;border-radius:99px}
.mpred.y{background:var(--green-soft);color:var(--green)}
.mpred.n{background:var(--amber-soft);color:var(--amber)}
.grp-head{grid-column:1/-1;display:flex;align-items:center;gap:10px;font-size:14.5px;
  font-weight:750;margin:14px 2px 0}
.grp-head:first-child{margin-top:0}
.grp-dot{width:9px;height:9px;border-radius:50%;background:linear-gradient(135deg,var(--aqua),var(--aqua2));
  box-shadow:0 0 0 4px var(--aqua-soft)}
.grp-n{font-size:11.5px;font-weight:650;color:var(--muted);background:rgba(128,168,195,.12);
  border:1px solid var(--line);padding:2px 10px;border-radius:99px}
.more{grid-column:1/-1;text-align:center;color:var(--muted);font-size:13px;font-weight:650;
  padding:16px;border:1px dashed var(--line);border-radius:14px;cursor:pointer;margin-top:6px;transition:.15s}
.more:hover{color:var(--aqua);border-color:var(--aqua)}
/* 骨架屏 */
.skel{border-radius:18px;aspect-ratio:4/3.6;background:linear-gradient(100deg,
  rgba(128,168,195,.10) 40%,rgba(128,168,195,.22) 50%,rgba(128,168,195,.10) 60%);
  background-size:220% 100%;animation:shimmer 1.3s linear infinite}
@keyframes shimmer{to{background-position:-120% 0}}
@media (prefers-reduced-motion:reduce){.skel{animation:none}}

/* ---------- 趋势 ---------- */
.trend-hero{display:flex;align-items:center;gap:24px;padding:6px 4px}
.trend-drop{width:64px;height:64px;flex:none;border-radius:20px;display:grid;place-items:center;
  background:var(--aqua-soft);color:var(--aqua)}
.trend-drop svg{width:32px;height:32px}
.trend-hero.good .trend-drop{background:var(--green-soft);color:var(--green)}
.trend-hero.low .trend-drop{background:var(--amber-soft);color:var(--amber)}
.trend-hero.none .trend-drop{background:rgba(128,168,195,.12);color:var(--muted)}
.trend-today{font-size:44px;font-weight:800;letter-spacing:-.03em;line-height:1;font-variant-numeric:tabular-nums}
.trend-today small{font-size:14px;color:var(--muted);font-weight:650;margin-left:6px;letter-spacing:0}
.trend-status{display:inline-block;margin-top:10px;font-size:13.5px;font-weight:700;
  padding:5px 14px;border-radius:99px;background:var(--aqua-soft);color:var(--aqua)}
.trend-hero.good .trend-status{background:var(--green-soft);color:var(--green)}
.trend-hero.low .trend-status{background:var(--amber-soft);color:var(--amber)}
.trend-hero.none .trend-status{background:rgba(128,168,195,.12);color:var(--muted)}
.trend-sub{margin-top:10px;color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.trend-sub b{color:var(--ink);font-weight:750}
.heatrow{display:grid;grid-template-columns:repeat(24,1fr);gap:3px}
.heatrow .hcell{height:42px;border-radius:6px;transition:transform .15s;transform-origin:bottom;
  animation:hgrow .5s cubic-bezier(.22,1,.36,1) both;animation-delay:calc(var(--i,0)*18ms)}
@keyframes hgrow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.heatrow .hcell:hover{transform:scaleY(1.12)}
.heataxis{display:flex;justify-content:space-between;margin-top:8px;color:var(--faint);
  font-size:11px;font-variant-numeric:tabular-nums}
.calheat{display:flex;flex-wrap:wrap;gap:6px}
.calheat .ccell{width:40px;height:40px;border-radius:10px;display:grid;place-items:center;
  font-size:11.5px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;
  transition:transform .15s;animation:cin .4s cubic-bezier(.3,1.5,.4,1) both;animation-delay:calc(var(--i,0)*22ms)}
@keyframes cin{from{transform:scale(0)}to{transform:scale(1)}}
.calheat .ccell:hover{transform:scale(1.14)}
.callegend{display:flex;align-items:center;gap:5px;margin-top:14px;color:var(--faint);font-size:12px}
.callegend .cl{width:16px;height:16px;border-radius:5px;display:inline-block}

/* ---------- 实验室 ---------- */
.lab-tabs{margin-bottom:20px}
.lab-pane{display:none}
.lab-pane.on{display:block;animation:rise .4s cubic-bezier(.22,1,.36,1)}
.krow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
@media (max-width:900px){.krow{grid-template-columns:repeat(2,1fr)}}
.kbox{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:15px 18px;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
.kbox .k{font-size:11.5px;color:var(--muted);font-weight:650;margin-bottom:8px}
.kbox .v{font-size:26px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kbox .v small{font-size:12px;color:var(--muted);font-weight:650;margin-left:3px}
.pbar-track{height:9px;border-radius:99px;background:rgba(128,168,195,.14);border:1px solid var(--line);
  overflow:hidden;position:relative}
.pbar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--aqua),var(--aqua2))}
.pbar-fill.det{transition:width .45s cubic-bezier(.22,1,.36,1)}
.pbar-fill.indet{position:absolute;left:0;width:34%;animation:indet 1.15s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-110%)}100%{transform:translateX(310%)}}
.pbar-lab{display:flex;justify-content:space-between;gap:10px;margin-top:9px;
  font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.t-status{margin-top:14px;color:var(--muted);font-size:13.5px;min-height:18px;line-height:1.7}
.mlist{display:flex;flex-direction:column;gap:12px}
.mrow{display:flex;align-items:center;gap:16px;background:var(--card);border:1px solid var(--line);
  border-radius:16px;padding:15px 20px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  transition:border-color .2s,box-shadow .2s}
.mrow.on{border-color:var(--aqua);box-shadow:0 0 0 2.5px var(--aqua-soft)}
.mrow .mv{font-size:15.5px;font-weight:750;display:flex;align-items:center;gap:9px}
.mrow .macc{font-size:11.5px;font-weight:700;color:var(--aqua);background:var(--aqua-soft);padding:2px 9px;border-radius:99px}
.mrow .mmeta{color:var(--muted);font-size:12.5px;margin-top:5px;font-variant-numeric:tabular-nums}
.mrow .grow{flex:1}
.mbtn{border:1px solid var(--line);background:transparent;color:var(--aqua);border-radius:11px;
  padding:9px 18px;font-weight:650;font-size:13px;cursor:pointer;transition:.15s;white-space:nowrap}
.mbtn:hover:not(:disabled){background:var(--aqua-soft)}
.mbtn.cur{background:var(--green);border-color:var(--green);color:#fff;cursor:default}
.mbtn.off{color:var(--muted)}
.mbtn:disabled{opacity:.5}
.amini{color:var(--muted);font-size:12.5px;line-height:1.7;margin:14px 0 0}
.amini b{color:var(--ink)}
.lab-note{color:var(--muted);font-size:13px;line-height:1.8;margin:0 0 16px}
.lab-note b{color:var(--ink)}

/* ---------- 灯箱 / 模态 / toast ---------- */
.lbox{position:fixed;inset:0;z-index:80;display:none;place-items:center;padding:38px;
  background:rgba(5,12,20,.68);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
.lbox.show{display:grid;animation:lfade .25s ease}
@keyframes lfade{from{opacity:0}to{opacity:1}}
.lbox-inner{position:relative;max-width:min(1080px,92vw);width:100%;
  animation:lpop .35s cubic-bezier(.25,1.3,.4,1)}
@keyframes lpop{from{opacity:0;transform:scale(.94) translateY(10px)}to{opacity:1;transform:none}}
.lbox-inner video,.lbox-inner img.big{display:block;width:100%;max-height:76vh;object-fit:contain;
  border-radius:18px;background:#000;box-shadow:0 30px 80px rgba(0,0,0,.5)}
.lbox-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:14px;color:#fff}
.lbox-bar .lt{font-size:15px;font-weight:750;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:9px;white-space:nowrap}
.lbox-bar .lt svg{width:16px;height:16px;flex:none}
.lbox-bar a{color:#fff;text-decoration:none;display:inline-flex;align-items:center;gap:6px;white-space:nowrap;
  font-size:13px;font-weight:650;background:rgba(255,255,255,.14);border-radius:11px;padding:9px 16px;transition:.15s}
.lbox-bar a svg{width:14px;height:14px;flex:none}
.lbox-bar a:hover{background:rgba(255,255,255,.24)}
.lbox-x{position:absolute;top:-14px;right:-14px;width:38px;height:38px;border:0;border-radius:50%;
  background:#fff;color:#0a1420;display:grid;place-items:center;cursor:pointer;
  box-shadow:0 6px 20px rgba(0,0,0,.4);transition:transform .15s;z-index:2}
.lbox-x:hover{transform:scale(1.1) rotate(90deg)}
.lbox-x svg{width:16px;height:16px}
.lbox-nav{position:absolute;top:50%;transform:translateY(-50%);width:46px;height:46px;border:0;border-radius:50%;
  background:rgba(255,255,255,.14);color:#fff;display:grid;place-items:center;cursor:pointer;z-index:2;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);transition:background .15s,transform .15s}
.lbox-nav:hover{background:rgba(255,255,255,.3);transform:translateY(-50%) scale(1.08)}
.lbox-nav.prev{left:-60px}.lbox-nav.next{right:-60px}
.lbox-nav svg{width:20px;height:20px}
.lbox-nav:disabled{opacity:.25;cursor:default}
@media (max-width:1200px){.lbox-nav.prev{left:10px}.lbox-nav.next{right:10px}}
.lbox-bar .ltags{display:flex;align-items:center;gap:8px}
/* AI 置信度条（标注卡） */
.confbar{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);font-weight:650}
.confbar .cb-t{height:5px;border-radius:4px;background:rgba(128,168,195,.16);flex:1;overflow:hidden}
.confbar .cb-f{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--aqua),var(--aqua2));
  transition:width .6s cubic-bezier(.22,1,.36,1)}
.modal{position:fixed;inset:0;z-index:90;display:none;place-items:center;padding:20px;
  background:rgba(5,12,20,.55);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.modal.show{display:grid;animation:lfade .22s ease}
.modal-card{background:var(--card-solid);border:1px solid var(--line);border-radius:22px;
  box-shadow:0 30px 80px rgba(0,0,0,.35);width:min(430px,92vw);padding:26px 26px 22px;
  animation:lpop .34s cubic-bezier(.25,1.3,.4,1)}
.modal-card h3{margin:0 0 6px;font-size:18px;font-weight:800;letter-spacing:-.02em}
.modal-card .msub{color:var(--muted);font-size:13px;margin:0 0 18px;line-height:1.6}
.modal-acts{display:flex;gap:10px;justify-content:flex-end;margin-top:22px}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.presets button{border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:11px;
  padding:8px 14px;font-size:13px;font-weight:650;cursor:pointer;transition:.15s;font-variant-numeric:tabular-nums}
.presets button:hover{border-color:var(--aqua)}
.presets button.on{background:var(--aqua-soft);border-color:var(--aqua);color:var(--aqua)}
.mlrow{display:flex;align-items:center;gap:14px}
.mlrow input[type=range]{flex:1;accent-color:var(--aqua)}
.mlnum{display:flex;align-items:baseline;gap:4px;font-variant-numeric:tabular-nums}
.mlnum input{width:84px;border:1px solid var(--line);background:var(--card);color:var(--ink);
  border-radius:10px;padding:9px 11px;font-size:16px;font-weight:750;font-family:inherit;text-align:right;
  font-variant-numeric:tabular-nums}
.mlnum input:focus{outline:2px solid var(--aqua);outline-offset:1px;border-color:transparent}
.mlnum span{color:var(--muted);font-size:13px;font-weight:650}
.stepper{display:flex;align-items:center;gap:14px;justify-content:center;margin:6px 0 2px}
.stepper button{width:42px;height:42px;border:1px solid var(--line);background:transparent;color:var(--ink);
  border-radius:13px;font-size:20px;font-weight:700;cursor:pointer;transition:.15s}
.stepper button:hover{background:var(--aqua-soft);border-color:var(--aqua);color:var(--aqua)}
.stepper .sv{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;min-width:86px;text-align:center}
.stepper .sv small{font-size:13px;color:var(--muted);font-weight:650}
#toasts{position:fixed;right:24px;bottom:24px;z-index:120;display:flex;flex-direction:column;gap:10px}
.toast{display:flex;align-items:center;gap:10px;background:var(--card-solid);border:1px solid var(--line);
  border-radius:14px;padding:13px 18px;font-size:13.5px;font-weight:650;box-shadow:var(--shadow-h);
  animation:tin .4s cubic-bezier(.25,1.4,.4,1)}
.toast.out{animation:tout .3s ease forwards}
@keyframes tin{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:none}}
@keyframes tout{to{opacity:0;transform:translateX(40px)}}
.toast .ti{width:26px;height:26px;flex:none;border-radius:9px;display:grid;place-items:center;color:#fff}
.toast .ti svg{width:14px;height:14px}
.toast.ok .ti{background:var(--green)} .toast.err .ti{background:var(--red)}
.toast.info .ti{background:var(--aqua)}
</style></head>
<body>
<div class="blob b1"></div><div class="blob b2"></div>

<aside class="side">
  <div class="logo">
    <span class="cat" aria-hidden="true"><svg viewBox="0 0 44 44" fill="none">
      <path d="M9 8l6 6a13 13 0 0 1 14 0l6-6-1.5 12" fill="currentColor"/>
      <ellipse cx="22" cy="25" rx="15" ry="13" fill="currentColor"/>
      <circle class="eye" cx="16" cy="23" r="2.1" fill="#0b3b4d"/>
      <circle class="eye" cx="28" cy="23" r="2.1" fill="#0b3b4d"/>
      <path d="M22 28l-2 2h4z" fill="#0b3b4d"/>
      <path d="M6 27h7M31 27h7M6 31h6M32 31h6" stroke="#0b3b4d" stroke-width="1.4" stroke-linecap="round" opacity=".7"/></svg></span>
    <div><b>喵喵水站</b><small>CAT · WATER · CAM</small></div>
  </div>
  <nav class="menu" id="menu">
    <span class="ind" id="navInd"></span>
    <button data-pg="home"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9.5h13V10"/></svg><span>总览</span></button>
    <button data-pg="disp" id="navDisp"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.2S5.8 9.8 5.8 14a6.2 6.2 0 0 0 12.4 0C18.2 9.8 12 3.2 12 3.2Z"/><path d="M9.4 14.6a2.8 2.8 0 0 0 2.3 2.7"/></svg><span>饮水机</span></button>
    <button data-pg="records"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="m10 9.5 5 2.5-5 2.5z" fill="currentColor" stroke="none"/></svg><span>喝水记录</span></button>
    <button data-pg="trend"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19 9.5 11l4 4.5L20 7"/><path d="M15.5 7H20v4.5"/></svg><span>趋势</span></button>
    <div class="gap"></div>
    <div class="sep"></div>
    <button data-pg="lab" class="lab-link"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 3v6L4.6 17.2A2 2 0 0 0 6.3 20h11.4a2 2 0 0 0 1.7-2.8L14.5 9V3"/><path d="M8 3h8M7.5 14.5h9"/></svg><span>实验室</span></button>
  </nav>
  <div class="side-foot">
    <div class="side-pill">
      <div><div class="n" id="sideCount">–</div><div class="t">今日喝水（次）</div></div>
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg>
    </div>
  </div>
</aside>

<main>

<!-- ============ 总览 ============ -->
<section id="pg-home" class="pg">
  <div class="ph an" style="--i:0"><div><h1>总览</h1><p id="homeDate"></p></div>
    <div class="phr"><span class="tag ok"><i></i>本地运行 · 画面不出局域网</span></div></div>
  <div class="grid">
    <div class="card sp4 an" style="--i:1"><div class="card-h">今日喝水</div>
    <div class="card-b" style="display:flex;flex-direction:column">
      <div class="hero-num"><span id="heroCount">0</span><small>次</small></div>
      <div class="hero-sub"><span class="tag mute" id="heroLast"><i></i>暂无记录</span></div>
      <div class="mini-stats">
        <div class="ms"><div class="k">近 7 天</div><div class="v"><span id="msWeek">–</span><small>次</small></div></div>
        <div class="ms"><div class="k">日均</div><div class="v"><span id="msAvg">–</span><small>次</small></div></div>
        <div class="ms"><div class="k">今日约喝</div><div class="v"><span id="msMl">–</span><small>ml</small></div></div>
      </div>
      <div class="spark" id="spark"></div>
    </div></div>
    <div class="card sp5 an" style="--i:2"><div class="card-h">实时画面</div><div class="card-b">
      <div class="live-frame" onclick="openLive()">
        <img id="live" src="/stream.mjpg" alt="实时画面" onerror="this.src='/snapshot.jpg?t='+Date.now()">
        <span class="live-tag"><span class="live-dot"></span>LIVE</span>
        <button class="live-zoom" aria-label="放大" onclick="event.stopPropagation();openLive()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5"/></svg></button>
      </div>
      <p id="visibilityStatus" class="lab-note" role="status" style="margin-top:12px">正在检查水碗区域光照…</p>
    </div></div>
    <div class="card sp3 hoverable an" style="--i:3"><div class="card-h">饮水机 <span class="note" id="dmNote"></span></div>
      <div class="card-b"><div id="dmBody" class="dmini" onclick="go('disp')"></div></div></div>
    <div class="card sp12 an" style="--i:4"><div class="card-h">今日时间线 <span class="note" id="tlNow"></span></div>
      <div class="card-b"><div id="timeline" class="timeline"></div></div></div>
  </div>
</section>

<!-- ============ 饮水机 ============ -->
<section id="pg-disp" class="pg">
  <div class="ph an" style="--i:0"><div><h1>饮水机</h1><p>剩余水量按「喝水次数 × 每次毫升」反推 · 每次加满水都会自动校准</p></div></div>
  <div class="card an" style="--i:1"><div class="card-b"><div id="dispBody"></div></div></div>
</section>

<!-- ============ 喝水记录 ============ -->
<section id="pg-records" class="pg">
  <div class="ph an" style="--i:0"><div><h1>喝水记录</h1><p>只展示已确认「真喝水」的回放 · 点封面即可播放</p></div>
    <div class="phr"><span class="tag ok" id="recCount" style="display:none"><i></i></span>
      <label class="fchip" style="display:inline-flex;align-items:center;gap:7px;cursor:pointer">
        <input type="checkbox" id="dlAudio" checked style="accent-color:var(--aqua);cursor:pointer"> 下载含声音</label></div></div>
  <div class="clips an" id="recGrid" style="--i:1"></div>
</section>

<!-- ============ 趋势 ============ -->
<section id="pg-trend" class="pg">
  <div class="ph an" style="--i:0"><div><h1>趋势</h1><p>一眼看懂小猫的喝水情况</p></div>
    <div class="seg-ctl" id="rangeCtl">
      <button class="on" onclick="setRange(7,this)">近 7 天</button>
      <button onclick="setRange(30,this)">近 30 天</button></div></div>
  <div class="grid">
    <div class="card sp12 an" style="--i:1"><div class="card-b">
      <div class="trend-hero" id="trendHero">
        <div class="trend-drop"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg></div>
        <div><div class="trend-today"><span id="trToday">–</span><small>次 · 今日</small></div>
          <div class="trend-status" id="trStatus">–</div>
          <div class="trend-sub" id="trSub"></div></div>
      </div></div></div>
    <div class="card sp6 an" style="--i:2"><div class="card-h">一天里什么时候爱喝水 <span class="note" id="hourPeak"></span></div>
      <div class="card-b"><div class="heatrow" id="hourHeat"></div>
      <div class="heataxis"><span>0</span><span>6</span><span>12</span><span>18</span><span>24 点</span></div></div></div>
    <div class="card sp6 an" style="--i:3"><div class="card-h">最近每天喝了多少 <span class="note" id="calNote"></span></div>
      <div class="card-b"><div class="calheat" id="calHeat"></div>
      <div class="callegend"><span>少</span><i class="cl" data-l="0"></i><i class="cl" data-l="1"></i><i class="cl" data-l="2"></i><i class="cl" data-l="3"></i><i class="cl" data-l="4"></i><span>多</span></div></div></div>
  </div>
</section>

<!-- ============ 实验室 ============ -->
<section id="pg-lab" class="pg">
  <div class="ph an" style="--i:0"><div><h1>实验室</h1><p>模型先判断 · 人工复核不确定项与纠错 · 用新标注继续训练</p></div>
    <div class="seg-ctl lab-tabs" id="labTabs" style="margin:0">
      <button class="on" onclick="setLab('label',this)">复核与纠错</button>
      <button onclick="setLab('train',this)">模型训练</button>
      <button onclick="setLab('models',this)">模型版本</button></div></div>

  <div class="lab-pane on an" id="lab-label" style="--i:1">
    <p class="lab-note">优先复核无结果、不确定、AI 与本地意见不同的片段，并抽查约 10% 的高置信结果。只有人工确认进入视频训练；部分样本固定留作验证，不参与拟合。</p>
    <div class="toolbar" id="labelFilters">
      <button class="fchip on" data-f="review" onclick="setLabelFilter('review',this)">待复核</button>
      <button class="fchip" data-f="all" onclick="setLabelFilter('all',this)">全部</button>
      <button class="fchip" data-f="yes" onclick="setLabelFilter('yes',this)">模型：喝水</button>
      <button class="fchip" data-f="no" onclick="setLabelFilter('no',this)">模型：没喝</button>
      <span style="margin-left:auto;color:var(--faint);font-size:12px" id="labCap"></span>
    </div>
    <div class="clips" id="labelClips"></div>
  </div>

  <div class="lab-pane" id="lab-train">
    <div class="krow">
      <div class="kbox"><div class="k">尚无标签的录像</div><div class="v"><span id="dsUn">–</span><small>段</small></div></div>
      <div class="kbox"><div class="k">人工确认</div><div class="v"><span id="dsNew">–</span><small>段</small></div></div>
      <div class="kbox"><div class="k">机器标签 · 待人工确认</div><div class="v"><span id="dsTr">–</span><small>段</small></div></div>
      <div class="kbox"><div class="k">人工 喝水 / 没喝（每类需 ≥4）</div><div class="v" id="dsBal">–</div></div>
    </div>
    <div class="card"><div class="card-b">
      <p class="lab-note">训练看动作的本地视频模型（s3d 冻结特征 + 分类头）。重点查看喝水召回；训练完成后不会自动生效，请到「模型版本」启用。</p>
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <button id="trainVideoBtn" class="btn" onclick="trainVideo()">训练视频模型</button>
        <label style="font-size:13px;color:var(--muted);cursor:pointer;display:inline-flex;align-items:center;gap:6px">
          <input type="checkbox" id="trainVideoRebuild" style="accent-color:var(--aqua)"> 修复缓存：重新处理全部录像（通常无需勾选）</label>
      </div>
      <div id="trainVideoProg" style="display:none;margin-top:16px"></div>
      <div class="t-status" id="trainVideoStatus"></div>
    </div></div>
  </div>

  <div class="lab-pane" id="lab-models">
    <div class="card" style="margin-bottom:18px"><div class="card-h">当前生效模型</div>
      <div class="card-b" id="activeBox"></div></div>
    <div id="modelList" class="mlist"></div>
  </div>
</section>

</main>

<!-- 灯箱 / 模态 / toast -->
<div class="lbox" id="lbox" onclick="if(event.target===this)closeLbox()"><div class="lbox-inner" id="lboxInner"></div></div>
<div class="modal" id="modal" onclick="if(event.target===this)closeModal()"><div class="modal-card" id="modalCard"></div></div>
<div id="toasts"></div>

<svg width="0" height="0" style="position:absolute"><defs>
  <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="var(--aqua)"/><stop offset="1" stop-color="var(--aqua2)"/></linearGradient>
</defs></svg>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const I={
  check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>',
  x:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  dl:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg>',
  play:'<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  clock:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  drop:'<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg>',
  info:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/></svg>',
  arrow:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  left:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg>',
  right:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg>'
};
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const reduceMotion=matchMedia('(prefers-reduced-motion:reduce)').matches;

/* ---------- 数字滚动 ---------- */
function countUp(el,to,dec){
  if(!el)return; to=+to||0; dec=dec||0;
  const from=parseFloat(el.dataset.v||'0')||0;
  el.dataset.v=to;
  if(reduceMotion||from===to){el.textContent=to.toFixed(dec);return;}
  const t0=performance.now(),dur=750;
  (function tick(t){
    const p=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-p,3);
    el.textContent=(from+(to-from)*e).toFixed(dec);
    if(p<1)requestAnimationFrame(tick);
  })(t0);
}

/* ---------- toast ---------- */
function toast(msg,type){
  type=type||'ok';
  const el=document.createElement('div');
  el.className='toast '+type;
  el.innerHTML=`<span class="ti">${type==='err'?I.x:type==='info'?I.info:I.check}</span><span>${esc(msg)}</span>`;
  $('#toasts').appendChild(el);
  setTimeout(()=>{el.classList.add('out');setTimeout(()=>el.remove(),320);},3200);
}

/* ---------- 模态 / 灯箱 ---------- */
function openModal(html){$('#modalCard').innerHTML=html;$('#modal').classList.add('show');}
function closeModal(){$('#modal').classList.remove('show');}
function openLbox(html){$('#lboxInner').innerHTML=html+
  `<button class="lbox-x" onclick="closeLbox()" aria-label="关闭">${I.x}</button>`;
  $('#lbox').classList.add('show');}
function closeLbox(){$('#lbox').classList.remove('show');$('#lboxInner').innerHTML='';}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){closeModal();closeLbox();return;}
  if($('#lbox').classList.contains('show')){
    if(e.key==='ArrowLeft')lboxStep(-1);
    if(e.key==='ArrowRight')lboxStep(1);
    return;
  }
  if(e.target.matches('input,textarea')||$('#modal').classList.contains('show'))return;
  const k={'1':'home','2':'disp','3':'records','4':'trend','5':'lab'}[e.key];
  if(k)go(k);
});

/* ---------- 路由（hash → 页面），懒加载各页数据 ---------- */
const PAGES=['home','disp','records','trend','lab'];
function go(p){location.hash='#/'+p;}
function show(p){
  if(!PAGES.includes(p))p='home';
  $$('.menu [data-pg]').forEach(b=>b.classList.toggle('on',b.dataset.pg===p));
  $$('.pg').forEach(s=>s.classList.toggle('on',s.id==='pg-'+p));
  const btn=$(`.menu [data-pg="${p}"]`),ind=$('#navInd');
  if(btn&&ind){ind.style.opacity='1';
    ind.style.transform=`translateY(${btn.offsetTop+(btn.offsetHeight-26)/2}px)`;}
  if(p==='home'){loadStats();loadDispenser();}
  if(p==='disp')loadDispenser();
  if(p==='records')loadRecords();
  if(p==='trend')renderTrend();
  if(p==='lab')loadLabelClips();
}
window.addEventListener('hashchange',()=>show(location.hash.replace('#/','')||'home'));
$$('.menu [data-pg]').forEach(b=>b.onclick=()=>go(b.dataset.pg));

/* ---------- 实时画面全屏 ---------- */
function openLive(){
  openLbox(`<img class="big" src="/stream.mjpg" alt="实时画面">
    <div class="lbox-bar"><span class="lt"><span class="live-dot"></span>实时画面 · LIVE</span></div>`);
}

/* ---------- 总览统计 ---------- */
function relTime(ts){if(!ts)return '—';const s=Date.now()/1000-ts;
  if(s<60)return '刚刚'; if(s<3600)return Math.floor(s/60)+' 分钟前';
  if(s<86400)return Math.floor(s/3600)+' 小时前'; return Math.floor(s/86400)+' 天前';}
async function loadStats(){
  try{
    const s=await (await fetch('/api/stats/today')).json();
    countUp($('#sideCount'),s.count); countUp($('#heroCount'),s.count);
    const last=s.times.length?s.times[s.times.length-1]:null;
    $('#heroLast').innerHTML=last?`<i></i>最近一次 ${esc(last.slice(0,5))}`:'<i></i>今日尚无确认记录';
    $('#heroLast').className='tag '+(last?'ok':'mute');
    renderTimeline(s.times);
    const w=await (await fetch('/api/stats/range?days=7')).json();
    const days=w.days||[],wt=days.reduce((a,d)=>a+d.count,0);
    countUp($('#msWeek'),wt); countUp($('#msAvg'),wt/7,1);
    const mx=Math.max(1,...days.map(d=>d.count)),wd='日一二三四五六';
    $('#spark').innerHTML=`<div class="sp-l">近 7 天走势</div>
      <div class="sp-bars">${days.map((d,i)=>
        `<div class="sb ${d.count?'':'zero'}" style="height:${Math.max(8,d.count/mx*100)}%;--i:${i}" title="${d.date} · ${d.count} 次"></div>`).join('')}</div>
      <div class="sp-days">${days.map(d=>{const dt=new Date(d.date);
        return `<span>${isNaN(dt)?'':wd[dt.getDay()]}</span>`;}).join('')}</div>`;
  }catch(e){}
}
function hms2frac(t){const a=t.split(':').map(Number);return (a[0]+(a[1]||0)/60+(a[2]||0)/3600)/24;}
function renderTimeline(times){
  const tl=$('#timeline'),now=new Date(),p=n=>String(n).padStart(2,'0');
  $('#tlNow').textContent=`现在 ${p(now.getHours())}:${p(now.getMinutes())}`;
  $('#homeDate').textContent=`${now.getMonth()+1} 月 ${now.getDate()} 日 · ${'日一二三四五六'.split('')[now.getDay()].replace(/^/,'星期')}`;
  const nowFrac=(now.getHours()+now.getMinutes()/60)/24;
  const ticks=[0,3,6,9,12,15,18,21,24].map(h=>
    `<div class="tl-tick" style="left:${(h/24*100).toFixed(2)}%"><i></i><span>${h}:00</span></div>`).join('');
  const dots=(times||[]).map((t,i)=>`<div class="tl-dot" style="left:${(hms2frac(t)*100).toFixed(2)}%;--i:${i}" title="${esc(t)}"></div>`).join('');
  const nowLine=`<div class="tl-now" style="left:${(nowFrac*100).toFixed(2)}%"></div>`;
  tl.innerHTML=(times&&times.length)?`<div class="tl-track"></div>${ticks}${nowLine}${dots}`
    :`<div class="tl-track"></div>${ticks}${nowLine}<div class="tl-empty">今天还没记录到喝水 🐾</div>`;
}

/* ---------- 饮水机组件：霍曼三代 Pro 的二次元 Q 版 ----------
   还原实物：白色圆筒机身 + 顶部透明水盆（涟漪）+ 中央橙色出水舌（喷泉）+
   正面竖长水位窗（映射剩余水量）+ 底部 LED；加上眨眼表情 / 低水位担忧脸 / 星星点缀。 */
const tankFills={};   // uid → 当前水位（用于加水/变化时的过渡动画）
function tankSvg(uid,size){
  const h=Math.round(size*260/220);
  const winWave=y=>{let d=`M85 ${y}`;for(let x=85;x<157;x+=12)d+=` q6 -3.5 12 0`;
    return d+` L157 215 L85 215 Z`;};
  return `<svg id="tk-${uid}" class="tank" width="${size}" height="${h}" viewBox="0 0 220 260">
  <defs>
    <clipPath id="tkw-${uid}"><rect x="97" y="124" width="26" height="74" rx="12"/></clipPath>
    <linearGradient id="tkg-${uid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" style="stop-color:var(--tk1)"/><stop offset="1" style="stop-color:var(--tk2)"/></linearGradient>
    <linearGradient id="tkb-${uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#ffffff"/><stop offset=".55" stop-color="#f3f7fb"/><stop offset="1" stop-color="#d9e4ee"/></linearGradient>
  </defs>
  <ellipse cx="110" cy="245" rx="74" ry="9" fill="rgba(25,60,90,.10)"/>
  <!-- 机身 -->
  <path d="M52 82 h116 v132 a26 26 0 0 1 -26 26 h-64 a26 26 0 0 1 -26 -26 Z"
    fill="url(#tkb-${uid})" stroke="rgba(120,150,175,.4)" stroke-width="1.5"/>
  <!-- 顶部透明水盆 + 水面涟漪 -->
  <path d="M44 64 v12 a66 18 0 0 0 132 0 v-12" fill="rgba(185,222,244,.25)" stroke="rgba(150,195,225,.5)" stroke-width="1.5"/>
  <ellipse cx="110" cy="64" rx="66" ry="17" fill="rgba(208,236,250,.55)" stroke="rgba(160,205,235,.75)" stroke-width="1.5"/>
  <ellipse class="basin-w" cx="110" cy="65" rx="56" ry="13" fill="url(#tkg-${uid})" opacity=".7"/>
  <ellipse class="rip" cx="110" cy="65" rx="42" ry="9.5" fill="none" stroke="#fff" stroke-width="1.5"/>
  <ellipse class="rip r2" cx="110" cy="65" rx="42" ry="9.5" fill="none" stroke="#fff" stroke-width="1.5"/>
  <!-- 喷泉水流 + 水珠 -->
  <g class="fount">
    <path class="jet" d="M103 55 Q86 57 80 66" fill="none" stroke="var(--tk1)" stroke-width="3" stroke-linecap="round" stroke-dasharray="4 6"/>
    <path class="jet" d="M117 55 Q134 57 140 66" fill="none" stroke="var(--tk1)" stroke-width="3" stroke-linecap="round" stroke-dasharray="4 6" style="animation-delay:.3s"/>
    <circle class="jd" cx="84" cy="60" r="2.3" fill="var(--tk1)"/>
    <circle class="jd d2" cx="136" cy="60" r="2.3" fill="var(--tk1)"/>
  </g>
  <!-- 橙色出水舌（招牌小舌头） -->
  <path d="M99 58 q11 -17 22 0 q-11 8 -22 0" fill="#ff9440" stroke="#ee7d26" stroke-width="1.2"/>
  <circle cx="106" cy="49.5" r="2" fill="#fff" opacity=".75"/>
  <circle class="pourdrop" cx="110" cy="34" r="5" fill="var(--tk1)"/>
  <!-- 二次元表情（正常眨眼 / 低水位担忧 + 汗滴） -->
  <g class="face f-normal">
    <circle class="eye" cx="86" cy="106" r="3.4" fill="#41535f"/><circle class="eye" cx="134" cy="106" r="3.4" fill="#41535f"/>
    <circle cx="87.3" cy="104.8" r="1.1" fill="#fff"/><circle cx="135.3" cy="104.8" r="1.1" fill="#fff"/>
    <path d="M104 110 q3 3.5 6 0 q3 3.5 6 0" fill="none" stroke="#41535f" stroke-width="1.8" stroke-linecap="round"/>
  </g>
  <g class="face f-low">
    <path d="M80 102 l8 4 -8 4 M140 102 l-8 4 8 4" fill="none" stroke="#41535f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M104 112 q6 -5 12 0" fill="none" stroke="#41535f" stroke-width="1.8" stroke-linecap="round"/>
    <path class="sweat" d="M148 92 q4.5 6 0 9 q-4.5 -3 0 -9" fill="var(--tk1)" opacity=".85"/>
  </g>
  <ellipse cx="72" cy="114" rx="6" ry="3.4" fill="rgba(255,130,140,.32)"/>
  <ellipse cx="148" cy="114" rx="6" ry="3.4" fill="rgba(255,130,140,.32)"/>
  <!-- 水位窗（剩余水量就画在这） -->
  <rect x="95" y="122" width="30" height="78" rx="14" fill="rgba(40,70,95,.10)" stroke="rgba(120,150,175,.5)" stroke-width="1.5"/>
  <g clip-path="url(#tkw-${uid})"><g class="lvl" style="transform:translateY(74px)">
    <path class="wv" d="${winWave(124)}" fill="url(#tkg-${uid})"/>
    <path class="wv wv2" d="${winWave(126)}" fill="url(#tkg-${uid})" opacity=".55"/>
    <circle class="bub" cx="106" cy="192" r="1.8" fill="#fff"/>
    <circle class="bub" cx="115" cy="196" r="1.4" fill="#fff" style="animation-delay:1.5s"/>
  </g></g>
  <circle class="led" cx="110" cy="214" r="4" fill="var(--aqua)"/>
  <path class="spk" d="M40 44 l1.4 3.8 3.8 1.4 -3.8 1.4 -1.4 3.8 -1.4 -3.8 -3.8 -1.4 3.8 -1.4 Z" fill="var(--tk1)"/>
  <path class="spk s2" d="M182 36 l1.2 3.2 3.2 1.2 -3.2 1.2 -1.2 3.2 -1.2 -3.2 -3.2 -1.2 3.2 -1.2 Z" fill="var(--tk1)"/>
  </svg>`;
}
/* 设置水位：0-1。水位窗 1s 弹性过渡；state: ''|'warn'|'low' 控制水色/表情 */
function setTank(uid,fill,state){
  const svg=$('#tk-'+uid); if(!svg)return;
  fill=Math.max(0,Math.min(1,fill||0));
  const prev=tankFills[uid]; tankFills[uid]=fill;
  const y=v=>74*(1-v);
  const lvl=svg.querySelector('.lvl');
  svg.classList.toggle('warn',state==='warn');
  svg.classList.toggle('low',state==='low');
  svg.classList.toggle('empty',fill<=0);
  const box=svg.closest('.tankbox');
  if(box){box.classList.toggle('glow-low',state==='low');box.classList.toggle('glow-ok',state!=='low'&&fill>0);}
  if(prev===undefined&&!reduceMotion){    // 首次：从空升到目标水位
    lvl.style.transition='none';lvl.style.transform=`translateY(${y(0)}px)`;
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      lvl.style.transition='';lvl.style.transform=`translateY(${y(fill)}px)`;}));
  }else{lvl.style.transform=`translateY(${y(fill)}px)`;}
}
function pourTank(uid){
  const svg=$('#tk-'+uid); if(!svg||reduceMotion)return;
  svg.classList.remove('pouring'); void svg.offsetWidth; svg.classList.add('pouring');
  setTimeout(()=>svg.classList.remove('pouring'),2200);
}

/* ---------- 饮水机 ---------- */
let dispData=null, dispLastMl=2000;
function dispState(d){return !d.has_refill?'':(d.need_water?'low':(d.remaining_pct!=null&&d.remaining_pct<.4?'warn':''));}
async function loadDispenser(){
  try{const d=await (await fetch('/api/dispenser')).json();
    if(d&&!d.detail){dispData=d;if(d.last_refill_ml)dispLastMl=d.last_refill_ml;
      renderDispMini(d);renderDisp(d);
      const nav=$('#navDisp');
      const dot=nav.querySelector('.adot');
      if(d.need_water||d.need_filter){if(!dot)nav.insertAdjacentHTML('beforeend','<span class="adot"></span>');}
      else if(dot)dot.remove();
    }}catch(e){}
}
/* 总览速览小水箱 */
function renderDispMini(d){
  const box=$('#dmBody'),note=$('#dmNote'); if(!box)return;
  const st=dispState(d);
  note.textContent=d.need_water?'💧 该加水了':(d.need_filter?'🧽 该换滤芯':'');
  const pct=d.remaining_pct==null?null:Math.round(d.remaining_pct*100);
  const fsub=!d.has_filter?'':(d.need_filter?'滤芯超期了，该换啦 🧽'
    :`滤芯还能用 ${Math.max(0,Math.round(d.filter_days_left||0))} 天`);
  const bubCls='tank-bub mini'+(st==='low'?' low':(st==='warn'?' warn':''));
  box.innerHTML=`<div class="tankbox">${tankSvg('mini',108)}
      ${d.has_refill?`<div class="${bubCls}"><span id="dmPct">0</span>%</div>`
        :'<div class="tank-bub mini">?</div>'}</div>
    <div class="dmini-info">
      <div class="dm-l">剩余水量</div>
      <div class="dm-v">${d.has_refill?`约 ${d.remaining_ml}<small> ml</small>`:'未记录蓄水'}</div>
      ${fsub?`<div class="dm-sub">${fsub}</div>`:''}
      <div class="go">去看看 ${I.arrow}</div>
    </div>`;
  setTank('mini',d.has_refill?(d.remaining_pct||0):0,st);
  if(pct!=null)countUp($('#dmPct'),pct);
  countUp($('#msMl'),d.today_ml||0);
}
/* 饮水机页 */
function renderDisp(d){
  const box=$('#dispBody'); if(!box)return;
  const st=dispState(d);
  if(!d.has_refill){
    box.innerHTML=`<div class="disp-hero">
      <div class="tankbox">${tankSvg('main',230)}
        <div class="tank-bub"><b>?</b><small>还没记录蓄水</small></div></div>
      <div class="disp-main">
        <div style="font-size:19px;font-weight:800;margin-bottom:8px">还没记录过蓄水 🐱</div>
        <p class="lab-note" style="max-width:420px">点「加满水」填这次加了多少毫升——之后喵喵每来喝一次，
        水箱就会跟着变少；快没水、该换滤芯时，这里都会提醒你。</p>
        <div class="disp-actions">
          <button class="btn" onclick="dispRefill()">${I.drop} 加满水</button>
          <button class="btn ghost" onclick="dispFilter()">换了滤芯</button>
        </div></div></div>`;
    setTank('main',0,'');
    return;
  }
  const pct=Math.round((d.remaining_pct||0)*100);
  const fdays=d.filter_days_left,fBad=d.need_filter;
  const cyc=d.filter_cycle_days||30;
  const fpct=(!d.has_filter||fdays==null)?0:Math.max(0,Math.min(1,fdays/cyc));
  const C=2*Math.PI*26;
  const bubCls='tank-bub'+(st==='low'?' low':(st==='warn'?' warn':''));
  const bubTxt=st==='low'?'快没水啦，加水喵！':(st==='warn'?'有点渴了…':'水量充足~');
  box.innerHTML=`<div class="disp-hero">
    <div class="tankbox">${tankSvg('main',230)}
      <div class="${bubCls}"><b><span id="dpPct">0</span>%</b><small>${bubTxt}</small></div></div>
    <div class="disp-main">
      <div class="disp-remain-line">
        <span class="big"><span id="dpMl">0</span><small> / ${d.last_refill_ml} ml</small></span>
        <span class="calib-badge ${d.calibrated?'':'default'}">${d.calibrated?'✓ 已自校准':'默认估计'}</span>
      </div>
      ${d.need_water?`<div class="disp-alert">${I.drop} 水不多了，快给喵喵加水！</div>`
        :(fBad?`<div class="disp-alert amber">🧽 滤芯超期了，该换啦</div>`:'')}
      <div class="disp-cells">
        <div class="dcell"><div class="k">上次蓄水</div><div class="v" style="font-size:17px">${relTime(d.last_refill_ts)}</div>
          <div class="sub">加了 ${d.last_refill_ml} ml</div></div>
        <div class="dcell ${fBad?'warn':''}"><div class="k">滤芯${fBad?'（该换了）':'剩余'}</div>
          <div class="ringwrap">
            <svg class="ring ${fBad?'bad':''}" width="60" height="60" viewBox="0 0 60 60">
              <circle class="bgc" cx="30" cy="30" r="26" stroke-width="6"/>
              <circle class="fgc" cx="30" cy="30" r="26" stroke-width="6"
                stroke-dasharray="${C.toFixed(1)}" stroke-dashoffset="${(C*(1-fpct)).toFixed(1)}"/></svg>
            <div class="v">${!d.has_filter?'—':(fBad?'超期':Math.round(fdays))}<small>${d.has_filter&&!fBad?'天':''}</small></div>
          </div><div class="sub">周期 ${cyc} 天</div></div>
        <div class="dcell"><div class="k">今日约喝</div><div class="v"><span id="dpToday">0</span><small>ml</small></div>
          <div class="sub">按每次 ${d.ml_per_drink} ml 估算</div></div>
        <div class="dcell"><div class="k">蓄水以来</div><div class="v">${d.drinks_since_refill}<small>次</small></div>
          <div class="sub">每次约 ${d.ml_per_drink} ml（${d.calibrated?'自校准':'默认值'}）</div></div>
      </div>
      <div class="disp-actions">
        <button class="btn" onclick="dispRefill()">${I.drop} 加满水</button>
        <button class="btn ghost" onclick="dispFilter()">换了滤芯</button>
        <button class="btn ghost" onclick="dispCycle()">滤芯周期</button>
      </div>
    </div></div>`;
  setTank('main',d.remaining_pct||0,st);
  countUp($('#dpPct'),pct); countUp($('#dpMl'),d.remaining_ml); countUp($('#dpToday'),d.today_ml);
}
/* 加水模态：预设芯片 + 滑杆 + 数字输入 */
function dispRefill(){
  const presets=[500,1000,1500,2000,3000];
  openModal(`<h3>给喵喵加水 💧</h3><p class="msub">填这次加满后的总水量（毫升）——之后按喝水次数反推剩余。</p>
    <div class="presets">${presets.map(v=>`<button class="${v===dispLastMl?'on':''}" onclick="rfSet(${v})">${v} ml</button>`).join('')}</div>
    <div class="mlrow">
      <input type="range" id="rfRange" min="200" max="5000" step="50" value="${dispLastMl}" oninput="rfSync(this.value)">
      <span class="mlnum"><input type="number" id="rfNum" value="${dispLastMl}" min="1" step="50"
        oninput="rfSync(this.value,true)"><span>ml</span></span></div>
    <div class="modal-acts">
      <button class="btn ghost" onclick="closeModal()">取消</button>
      <button class="btn" onclick="rfGo()">确认加水</button></div>`);
}
function rfSet(v){$('#rfRange').value=v;$('#rfNum').value=v;
  $$('#modalCard .presets button').forEach(b=>b.classList.toggle('on',b.textContent.startsWith(v+' ')));}
function rfSync(v,fromNum){if(!fromNum)$('#rfNum').value=v;else $('#rfRange').value=v;
  $$('#modalCard .presets button').forEach(b=>b.classList.toggle('on',b.textContent.startsWith(v+' ')));}
async function rfGo(){
  const ml=parseFloat($('#rfNum').value);
  if(!(ml>0)){toast('请输入大于 0 的毫升数','err');return;}
  closeModal();
  try{
    const d=await (await fetch('/api/dispenser/refill',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({ml})})).json();
    if(d&&!d.detail){dispData=d;dispLastMl=ml;
      renderDisp(d);renderDispMini(d);pourTank('main');pourTank('mini');
      toast(`已加水 ${ml} ml，水箱满啦`);loadDispenser();}
    else toast(d.detail||'加水失败','err');
  }catch(e){toast('加水失败：'+e,'err');}
}
function dispFilter(){
  openModal(`<h3>换了滤芯？🧽</h3><p class="msub">确认后将从今天重新倒计时（周期 ${dispData?dispData.filter_cycle_days:30} 天）。</p>
    <div class="modal-acts">
      <button class="btn ghost" onclick="closeModal()">取消</button>
      <button class="btn" onclick="fcGo()">确认已更换</button></div>`);
}
async function fcGo(){
  closeModal();
  try{const d=await (await fetch('/api/dispenser/filter',{method:'POST'})).json();
    if(d&&!d.detail){dispData=d;renderDisp(d);renderDispMini(d);toast('滤芯倒计时已重置');}
  }catch(e){toast('操作失败','err');}
}
function dispCycle(){
  const cur=dispData?dispData.filter_cycle_days:30;
  openModal(`<h3>滤芯更换周期</h3><p class="msub">多少天提醒换一次滤芯？</p>
    <div class="stepper">
      <button onclick="cyStep(-5)">−</button>
      <div class="sv"><span id="cyV">${cur}</span> <small>天</small></div>
      <button onclick="cyStep(5)">＋</button></div>
    <div class="modal-acts">
      <button class="btn ghost" onclick="closeModal()">取消</button>
      <button class="btn" onclick="cyGo()">保存</button></div>`);
}
function cyStep(d){const el=$('#cyV');el.textContent=Math.max(5,(parseInt(el.textContent,10)||30)+d);}
async function cyGo(){
  const days=parseInt($('#cyV').textContent,10);
  closeModal();
  try{const d=await (await fetch('/api/dispenser/config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({filter_cycle_days:days})})).json();
    if(d&&!d.detail){dispData=d;renderDisp(d);renderDispMini(d);toast(`滤芯周期已设为 ${days} 天`);}
  }catch(e){toast('保存失败','err');}
}

/* ---------- 视频数据（记录页 + 标注台共用一份 /api/clips） ---------- */
let clipsData=null;
async function fetchClips(){clipsData=await (await fetch('/api/clips')).json();return clipsData;}
function clipTime(name){const m=String(name).match(/clip_(\\d+)/);return m?new Date(parseInt(m[1],10)):null;}
function dayLabel(d){
  if(!d)return '未知时间';
  const now=new Date(),a=new Date(now.getFullYear(),now.getMonth(),now.getDate());
  const b=new Date(d.getFullYear(),d.getMonth(),d.getDate());
  const diff=Math.round((a-b)/86400000);
  if(diff===0)return '今天'; if(diff===1)return '昨天';
  return `${d.getMonth()+1} 月 ${d.getDate()} 日`;
}
function dayKey(d){return d?`${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`:'?';}
function hourKey(d){return d?dayKey(d)+'-'+d.getHours():'?';}
function hourLabel(d){if(!d)return '未知时间';const p=n=>String(n).padStart(2,'0');
  return `${dayLabel(d)} ${p(d.getHours())}:00–${p((d.getHours()+1)%24)}:00`;}
function hhmm(d){const p=n=>String(n).padStart(2,'0');return d?`${p(d.getHours())}:${p(d.getMinutes())}`:'';}
function hhmmss(d){const p=n=>String(n).padStart(2,'0');return d?`${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`:'';}
function dlUrl(name){const e=encodeURIComponent(name);
  return ($('#dlAudio')&&$('#dlAudio').checked)?('/clips/'+e+'?download=1'):('/clips/'+e+'?audio=0&download=1');}
/* 灯箱播放：支持在当前列表里 ←/→ 或点箭头切上一段/下一段 */
let lboxList=[],lboxIdx=-1;
function openClip(name,list){
  lboxList=(list&&list.length)?list:[name];
  lboxIdx=Math.max(0,lboxList.indexOf(name));
  renderLboxClip();
}
function renderLboxClip(){
  const name=lboxList[lboxIdx];
  const t=clipTime(name),dur=(clipsData&&clipsData.durations||{})[name];
  const lab=(clipsData&&clipsData.labels||{})[name];
  const nav=lboxList.length>1?`
    <button class="lbox-nav prev" onclick="lboxStep(-1)" ${lboxIdx<=0?'disabled':''} aria-label="上一段">${I.left}</button>
    <button class="lbox-nav next" onclick="lboxStep(1)" ${lboxIdx>=lboxList.length-1?'disabled':''} aria-label="下一段">${I.right}</button>`:'';
  const pos=lboxList.length>1?` · ${lboxIdx+1}/${lboxList.length}`:'';
  openLbox(`<video src="/clips/${encodeURIComponent(name)}" controls autoplay playsinline></video>${nav}
    <div class="lbox-bar"><span class="lt">${I.clock} ${t?dayLabel(t)+' '+hhmmss(t):esc(name)}${dur?` · ${dur.toFixed(1)} 秒`:''}${pos}</span>
      <span class="ltags">${statusTag(lab)}
      <a href="/clips/${encodeURIComponent(name)}" download onclick='this.href=dlUrl(${JSON.stringify(name)})'>${I.dl} 下载</a></span></div>`);
}
function lboxStep(d){const i=lboxIdx+d;if(i<0||i>=lboxList.length)return;lboxIdx=i;renderLboxClip();}

/* ---------- 喝水记录页：只展示已确认「真喝水」的段 ---------- */
let recPage=1, recObserver=null;
const REC_PAGE=12;
function skeletons(n){let h='';for(let i=0;i<n;i++)h+='<div class="skel"></div>';return h;}
async function loadRecords(){
  recPage=1;
  $('#recGrid').innerHTML=skeletons(8);
  await fetchClips();
  renderRecords();
}
function recItems(){
  const lab=clipsData.labels||{};
  return clipsData.clips.filter(n=>lab[n]===true);
}
function recCard(n,i){
  const dur=(clipsData.durations||{})[n],t=clipTime(n),en=esc(n),jn=JSON.stringify(n);
  return `<div class="clip" data-name="${en}" style="--i:${Math.min(i,10)}">
    <div class="thumb">
      <video class="inline-clip" src="/clips/${encodeURIComponent(n)}" poster="/clips/${encodeURIComponent(n)}/thumb.jpg" controls preload="metadata" playsinline></video>
      ${dur?`<span class="dur">${dur.toFixed(1)} 秒</span>`:''}</div>
    <div class="cmeta"><div class="row">
      <span class="ctime">${I.clock}${t?hhmm(t):''}</span>
      <a class="dlbtn" href="/clips/${encodeURIComponent(n)}" download onclick='this.href=dlUrl(${jn})'>${I.dl}下载</a>
    </div></div></div>`;
}
function renderRecords(){
  if(!clipsData)return;
  const box=$('#recGrid'),items=recItems();
  const cnt=$('#recCount');
  cnt.style.display='inline-flex';cnt.innerHTML=`<i></i>共 ${items.length} 段确认喝水`;
  if(!items.length){
    box.innerHTML=`<div class="empty" style="grid-column:1/-1"><span class="eico">🐟</span>
      还没有确认喝水的回放～<br>喵喵来水碗喝水会自动录像，AI 判定「真喝水」后就会出现在这里。</div>`;
    return;
  }
  const dayCounts={}; items.forEach(n=>{const k=dayKey(clipTime(n));dayCounts[k]=(dayCounts[k]||0)+1;});
  const shown=items.slice(0,recPage*REC_PAGE);
  let html='',lastDay=null;
  shown.forEach((n,i)=>{const dt=clipTime(n),k=dayKey(dt);
    if(k!==lastDay){lastDay=k;
      html+=`<div class="grp-head"><span class="grp-dot"></span>${dayLabel(dt)}<span class="grp-n">${dayCounts[k]} 次</span></div>`;}
    html+=recCard(n,i%REC_PAGE);});
  const remaining=items.length-shown.length;
  if(remaining>0)html+=`<div class="more" id="recMore" onclick="recPage++;renderRecords()">下滑加载更多 · 还有 ${remaining} 段</div>`;
  box.innerHTML=html;
  if(recObserver)recObserver.disconnect();
  const s=$('#recMore');
  if(s&&'IntersectionObserver' in window){
    recObserver=new IntersectionObserver(es=>{if(es[0].isIntersecting){recPage++;renderRecords();}},{rootMargin:'300px'});
    recObserver.observe(s);
  }
}
/* 卡片高级交互：3D 倾斜 + 跟手光晕（事件委托，绑一次）。 */
function bindCardFx(box){
  if(!box||box.dataset.fx)return; box.dataset.fx='1';
  if(!reduceMotion){
    box.addEventListener('pointermove',e=>{
      const card=e.target.closest('.clip'); if(!card)return;
      const r=card.getBoundingClientRect();
      const px=(e.clientX-r.left)/r.width, py=(e.clientY-r.top)/r.height;
      card.style.setProperty('--rx',((px-.5)*4.5).toFixed(2));
      card.style.setProperty('--ry',((.5-py)*4.5).toFixed(2));
      card.style.setProperty('--mx',(px*100).toFixed(1)+'%');
      card.style.setProperty('--my',(py*100).toFixed(1)+'%');
    });
  }
  box.addEventListener('pointerout',e=>{
    const card=e.target.closest('.clip');
    if(card&&!(e.relatedTarget&&card.contains(e.relatedTarget))){
      card.style.setProperty('--rx',0);card.style.setProperty('--ry',0);
    }
  });
}

/* ---------- 趋势 ---------- */
let trendDays=7;
function heatColor(v,max){const a=v<=0?0.07:0.22+0.78*(v/Math.max(1,max));return `rgba(24,166,214,${a.toFixed(3)})`;}
async function renderTrend(){
  const r=await (await fetch('/api/stats/trend?days='+trendDays)).json();
  const pts=r.days||[],vals=pts.map(p=>p.count);
  const total=r.total??vals.reduce((a,b)=>a+b,0),days=pts.length;
  const avg=days?total/days:0,today=days?vals[vals.length-1]:0;
  const hero=$('#trendHero');
  let st,cls;
  if(today===0){st='今日尚无确认记录 🐾';cls='none';}
  else if(avg>0&&today>=avg*1.15){st='喝得挺积极 🐱';cls='good';}
  else if(avg>0&&today<=avg*0.6){st='今天偏少，多留意 💧';cls='low';}
  else{st='喝水正常 👍';cls='ok';}
  hero.className='trend-hero '+cls;
  countUp($('#trToday'),today);
  $('#trStatus').textContent=st;
  const prev=r.prev_total||0;
  const deltaTxt=prev?`　·　环比 ${total>=prev?'▲':'▼'} ${Math.round(Math.abs(total-prev)/prev*100)}%`:'';
  $('#trSub').innerHTML=`日均 <b>${avg.toFixed(1)}</b> 次　·　近 ${days} 天共 <b>${total}</b> 次　·　活跃 <b>${r.active_days??0}/${days}</b> 天${deltaTxt}`;
  const hourly=r.hourly||[],hmax=Math.max(1,...hourly);
  $('#hourHeat').innerHTML=hourly.map((v,h)=>
    `<div class="hcell" style="background:${heatColor(v,hmax)};--i:${h}" title="${h}:00–${(h+1)%24}:00 · ${v} 次"></div>`).join('');
  const pk=Math.max(0,...hourly);
  $('#hourPeak').textContent=pk>0?`最爱 ${hourly.indexOf(pk)}:00 前后`:'';
  const cmax=Math.max(1,...vals);
  $('#calHeat').innerHTML=pts.map((p,i)=>
    `<div class="ccell" style="background:${heatColor(p.count,cmax)};--i:${i}" title="${p.date} · ${p.count} 次"><span>${p.count||''}</span></div>`).join('');
  $('#calNote').textContent=`单日最多 ${cmax} 次`;
  $$('#pg-trend .callegend .cl').forEach(el=>{el.style.background=heatColor(+el.dataset.l,4);});
}
function setRange(d,btn){trendDays=d;for(const b of $('#rangeCtl').children)b.classList.toggle('on',b===btn);renderTrend();}

/* ---------- 标注工作台 ---------- */
let labelFilter='review', labelPage=1;
const LAB_PAGE=12;
const REVIEW_LOW=.30, REVIEW_HIGH=.70;
function setLab(t,btn){
  for(const b of $('#labTabs').children)b.classList.toggle('on',b===btn);
  $$('.lab-pane').forEach(p=>p.classList.toggle('on',p.id==='lab-'+t));
  if(t==='label')loadLabelClips();
  if(t==='train'){pollTrain();pollTrainVideo();}
  if(t==='models')pollTrain();
}
function statusTag(v){return v===true?`<span class="tag ok"><i></i>真喝水</span>`
  :v===false?`<span class="tag bad"><i></i>没喝</span>`:`<span class="tag mute"><i></i>未人工确认</span>`;}
function modelDetail(n){
  const p=(clipsData.prediction_details||{})[n];
  if(p)return p;
  const m=(clipsData.meta||{})[n];
  if(m&&m.source!=='human'&&m.confidence!=null)return {
    drinking:!!m.is_drinking,
    confidence:m.is_drinking?m.confidence:1-m.confidence,
    by:m.source
  };
  return null;
}
function reviewReason(n){
  const m=(clipsData.meta||{})[n];
  if(m&&m.source==='human')return '';
  const p=modelDetail(n);
  if(!p)return '模型未判断';
  if(m&&m.source==='ai'&&!!m.is_drinking!==!!p.drinking)return 'AI 与本地模型意见不同';
  if(p.confidence!=null&&p.confidence>=REVIEW_LOW&&p.confidence<=REVIEW_HIGH)return '模型不确定';
  // Stable 10% audit catches confidently wrong predictions, including negatives.
  let h=2166136261;for(const c of n)h=Math.imul(h^c.charCodeAt(0),16777619);
  return (h>>>0)%10===0?'高置信结果抽查':'';
}
function needsReview(n){return !!reviewReason(n);}
function modelTag(n){
  const p=modelDetail(n);
  if(!p)return `<span class="tag mute"><i></i>模型未判断</span>`;
  const pct=p.confidence==null?'':` ${Math.round(p.confidence*100)}%`;
  const uncertain=needsReview(n)?' · '+reviewReason(n):'';
  return `<span class="mpred ${p.drinking?'y':'n'}" title="${esc(p.by||'模型')}">模型：${p.drinking?'喝水':'没喝'}${pct}${uncertain}</span>`;
}
async function loadLabelClips(){
  labelPage=1;
  $('#labelClips').innerHTML=skeletons(8);
  await fetchClips();
  const cap=clipsData.max_clips||1000;
  $('#labCap').textContent=`最多保留 ${cap} 段 · 超量只删最旧的「没喝」`;
  updateLabelChipCounts();
  renderLabelClips();
}
function updateLabelChipCounts(){
  let a=0,r=0,y=0,no=0;
  clipsData.clips.forEach(c=>{const p=modelDetail(c);a++;if(needsReview(c))r++;if(p){p.drinking?y++:no++;}});
  const map={all:`全部 ${a}`,review:`待复核 ${r}`,yes:`模型：喝水 ${y}`,no:`模型：没喝 ${no}`};
  $$('#labelFilters .fchip').forEach(b=>{if(map[b.dataset.f])b.textContent=map[b.dataset.f];});
}
function labelItems(){
  return clipsData.clips.filter(n=>{const p=modelDetail(n);
    return labelFilter==='all'||(labelFilter==='review'&&needsReview(n))||
      (labelFilter==='yes'&&p&&p.drinking)||(labelFilter==='no'&&p&&!p.drinking);});
}
function labelCard(n,i){
  const lab=clipsData.labels||{},dur=clipsData.durations||{};
  const v=lab[n],d=dur[n],en=esc(n),jn=JSON.stringify(n),t=clipTime(n);
  return `<div class="clip review-card" data-name="${en}">
    <div class="thumb">
      <video class="inline-clip" src="/clips/${encodeURIComponent(n)}" poster="/clips/${encodeURIComponent(n)}/thumb.jpg" controls preload="metadata" playsinline></video>
      ${d?`<span class="dur">${d.toFixed(1)} 秒</span>`:''}</div>
    <div class="review-side">
      <div class="review-head"><div class="review-title">${t?dayLabel(t)+' '+hhmmss(t):'候选视频'}</div>${statusTag(v)}</div>
      <div class="row">${modelTag(n)}</div>
      <div class="seg2">
        <button class="yes ${v===true?'on':''}" onclick='fb(${jn},true)'>${I.check}喝了</button>
        <button class="no ${v===false?'on':''}" onclick='fb(${jn},false)'>${I.x}没喝</button>
      </div>
      <div class="review-foot"><span class="fname">${en}</span><a class="dlbtn" href="/clips/${encodeURIComponent(n)}" download onclick='this.href=dlUrl(${jn})'>${I.dl}下载</a></div>
    </div></div>`;
}
function renderLabelClips(){
  const box=$('#labelClips'); if(!box||!clipsData)return;
  if(!clipsData.clips.length){box.innerHTML=`<div class="empty" style="grid-column:1/-1"><span class="eico">🎬</span>还没有视频可标注。接上摄像头跑 <code>python -m catcam</code>。</div>`;return;}
  const items=labelItems();
  if(!items.length){box.innerHTML='<div class="empty" style="grid-column:1/-1">这个筛选下没有视频</div>';return;}
  const shown=items.slice(0,labelPage*LAB_PAGE);
  let html=''; shown.forEach((n,i)=>{html+=labelCard(n,i);});
  const remaining=items.length-shown.length;
  if(remaining>0)html+=`<div class="more" onclick="labelPage++;renderLabelClips()">加载更多 · 还有 ${remaining} 段</div>`;
  box.innerHTML=html;
}
function setLabelFilter(f,btn){labelFilter=f;labelPage=1;
  for(const b of $('#labelFilters').querySelectorAll('.fchip'))b.classList.toggle('on',b===btn);renderLabelClips();}
async function fb(clip,is){
  await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip,is_drinking:is})});
  if(clipsData){clipsData.labels[clip]=is;
    if(clipsData.meta)clipsData.meta[clip]={is_drinking:is,source:'human',confidence:null,reason:null};
    // 局部更新那张卡（限定 #labelClips，避免误伤其他视图）
    const card=$(`#labelClips .clip[data-name="${CSS.escape(clip)}"]`);
    if(card){const stEl=card.querySelector('.tag');if(stEl)stEl.outerHTML=statusTag(is);
      const y=card.querySelector('.seg2 .yes'),no=card.querySelector('.seg2 .no');
      if(y)y.classList.toggle('on',is===true);if(no)no.classList.toggle('on',is===false);
      if(labelFilter!=='all'){const p=modelDetail(clip);const keep=(labelFilter==='yes'&&p&&p.drinking)||(labelFilter==='no'&&p&&!p.drinking);
        if(!keep){card.style.transition='opacity .3s,transform .3s';card.style.opacity='0';
          card.style.transform='scale(.94)';setTimeout(renderLabelClips,300);}}
      else{const next=card.nextElementSibling;if(next)next.scrollIntoView({behavior:'smooth',block:'start'});}
    }}
  updateLabelChipCounts();
  toast(is?'已标注：真喝水 💧':'已标注：没喝');
  loadStats();
}

/* ---------- 模型训练 / 版本 ---------- */
function fmtAcc(a){return (typeof a==='number')?(a*100).toFixed(1)+'%':'—';}
function fmtPct(a){return (typeof a==='number')?(a*100).toFixed(0)+'%':'—';}
function trainingEvidence(e){
  if(!e)return '';
  const c=e.confusion,p=e.performance,cmp=e.comparison;
  let html=c?`<p>验证结果：漏掉 <b>${c.fn}</b> 段喝水，误报 <b>${c.fp}</b> 段；正确识别 ${c.tp+c.tn} 段。</p>`:'';
  if(cmp&&cmp.status==='compared'&&cmp.misses_reduced!=null){
    const delta=(n,label)=>n>0?`${label}减少 ${n} 段`:n<0?`${label}增加 ${-n} 段`:`${label}不变`;
    html+=`<p>同一批视频对比 ${esc(cmp.version)}：${delta(cmp.misses_reduced,'漏判')}，${delta(cmp.false_alarms_reduced,'误报')}。</p>`;
  }
  if(p)html+=`<p>复用 ${p.cache_hits||0} 段，新增处理 ${p.extracted||0} 段，跳过 ${p.skipped||0} 段。耗时：视频处理 ${p.feature_seconds||0}s / 拟合 ${p.fit_seconds||0}s / 评估 ${p.evaluation_seconds||0}s。</p>`;
  const labels={fixed:'本次纠正',regressed:'本次退步',missed:'漏掉喝水',false_alarm:'误报喝水'};
  if(e.validation_examples&&e.validation_examples.length)html+='<details><summary>查看需要关注的视频（最多 12 段）</summary>'+e.validation_examples.map(x=>
    `<p>${labels[x.outcome]||'待检查'} · 人工：${x.label?'喝水':'没喝'} · <a href="/clips/${encodeURIComponent(x.clip)}" target="_blank" rel="noopener">${esc(x.clip)}</a></p>`).join('')+'<small>录像已清理时，链接可能不可用；评估仍可使用保留的特征。</small></details>';
  return html;
}
function fmtTime(ts){if(!ts)return '';const d=new Date(ts*1000);
  const p=n=>String(n).padStart(2,'0');return `${d.getMonth()+1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;}
async function pollTrain(){
  const active=$('#activeBox'),models=$('#modelList');
  if(active&&!active.innerHTML.trim())active.innerHTML='<div class="empty">正在读取模型状态…</div>';
  if(models&&!models.innerHTML.trim())models.innerHTML='<div class="empty">正在读取模型版本…</div>';
  try{
    const response=await fetch('/api/train/status',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const s=await response.json();
    const ls=s.training_labels||{human:0,machine:0,drinking:0,not_drinking:0};
    $('#dsUn').textContent=(s.unlabeled??'–');
    $('#dsNew').textContent=ls.human; $('#dsTr').textContent=ls.machine;
    $('#dsBal').textContent=`${ls.drinking} / ${ls.not_drinking}`;
    renderActive(s); renderModels(s);
  }catch(e){
    const msg='模型数据加载失败，请刷新重试';
    if(active)active.innerHTML=`<div class="empty">${msg}</div>`;
    if(models)models.innerHTML=`<div class="empty">${msg}</div>`;
    const st=$('#trainVideoStatus');if(st)st.textContent=msg;
    console.error('加载模型状态失败',e);
  }
}
let trainVideoTimer=null;
async function trainVideo(){
  const rebuild=!!($('#trainVideoRebuild')&&$('#trainVideoRebuild').checked);
  const r=await (await fetch('/api/train_video',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({rebuild})})).json();
  if(!r.started&&r.error){toast(r.error,'err');$('#trainVideoStatus').textContent=r.error;return;}
  toast('训练已启动','info');
  pollTrainVideo();
}
function renderTrainProg(s){
  const p=$('#trainVideoProg'); if(!p)return;
  const phase=s.phase,done=s.done||0,total=s.total||0,prog=s.progress||0;
  const lab=phase==='extracting'?`抽取特征 ${done}/${total} 段`
    :phase==='training'?'训练分类器…'
    :phase==='evaluating'?'对比验证结果…'
    :'检查缓存、准备视频特征…（首次缺少权重时需要下载）';
  const det=phase==='extracting';
  const pct=Math.round(prog*100);
  p.style.display='block';
  p.innerHTML=`<div class="pbar-track"><div class="pbar-fill ${det?'det':'indet'}" `+
    `style="${det?'width:'+pct+'%':''}"></div></div>`+
    `<div class="pbar-lab"><span>${lab} · 已用 ${s.elapsed_seconds||0}s</span><span>${det?pct+'%':''}</span></div>`+
    `<div class="amini">缓存复用 ${(s.performance||{}).cache_hits||0} 段 · 新处理 ${(s.performance||{}).extracted||0} 段</div>`;
}
async function pollTrainVideo(){
  try{
    const s=await (await fetch('/api/train_video/status')).json();
    const btn=$('#trainVideoBtn'),st=$('#trainVideoStatus'),prog=$('#trainVideoProg');
    if(s.state==='disabled'){btn.disabled=true;st.textContent='演示模式未接训练器——用 python -m catcam 跑真实采集后这里可用';return;}
    if(s.state==='running'){
      btn.disabled=true;st.textContent=s.detail||'训练中…';
      renderTrainProg(s);
      if(!trainVideoTimer)trainVideoTimer=setInterval(pollTrainVideo,2000);
    }else{
      if(trainVideoTimer){clearInterval(trainVideoTimer);trainVideoTimer=null;}
      if(prog)prog.style.display='none';
      btn.disabled=false;
      const r=s.result;
      if(s.state==='done'&&r){
        st.innerHTML=`完成 ${r.version} · <b>喝水召回 ${fmtPct(r.drinking_recall)}</b> `+
          `精确 ${fmtPct(r.drinking_precision)} <span style="color:var(--faint)">`+
          `(top1 ${fmtPct(r.top1)}，多数类基线 ${fmtPct(r.naive_baseline)}；`+
          `样本 👍${r.counts.drinking}/👎${r.counts.not_drinking})</span>`;
        if(r.release)st.innerHTML+=`<br>${r.release.eligible?'达到过滤模式门槛；仍建议先影子观察':esc(r.release.reasons.join('；'))}`;
        if(r.reused)st.innerHTML+='<br>数据和参数未变化，复用已有版本。';
        if(r.comparison&&r.comparison.note)st.innerHTML+=`<br>${esc(r.comparison.note)}`;
        st.innerHTML+=trainingEvidence(r);
        st.innerHTML+=s.active===r.version?'<p>此版本已用于后续录像；历史预测不会自动改写。</p>':
          `<p>此版本尚未用于后续录像。<button class="mbtn" onclick="activate('${esc(r.version)}','shadow')">启用影子观察</button></p>`;
      }else{st.textContent=s.detail||'';}
      if(s.models)renderModels(s);
    }
  }catch(e){}
}
function renderActive(s){
  const box=$('#activeBox'); if(!box)return;
  const m=(s.models||[]).find(x=>x.id===s.active),mode=s.active_mode||'shadow';
  if(!m){
    box.innerHTML=`<div style="display:flex;align-items:center;gap:13px;flex-wrap:wrap">
      <span class="tag mute" style="font-size:13px;padding:7px 14px"><i></i>未启用</span>
      <span style="font-size:13px;color:var(--muted)">仅用简单模型兜底（宁可多录候选）</span></div>
      <p class="amini">训练出的模型默认<b>测试模式</b>：只预测、不拦截录制，简单模型继续兜底全录；
      等它在真实数据上够准了，再切「过滤模式」让它过滤误触。</p>`;
    return;
  }
  const hr=s.hitrate;
  const hrTxt=hr&&hr.total?`人工复核 <b>${hr.correct}/${hr.total}</b> · 漏判 ${hr.fn||0} · 误报 ${hr.fp||0}`
    :'实战命中 <b>—</b>（录到新喝水并标注后累计）';
  box.innerHTML=`<div style="display:flex;align-items:center;gap:13px;flex-wrap:wrap">
    <span class="tag ok" style="font-size:13px;padding:7px 14px"><i></i>${m.id} 生效中</span>
    <span style="font-size:13px;color:var(--muted)">验证 <b style="color:var(--ink)">${fmtAcc(m.top1)}</b> · ${hrTxt}</span></div>
    <div class="seg-ctl" style="margin-top:14px">
      <button class="${mode==='shadow'?'on':''}" onclick="activate('${m.id}','shadow')">测试模式</button>
      <button class="${mode==='gate'?'on':''}" onclick="activate('${m.id}','gate')">过滤模式</button></div>
    <p class="amini">${mode==='gate'
      ?(m.base==='s3d+head'?'<b>过滤模式</b>：视频模型的判断用于饮水统计；候选录像继续保留供复核。':'<b>过滤模式</b>：单帧模型可能阻止候选录制，请留意漏录。')
      :'<b>测试模式</b>：只预测打分、<b>不拦截录制</b>，简单模型兜底全录；在标注工作台看模型判得准不准。'}</p>`;
}
function renderModels(s){
  const box=$('#modelList'); if(!box)return;
  const models=s.models||[];
  let html=`<div class="mrow ${!s.active?'on':''}"><div><div class="mv">不启用任何模型</div>
    <div class="mmeta">只用简单模型，宁可多录候选</div></div><div class="grow"></div>
    <button class="mbtn ${!s.active?'cur':'off'}" ${!s.active?'':"onclick=\\"activate(null)\\""}>${!s.active?'生效中':'停用模型'}</button></div>`;
  if(!models.length){html+=`<div class="empty">还没有训练过的模型。去「模型训练」标注后训一个。</div>`;}
  html+=models.map(m=>{const cur=m.id===s.active,ic=m.image_counts||{},lc=m.label_counts||{};
    const e=m.evaluation,release=e&&e.release;
    const evidence=e?`人工验证 · 召回 ${fmtPct(e.drinking_recall)} · 精确 ${fmtPct(e.drinking_precision)} · 平衡准确率 ${fmtPct(e.balanced_accuracy)}<br>${release&&release.eligible?'达到过滤门槛':esc((release&&release.reasons||[]).join('；'))}`:'旧版：缺少独立人工验证记录';
    const comparison=e&&e.comparison;
    const comparisonText=comparison&&comparison.status==='compared'?`<br>同场对比 ${esc(comparison.version)}：旧版召回 ${fmtPct(comparison.metrics.drinking_recall)} · 精确 ${fmtPct(comparison.metrics.drinking_precision)}`:comparison&&comparison.note?`<br>${esc(comparison.note)}`:'';
    return `<div class="mrow ${cur?'on':''}"><div><div class="mv">${m.id} <span class="macc">${fmtAcc(m.top1)}</span></div>
      <div class="mmeta">${fmtTime(m.created_ts)} · 样本 👍${ic.drinking||0}/👎${ic.not_drinking||0}<br>${evidence}${comparisonText}${trainingEvidence(e)}</div></div>
      <div class="grow"></div>
      <button class="mbtn ${cur?'cur':''}" ${cur?'':`onclick="activate('${m.id}')"`}>${cur?'生效中':'设为生效'}</button></div>`;
  }).join('');
  box.innerHTML=html;
}
async function activate(id,mode){
  $$('.mbtn,#activeBox .seg-ctl button').forEach(b=>b.disabled=true);
  try{const response=await fetch('/api/model/activate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,mode:mode||'shadow'})});
    const r=await response.json();
    if(!response.ok){toast(r.detail||'操作失败','err');return;}
    if(r&&r.note)toast(r.note,'info');else toast(id?`${id} 已生效`:'已停用模型');}
  catch(e){toast('操作失败','err');}
  finally{pollTrain();pollTrainVideo();}
}

async function pollVisibility(){
  const el=$('#visibilityStatus');
  try{
    const response=await fetch('/api/visibility/status');
    if(!response.ok)throw new Error('unavailable');
    const state=await response.json();
    el.textContent=state.reason+(state.can_judge?'':'。此时的零记录不代表猫没有喝水。');
    el.style.color=state.can_judge?'var(--muted)':'var(--coral)';
  }catch(e){el.textContent='暂时无法获取光照状态；请检查实时画面。';}
}

/* ---------- 启动 ---------- */
bindCardFx($('#recGrid')); bindCardFx($('#labelClips'));
show((location.hash||'#/home').replace('#/','').replace('#',''));
loadStats(); setInterval(loadStats,5000);
loadDispenser(); setInterval(loadDispenser,15000);
pollVisibility(); setInterval(pollVisibility,5000);
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
    video_model_switch=None,
    video_model_clear=None,
    audio_status_provider=None,
    dispenser=None,
    dispenser_low_water_pct: float = 0.2,
    visibility_status_provider=None,
) -> FastAPI:
    app = FastAPI()
    clips_dir = Path(clips_dir)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/api/stats/range")
    def stats_range(days: int = 7):
        days = max(1, min(int(days), 90))
        points = stats.daily_counts(datetime.now(), days)
        return {"days": [{"date": d, "count": c} for d, c in points]}

    @app.get("/api/audio/status")
    def audio_status():
        if audio_status_provider is None:
            return {"enabled": False, "available": False}
        return {"enabled": True, **audio_status_provider()}

    @app.get("/api/visibility/status")
    def visibility_status():
        if visibility_status_provider is None:
            return {"status": "unknown", "can_judge": False, "reason": "此入口未接入实时光照检测"}
        return visibility_status_provider()

    @app.get("/api/stats/trend")
    def stats_trend(days: int = 7):
        """趋势页一次取齐：每日次数 + KPI（总计/活跃天数）+ 环比 + 时段/星期分布。"""
        days = max(1, min(int(days), 90))
        now = datetime.now()
        points = stats.daily_counts(now, days)
        # 当前窗口 [start, end)
        end_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        start_day = end_day - timedelta(days=days)
        evs = stats.events_between(start_day.timestamp(), end_day.timestamp())
        buckets = bucket_events(evs)
        total = len(evs)
        active_days = sum(1 for _, c in points if c > 0)
        prev_start = start_day - timedelta(days=days)
        prev_total = stats.count_between(prev_start.timestamp(), start_day.timestamp())
        return {
            "days": [{"date": d, "count": c} for d, c in points],
            "total": total,
            "active_days": active_days,
            "prev_total": prev_total,
            "hourly": buckets["hourly"],
            "weekday": buckets["weekday"],
        }

    def _dispenser_payload() -> dict:
        st = dispenser.get_state()
        now = time.time()
        mpd = st["ml_per_drink"]
        refill_ts, refill_ml = st["last_refill_ts"], st["last_refill_ml"]
        drinks_since = stats.count_between(refill_ts, now) if refill_ts > 0 else 0
        remaining = estimate_remaining(refill_ml, drinks_since, mpd) if refill_ml > 0 else 0.0
        remaining_pct = (remaining / refill_ml) if refill_ml > 0 else None
        fdl = (filter_days_left(st["last_filter_change_ts"], st["filter_cycle_days"], now)
               if st["last_filter_change_ts"] > 0 else None)
        # 今日约喝 ml = 今日喝水次数 × 每次 ml
        d0, d1 = day_bounds(datetime.now())
        today_ml = stats.count_between(d0, d1) * mpd
        return {
            "has_refill": refill_ts > 0,
            "has_filter": st["last_filter_change_ts"] > 0,
            "last_refill_ts": refill_ts,
            "last_refill_ml": refill_ml,
            "drinks_since_refill": drinks_since,
            "remaining_ml": round(remaining),
            "remaining_pct": remaining_pct,
            "ml_per_drink": round(mpd, 1),
            "calibrated": st["calib_drinks"] > 0,
            "today_ml": round(today_ml),
            "last_filter_change_ts": st["last_filter_change_ts"],
            "filter_cycle_days": st["filter_cycle_days"],
            "filter_days_left": None if fdl is None else round(fdl, 1),
            "need_water": bool(refill_ts > 0 and remaining_pct is not None
                               and remaining_pct < dispenser_low_water_pct),
            "need_filter": bool(fdl is not None and fdl <= 0),
        }

    @app.get("/api/dispenser")
    def dispenser_get():
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        return _dispenser_payload()

    @app.post("/api/dispenser/refill")
    def dispenser_refill(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        try:
            ml = float(body.get("ml", 0))
        except (TypeError, ValueError):
            ml = 0.0
        if not math.isfinite(ml) or ml <= 0:
            raise HTTPException(status_code=400, detail="加水量要 > 0")
        now = time.time()
        st = dispenser.get_state()
        prev_drinks = stats.count_between(st["last_refill_ts"], now) if st["last_refill_ts"] > 0 else 0
        dispenser.refill(ml=ml, now=now, prev_drinks=prev_drinks)
        return _dispenser_payload()

    @app.post("/api/dispenser/filter")
    def dispenser_filter():
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        dispenser.mark_filter(time.time())
        return _dispenser_payload()

    @app.post("/api/dispenser/config")
    def dispenser_config(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        days = body.get("filter_cycle_days")
        if days is not None:
            try:
                d = int(days)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="周期要是整数天")
            if d < 1:
                raise HTTPException(status_code=400, detail="周期至少 1 天")
            dispenser.set_cycle(d)
        return _dispenser_payload()

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
        s["training_labels"] = feedback.training_summary()
        s["unlabeled"] = _unlabeled_count()  # 待标注（当前还在的视频里没标的）
        if registry is not None:
            s["active_mode"] = registry.active_mode()
            s["hitrate"] = stats.model_hitrate(registry.active_id()) if registry.active_id() else None
        return s

    @app.post("/api/train_video")
    def train_video(body: dict = Body(default={})):
        if video_trainer is None:
            return JSONResponse({"started": False, "error": "本入口未启用视频训练"})
        rebuild = bool((body or {}).get("rebuild"))   # 从头重建：忽略缓存、强制重抽特征
        started = video_trainer.start(rebuild=rebuild)
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
        if model_id is None:
            active_model.clear()
            if video_model_clear is not None:
                video_model_clear()
        else:
            entry = registry.get(model_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="没有这个版本")
            if entry and entry.get("base") == "s3d+head":
                if mode == "gate":
                    from catcam.models import check_video_release
                    try:
                        check_video_release(entry)
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e))
                if video_model_switch is None:
                    raise HTTPException(status_code=400, detail="当前进程未接入本地视频裁判")
                try:
                    video_model_switch(entry, mode)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=500, detail=f"视频模型加载失败：{e}")
                active_model.clear()
            else:
                path = entry.get("path")
                if not path or not Path(path).exists():
                    raise HTTPException(status_code=404, detail="模型文件丢了")
                try:
                    active_model.set(DrinkingClassifier.from_path(path), model_id, mode)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=500, detail=f"加载失败：{e}")
                if video_model_clear is not None:
                    video_model_clear()
        registry.set_active(model_id, mode)
        note = "模型已立即生效，将用于下一段录像" if model_id else "模型已停用"
        return {"active": registry.active_id(), "mode": registry.active_mode(), "note": note}

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
        predictions = {n: preds[n] for n in names if n in preds}
        details = stats.clip_prediction_details()
        prediction_details = {n: details[n] for n in names if n in details}
        meta = {n: feedback.label_meta(n) for n in names}
        return {"clips": names, "labels": labels, "durations": durations,
                "predictions": predictions, "prediction_details": prediction_details,
                "meta": meta,
                "max_clips": recorder.max_clips}

    @app.get("/clips/{name}/thumb.jpg")
    def clip_thumb(name: str):
        # 用中间帧做封面，比开头的空场/刚入画更有判断价值。
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="no frame")
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/clips/{name}")
    def get_clip(name: str, audio: int = 1, download: int = 0):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        # audio=0：现场用 ffmpeg 去掉音轨后流式吐出（视频轨 copy，不重编码）。
        # ffmpeg 不可用 → 回退原文件（fail-open，宁可给原始也不 500）。
        if audio == 0 and shutil.which("ffmpeg") is not None:
            proc = subprocess.Popen(muted_cmd(str(path)),
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def _stream():
                try:
                    while True:
                        chunk = proc.stdout.read(65536)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    proc.stdout.close()
                    proc.terminate()

            muted_name = name[:-4] + "_muted.mp4" if name.endswith(".mp4") else name + "_muted.mp4"
            return StreamingResponse(
                _stream(), media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{muted_name}"'})
        headers = ({"Content-Disposition": f'attachment; filename="{name}"'}
                   if download else None)
        return FileResponse(path, media_type="video/mp4", headers=headers)

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
