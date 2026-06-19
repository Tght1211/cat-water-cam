from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.stats import StatsStore, day_bounds

INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>猫咪饮水监控</title>
<style>
:root{
  --bg:#f5f5f7; --surface:#ffffff; --nav:rgba(245,245,247,.72);
  --ink:#1d1d1f; --muted:#86868b; --line:rgba(0,0,0,.08);
  --accent:#0071e3; --accent2:#0a84ff; --green:#34c759; --red:#ff3b30;
  --shadow:0 8px 30px rgba(0,0,0,.06); --shadow-h:0 14px 40px rgba(0,0,0,.10);
  --radius:20px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#000; --surface:#1c1c1e; --nav:rgba(28,28,30,.7);
  --ink:#f5f5f7; --muted:#98989d; --line:rgba(255,255,255,.12);
  --accent:#0a84ff; --accent2:#409cff; --green:#30d158; --red:#ff453a;
  --shadow:0 8px 30px rgba(0,0,0,.5); --shadow-h:0 16px 44px rgba(0,0,0,.6);
}}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","PingFang SC","Microsoft YaHei",sans-serif;
  letter-spacing:-.01em}
.wrap{max-width:980px;margin:0 auto;padding:0 22px}
header{position:sticky;top:0;z-index:9;background:var(--nav);
  backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;justify-content:space-between;height:54px}
.brand{display:flex;align-items:center;gap:9px;font-size:17px;font-weight:600}
.brand svg{width:22px;height:22px;color:var(--accent)}
.pill{display:inline-flex;align-items:center;gap:7px;background:var(--accent);color:#fff;
  border-radius:980px;padding:6px 14px;font-size:13px;font-weight:600;letter-spacing:0}
.pill b{font-variant-numeric:tabular-nums}
main{padding:30px 0 70px}
.hero{display:grid;grid-template-columns:1.55fr 1fr;gap:18px;margin-bottom:34px}
@media (max-width:740px){.hero{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden}
.card-h{display:flex;align-items:center;gap:8px;padding:15px 18px 0;
  font-size:12px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
.card-b{padding:14px 18px 18px}
.live-frame{position:relative;border-radius:14px;overflow:hidden;background:#000;aspect-ratio:4/3}
#live{display:block;width:100%;height:100%;object-fit:cover}
.live-tag{position:absolute;top:10px;left:10px;display:flex;align-items:center;gap:6px;
  background:rgba(0,0,0,.55);color:#fff;border-radius:980px;padding:5px 11px;font-size:11px;font-weight:600;letter-spacing:.02em;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(52,199,89,.6);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,199,89,.5)}70%{box-shadow:0 0 0 7px rgba(52,199,89,0)}100%{box-shadow:0 0 0 0 rgba(52,199,89,0)}}
.stat .num{font-size:58px;font-weight:700;line-height:1;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.stat .unit{font-size:20px;font-weight:600;color:var(--muted);margin-left:6px}
.stat .lbl{color:var(--muted);font-size:13px;margin-top:8px}
.times{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}
.chip{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:4px 10px;
  font-size:12px;font-variant-numeric:tabular-nums}
.times-empty{color:var(--muted);font-size:13px;margin-top:14px}
.sec{display:flex;align-items:baseline;justify-content:space-between;margin:0 2px 16px}
.sec h2{font-size:22px;font-weight:700;letter-spacing:-.02em;margin:0}
.sec span{color:var(--muted);font-size:13px}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(224px,1fr));gap:18px}
.clip{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden;display:flex;flex-direction:column;
  transition:transform .25s cubic-bezier(.22,1,.36,1),box-shadow .25s}
.clip:hover{transform:translateY(-3px);box-shadow:var(--shadow-h)}
.clip video{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:cover}
.meta{padding:13px 14px 14px;display:flex;flex-direction:column;gap:11px}
.meta .top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.fname{font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.status{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
  padding:4px 9px;border-radius:980px;white-space:nowrap}
.status i{width:6px;height:6px;border-radius:50%}
.s-none{background:var(--bg);color:var(--muted)} .s-none i{background:var(--muted)}
.s-yes{background:rgba(52,199,89,.14);color:var(--green)} .s-yes i{background:var(--green)}
.s-no{background:rgba(255,59,48,.14);color:var(--red)} .s-no i{background:var(--red)}
.seg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.seg button{display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);
  border-radius:11px;padding:9px 0;font-size:13px;font-weight:500;cursor:pointer;
  transition:transform .12s,background .15s,border-color .15s,color .15s;
  font-family:inherit}
.seg button svg{width:15px;height:15px}
.seg button:hover{background:var(--bg)}
.seg button:active{transform:scale(.96)}
.seg .yes.on{background:var(--green);border-color:var(--green);color:#fff}
.seg .no.on{background:var(--red);border-color:var(--red);color:#fff}
.dl{display:inline-flex;align-items:center;gap:5px;color:var(--accent2);
  font-size:12px;text-decoration:none;font-weight:500}
.dl svg{width:13px;height:13px}
.empty{grid-column:1/-1;color:var(--muted);background:var(--surface);
  border:1px solid var(--line);border-radius:var(--radius);padding:54px 24px;text-align:center}
.empty svg{width:34px;height:34px;color:var(--muted);margin-bottom:12px}
.empty code{background:var(--bg);border:1px solid var(--line);padding:2px 7px;border-radius:6px;font-size:12px}
</style></head>
<body>
<header><div class="wrap">
<div class="brand">
<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.2C12 2.2 4.8 9.9 4.8 14.7a7.2 7.2 0 0 0 14.4 0C19.2 9.9 12 2.2 12 2.2Z"/></svg>
猫咪饮水监控
</div>
<span class="pill">今日 <b id="count">–</b> 次</span>
</div></header>
<main class="wrap">
<section class="hero">
<div class="card">
<div class="card-h">实时画面</div>
<div class="card-b">
<div class="live-frame">
<img id="live" src="/snapshot.jpg" alt="实时画面">
<span class="live-tag"><span class="dot"></span>在线</span>
</div></div></div>
<div class="card">
<div class="card-h">今日喝水</div>
<div class="card-b stat">
<div><span class="num" id="count2">–</span><span class="unit">次</span></div>
<div class="lbl">各次时间点</div>
<div class="times" id="times"></div>
</div></div>
</section>
<div class="sec"><h2>最近喝水</h2><span>最多保留 10 段</span></div>
<div class="clips" id="clips"></div>
</main>
<script>
const I={
  check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>',
  x:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  dl:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg>',
  cam:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L19 6h0a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><circle cx="12" cy="12.5" r="3.2"/></svg>'
};
setInterval(()=>{document.getElementById('live').src='/snapshot.jpg?t='+Date.now()},1000);
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function refresh(){
  const s=await (await fetch('/api/stats/today')).json();
  document.getElementById('count').textContent=s.count;
  document.getElementById('count2').textContent=s.count;
  const t=document.getElementById('times');
  t.innerHTML=s.times.length?s.times.map(x=>`<span class="chip">${esc(x)}</span>`).join('')
    :'<div class="times-empty">今天还没记录到喝水</div>';
  const c=await (await fetch('/api/clips')).json();
  const lab=c.labels||{};
  const box=document.getElementById('clips');
  if(!c.clips.length){box.innerHTML=`<div class="empty">${I.cam}<div>还没有视频。接上摄像头跑 <code>python -m catcam</code>，猫在水碗停留就会自动录制。</div></div>`;return;}
  box.innerHTML=c.clips.map(n=>{
    const v=lab[n];
    const st=v===true?`<span class="status s-yes"><i></i>真喝水</span>`
      :v===false?`<span class="status s-no"><i></i>没喝</span>`
      :`<span class="status s-none"><i></i>未标注</span>`;
    const en=esc(n),jn=JSON.stringify(n);
    return `<div class="clip">
      <video src="/clips/${en}" controls preload="metadata"></video>
      <div class="meta">
        <div class="top"><span class="fname">${en}</span>${st}</div>
        <div class="seg">
          <button class="yes ${v===true?'on':''}" onclick='fb(${jn},true)'>${I.check}喝了</button>
          <button class="no ${v===false?'on':''}" onclick='fb(${jn},false)'>${I.x}没喝</button>
        </div>
        <a class="dl" href="/clips/${en}" download>${I.dl}下载</a>
      </div></div>`;
  }).join('');
}
async function fb(clip,is){
  await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip,is_drinking:is})});
  refresh();
}
refresh();setInterval(refresh,5000);
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
) -> FastAPI:
    app = FastAPI()
    clips_dir = Path(clips_dir)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

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
        return {"clips": names, "labels": labels}

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

    @app.post("/api/feedback")
    def post_feedback(body: FeedbackBody):
        if "/" in body.clip or "\\" in body.clip or ".." in body.clip:
            raise HTTPException(status_code=400, detail="bad clip")
        feedback.label_clip(clips_dir / body.clip, body.is_drinking)
        return JSONResponse({"ok": True})

    return app
