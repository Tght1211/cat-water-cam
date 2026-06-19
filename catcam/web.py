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
<html lang="zh"><head><meta charset="utf-8"><title>猫咪饮水监控</title>
<style>body{font-family:sans-serif;margin:1rem;max-width:760px}
img{max-width:100%;border:1px solid #ccc}.clip{margin:.5rem 0;padding:.5rem;border:1px solid #eee}
button{font-size:1rem;margin-right:.5rem}</style></head>
<body>
<h2>实时画面</h2>
<img id="live" src="/snapshot.jpg" alt="live">
<h2>今日喝水：<span id="count">-</span> 次</h2>
<ul id="times"></ul>
<h2>最近视频</h2>
<div id="clips"></div>
<script>
setInterval(()=>{document.getElementById('live').src='/snapshot.jpg?t='+Date.now()},1000);
async function refresh(){
  const s=await (await fetch('/api/stats/today')).json();
  document.getElementById('count').textContent=s.count;
  document.getElementById('times').innerHTML=s.times.map(t=>`<li>${t}</li>`).join('');
  const c=await (await fetch('/api/clips')).json();
  const lab=c.labels||{};
  document.getElementById('clips').innerHTML=c.clips.map(n=>{
    const v=lab[n];
    const status=v===true?'✅ 已标注：真喝水':v===false?'❌ 已标注：没喝':'⬜ 未标注';
    return `<div class="clip"><video src="/clips/${n}" controls width="320"></video><br>
    <a href="/clips/${n}" download>下载 ${n}</a><br>
    <span>${status}</span><br>
    <button onclick="fb('${n}',true)">👍 真喝水</button>
    <button onclick="fb('${n}',false)">👎 没喝</button></div>`;
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
