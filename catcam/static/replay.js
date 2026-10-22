"use strict";

const archive = { catalog: null, segments: [], selected: null, clock: null, pendingOffset: 0,
  pendingPlay: false, waiting: false, loading: false, scrubbing: false, start: 0, end: 1,
  eventLimit: 60, lastEventSignature: "", generation: 0, error: false, notice: "" };
const archiveNode = identifier => document.getElementById(identifier);
const archiveVideo = archiveNode("archiveVideo");
const archiveDate = seconds => new Date(seconds * 1000).toLocaleString("zh-CN", {hour12: false});
const archiveTime = seconds => new Date(seconds * 1000).toLocaleTimeString("zh-CN", {hour12: false});

function archiveStatus(message, failed = false) {
  archiveNode("archiveState").textContent = message;
  archiveNode("archiveState").classList.toggle("archive-error", failed);
}

function archiveLocalInput(seconds) {
  const date = new Date(seconds * 1000);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

function pauseReplay() {
  archiveVideo.pause();
  archive.pendingPlay = false;
  archive.waiting = false;
}

async function loadReplay(quiet = false) {
  if (archive.loading) return;
  archive.loading = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("/api/replay", {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "读取滚动录像失败");
    if (!Array.isArray(data.segments) || !Array.isArray(data.events)) throw new Error("录像目录格式无效");
    archive.catalog = data;
    archive.segments = data.segments.filter(item => Number.isFinite(item.start) && Number.isFinite(item.end) && item.end > item.start &&
      typeof item.url === "string" && item.url.startsWith("/api/replay/media/")).sort((first, second) => first.start - second.start || first.id.localeCompare(second.id));
    archiveNode("replaySource").textContent = data.source_label;
    const retention = data.retention_seconds;
    const retentionText = retention ? retention >= 86400 ? `${(retention / 86400).toFixed(1).replace(/\.0$/, "")} 天` : `${Math.round(retention / 60)} 分钟` : "按本机策略";
    archiveNode("replayCount").textContent = `滚动保留 ${retentionText}${data.source === "wosee" ? data.active ? " · 正在录像" : data.enabled ? " · 等待画面" : " · 录像已关闭" : " · 仅活动时段有录像"}`;
    archiveNode("replaySource").parentElement.classList.toggle("recording", Boolean(data.active));
    const available = archive.segments.length > 0;
    for (const identifier of ["archivePlay", "archiveBack", "archiveForward", "archiveSeek", "archiveLocate", "archiveDate", "archiveZoom"]) archiveNode(identifier).disabled = !available;
    if (!available) {
      pauseReplay(); archive.selected = null; archive.clock = null;
      archiveVideo.removeAttribute("src"); archiveVideo.load(); archiveVideo.hidden = true;
      archive.start = 0; archive.end = 1;
      archiveNode("archiveStart").textContent = "—"; archiveNode("archiveEnd").textContent = "—";
      archiveNode("archiveWindow").textContent = "等待录像";
      archiveNode("archiveEmpty").hidden = false;
      archiveNode("archiveClock").hidden = true;
      archiveNode("archiveEmpty").querySelector("p").textContent = data.source === "wosee" ?
        "当前设备当前目录暂无已完成的滚动录像。开启 WoSee 录像后，约 30 秒出现可回放内容。" : "暂无带时长信息的监控录像，未录制的时间无法回放。";
      archiveStatus("暂无可回放时段。录像完成分段后会自动加入时间轴。");
      archive.lastEventSignature = ""; renderArchiveEvents(); drawArchiveTimeline();
      return;
    }
    const retained = archive.selected && archive.segments.find(item => item.id === archive.selected.id && item.url === archive.selected.url);
    if (archive.selected && !retained) {
      pauseReplay(); archive.selected = null; archive.error = true;
      archiveVideo.removeAttribute("src"); archiveVideo.load(); archiveVideo.hidden = true;
      archiveNode("archiveEmpty").hidden = false;
      archiveNode("archiveEmpty").querySelector("p").textContent = "当前时段已被覆盖或存储目录已切换，请重新选择时间。";
      archiveStatus("当前录像已被覆盖或目录已切换，请在时间轴选择仍可用的时段。", true);
    }
    if (archive.clock === null) archive.clock = archive.segments[0].start;
    updateArchiveWindow(); renderArchiveEvents(); updateArchivePosition();
    if (archive.waiting) {
      const next = archiveNext();
      if (next) playArchiveSegment(next, Math.max(next.start, archive.clock), true);
      else if (!data.active) {
        archive.waiting = false;
        archiveStatus("已到可回放录像末尾，后台录像当前未运行。");
      }
    } else if (!quiet && !archive.selected && !archive.error) {
      archiveStatus("拖动时间轴，或点击播放从最早可用时间开始。青色为有录像的时段。");
    }
  } catch (error) {
    if (!quiet || !archive.catalog) archiveStatus(error.name === "AbortError" ? "读取录像目录超时，请检查本机录像服务。" : error.message, true);
    if (!archive.catalog) {
      archiveNode("replaySource").textContent = "录像源暂不可用";
      archiveNode("archiveEmpty").querySelector("p").textContent = "请检查 WoSee 管理台的设备与录像开关，再点击刷新时间轴。";
    }
  } finally {
    clearTimeout(timeout); archive.loading = false;
  }
}

function updateArchiveWindow() {
  if (!archive.segments.length) return;
  const first = archive.segments[0].start;
  const last = Math.max(...archive.segments.map(item => item.end));
  const selected = archiveNode("archiveZoom").value;
  const span = selected === "all" ? last - first : Math.min(Number(selected), last - first);
  const center = archive.clock ?? first;
  archive.start = selected === "all" ? first : Math.max(first, Math.min(center - span / 2, last - span));
  archive.end = Math.max(archive.start + 0.1, archive.start + span);
  const seek = archiveNode("archiveSeek"); seek.min = archive.start; seek.max = archive.end;
  if (!archive.scrubbing) seek.value = center;
  archiveNode("archiveStart").textContent = archiveDate(archive.start);
  archiveNode("archiveEnd").textContent = archiveDate(archive.end);
  archiveNode("archiveWindow").textContent = `${new Date(first * 1000).toLocaleDateString("zh-CN")} · ${archiveTime(first)} — ${archiveTime(last)}`;
  const date = archiveNode("archiveDate");
  date.min = archiveLocalInput(first); date.max = archiveLocalInput(last);
  drawArchiveTimeline();
}

function drawArchiveTimeline() {
  const canvas = archiveNode("archiveTimeline");
  const width = Math.max(1, canvas.clientWidth);
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio); canvas.height = 102 * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const style = getComputedStyle(document.documentElement);
  context.fillStyle = style.getPropertyValue("--line"); context.fillRect(0, 33, width, 26);
  const position = stamp => (stamp - archive.start) / Math.max(0.1, archive.end - archive.start) * width;
  context.fillStyle = style.getPropertyValue("--aqua");
  for (const item of archive.segments) {
    if (item.end < archive.start || item.start > archive.end) continue;
    context.globalAlpha = 0.65;
    context.fillRect(position(item.start), 33, Math.max(2, position(item.end) - position(item.start)), 26);
  }
  context.globalAlpha = 1;
  context.strokeStyle = style.getPropertyValue("--muted"); context.fillStyle = style.getPropertyValue("--muted");
  context.font = "10px ui-monospace, monospace";
  for (let index = 0; index <= 6; index++) {
    const left = index / 6 * width;
    context.beginPath(); context.moveTo(left, 61); context.lineTo(left, 67); context.stroke();
    const label = archiveTime(archive.start + index / 6 * (archive.end - archive.start));
    context.fillText(label, Math.max(0, Math.min(width - context.measureText(label).width, left - 24)), 84);
  }
  for (const event of archive.catalog?.events || []) {
    if (event.ts < archive.start || event.ts > archive.end || !["drinking", "review"].includes(event.kind)) continue;
    const left = position(event.ts);
    context.fillStyle = style.getPropertyValue(event.kind === "drinking" ? "--green" : "--amber");
    context.beginPath(); context.moveTo(left, 23); context.lineTo(left - 5, 15); context.lineTo(left + 5, 15); context.fill();
    context.fillRect(left - 0.5, 23, 1, 40);
  }
  if (archive.clock !== null) {
    const left = Math.max(1, Math.min(width - 1, position(archive.clock)));
    context.fillStyle = style.getPropertyValue("--ink"); context.fillRect(left - 1, 6, 2, 90);
    context.beginPath(); context.arc(left, 8, 4, 0, 2 * Math.PI); context.fill();
  }
}

function updateArchivePosition() {
  if (archive.clock === null) return;
  archiveNode("archivePosition").textContent = archiveDate(archive.clock);
  archiveNode("archiveClock").textContent = archiveDate(archive.clock);
  if (!archive.scrubbing) archiveNode("archiveSeek").value = archive.clock;
  if (document.activeElement !== archiveNode("archiveDate")) archiveNode("archiveDate").value = archiveLocalInput(archive.clock);
  const rows = archiveNode("archiveEvents").querySelectorAll("[data-time]");
  for (const row of rows) row.classList.toggle("current", Math.abs(Number(row.dataset.time) - archive.clock) < 15);
  drawArchiveTimeline();
}

function playArchiveSegment(item, stamp, autoplay) {
  archive.selected = item; archive.waiting = false; archive.error = false;
  archive.notice = "";
  archive.clock = Math.max(item.start, Math.min(stamp, item.end));
  archive.pendingOffset = archive.clock - item.start;
  archive.pendingPlay = autoplay; archive.generation++;
  archiveVideo.hidden = false; archiveNode("archiveEmpty").hidden = true;
  archiveNode("archiveClock").hidden = false;
  archiveVideo.src = item.url; archiveVideo.load();
  archiveStatus("正在载入所选时间的监控录像…"); updateArchivePosition();
}

function seekArchive(stamp, autoplay = true) {
  if (!Number.isFinite(stamp) || !archive.segments.length) return;
  const first = archive.segments[0].start;
  const last = Math.max(...archive.segments.map(item => item.end));
  if (stamp < first - 1 || stamp > last + 1) { archiveStatus("所选时间超出当前录像保留范围。", true); return; }
  stamp = Math.min(stamp, last - 0.05);
  const item = archive.segments.find(item => stamp >= item.start && stamp < item.end) || archive.segments.find(item => item.start >= stamp);
  if (!item) { archiveStatus("此时刻的录像尚未完成或已被覆盖，请选择更早的时间。", true); return; }
  const gap = stamp < item.start - 0.5;
  const actual = Math.max(item.start, stamp);
  if (archive.selected?.url === item.url && archiveVideo.readyState >= 1 && !archive.error) {
    archive.waiting = false; archive.pendingPlay = autoplay; archive.clock = actual;
    archiveVideo.currentTime = Math.min(actual - item.start, Math.max(0, archiveVideo.duration - 0.05));
    if (autoplay) archiveVideo.play().catch(() => archiveStatus("已定位，请点击播放。"));
    updateArchivePosition();
  } else playArchiveSegment(item, actual, autoplay);
  archive.notice = gap ? `该时段没有录像，已跳至下一可用时间 ${archiveDate(item.start)}。` : "";
  if (archive.notice) archiveStatus(archive.notice);
}

function archiveNext() {
  if (!archive.selected) return null;
  return archive.segments.find(item => item.start > archive.selected.start && item.end > archive.selected.end + 0.05 && item.id !== archive.selected.id);
}

function renderArchiveEvents() {
  const filter = archiveNode("archiveEventFilter").value;
  const events = (archive.catalog?.events || []).filter(event => event.ts >= archive.start && event.ts <= archive.end && (filter === "all" || event.kind === filter));
  const signature = JSON.stringify([filter, archive.start, archive.end, archive.eventLimit, events]);
  if (signature === archive.lastEventSignature) return;
  archive.lastEventSignature = signature;
  const list = archiveNode("archiveEvents"); list.replaceChildren();
  archiveNode("archiveEventCount").textContent = `${events.length} 条`;
  archiveNode("archiveMore").hidden = events.length <= archive.eventLimit;
  for (const event of events.slice(0, archive.eventLimit)) {
    const row = document.createElement("button"); row.className = `archive-event ${event.kind}`; row.dataset.time = event.ts;
    const time = document.createElement("time"); time.textContent = archiveTime(event.ts);
    const details = document.createElement("span"); details.className = "archive-event-detail";
    const title = document.createElement("strong"); title.textContent = event.label;
    const note = document.createElement("small"); note.textContent = event.note || (event.seconds ? `此片段喝水时长约 ${event.seconds} 秒` : "点击定位到对应监控画面");
    const arrow = document.createElement("span"); arrow.textContent = "↗"; arrow.className = "archive-event-arrow";
    details.append(title, note); row.append(time, details, arrow);
    row.addEventListener("click", () => seekArchive(event.ts)); list.append(row);
  }
  if (!events.length) {
    const empty = document.createElement("p"); empty.className = "archive-events-empty";
    empty.textContent = "当前时间范围没有对应记录。录像可直接回看，识别或人工确认后会在时间轴标记。"; list.append(empty);
  }
}

archiveNode("replayRefresh").addEventListener("click", () => loadReplay());
archiveNode("archivePlay").addEventListener("click", () => {
  if (!archiveVideo.paused) { pauseReplay(); return; }
  if (!archive.selected || archive.error) seekArchive(archive.clock ?? archive.segments[0]?.start);
  else if (archiveVideo.ended) {
    const next = archiveNext();
    if (next) playArchiveSegment(next, Math.max(next.start, archive.clock), true);
    else { archive.waiting = Boolean(archive.catalog?.active); archiveStatus(archive.waiting ? "已到最新录像，等待新的分段完成。" : "已到录像末尾。拖动时间轴重新选择时间。"); }
  } else archiveVideo.play().catch(() => archiveStatus("播放器暂不能播放，请刷新时间轴。", true));
});
archiveNode("archiveBack").addEventListener("click", () => seekArchive(Math.max(archive.segments[0].start, archive.clock - 10)));
archiveNode("archiveForward").addEventListener("click", () => seekArchive(archive.clock + 10));
archiveNode("archiveLocate").addEventListener("click", () => {
  const stamp = new Date(archiveNode("archiveDate").value).getTime() / 1000;
  if (!Number.isFinite(stamp)) archiveStatus("请先选择日期和时间。", true);
  else { seekArchive(stamp); updateArchiveWindow(); renderArchiveEvents(); }
});
archiveNode("archiveSeek").addEventListener("input", () => {
  archive.scrubbing = true; archive.clock = Number(archiveNode("archiveSeek").value); updateArchivePosition();
});
archiveNode("archiveSeek").addEventListener("change", () => {
  archive.scrubbing = false; seekArchive(Number(archiveNode("archiveSeek").value));
});
archiveNode("archiveSeek").addEventListener("pointermove", event => {
  const rect = event.currentTarget.getBoundingClientRect();
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const stamp = archive.start + fraction * (archive.end - archive.start);
  const hover = archiveNode("archiveHover"); hover.hidden = false;
  hover.textContent = archiveDate(stamp); hover.style.left = `${Math.max(12, Math.min(88, fraction * 100))}%`;
});
archiveNode("archiveSeek").addEventListener("pointerleave", () => { archiveNode("archiveHover").hidden = true; });
archiveNode("archiveRate").addEventListener("change", () => { archiveVideo.playbackRate = Number(archiveNode("archiveRate").value); });
archiveNode("archiveZoom").addEventListener("change", () => { archive.eventLimit = 60; updateArchiveWindow(); renderArchiveEvents(); });
archiveNode("archiveEventFilter").addEventListener("change", () => { archive.eventLimit = 60; renderArchiveEvents(); });
archiveNode("archiveMore").addEventListener("click", () => { archive.eventLimit += 60; renderArchiveEvents(); });
archiveNode("archiveFullscreen").addEventListener("click", () => {
  archiveNode("archiveVideo").closest(".archive-player").requestFullscreen().catch(() => archiveStatus("浏览器未允许全屏。"));
});
archiveVideo.addEventListener("loadedmetadata", () => {
  archiveVideo.playbackRate = Number(archiveNode("archiveRate").value);
  archiveVideo.currentTime = Math.min(archive.pendingOffset, Math.max(0, archiveVideo.duration - 0.05));
  if (archive.pendingPlay && archiveNode("pg-replay").classList.contains("on") && !document.hidden) {
    const generation = archive.generation;
    archiveVideo.play().catch(() => { if (generation === archive.generation) archiveStatus("已定位，请点击播放。 "); });
  } else archiveStatus(`已定位到 ${archiveDate(archive.clock)}，点击播放可连续回看。`);
});
archiveVideo.addEventListener("play", () => { archive.waiting = false; archiveNode("archivePlay").textContent = "暂停"; archiveNode("archivePlay").setAttribute("aria-label", "暂停回放"); archiveStatus(archive.notice || "正在连续回放监控录像，后台录制继续运行。"); });
archiveVideo.addEventListener("pause", () => { archiveNode("archivePlay").textContent = "播放"; archiveNode("archivePlay").setAttribute("aria-label", "播放回放"); });
archiveVideo.addEventListener("timeupdate", () => {
  if (!archive.selected || archive.scrubbing) return;
  archive.clock = Math.min(archive.selected.end, archive.selected.start + archiveVideo.currentTime);
  if (archive.clock > archive.end || archive.clock < archive.start) { updateArchiveWindow(); renderArchiveEvents(); }
  updateArchivePosition();
});
archiveVideo.addEventListener("ended", () => {
  const next = archiveNext();
  if (next) {
    const gap = next.start - archive.selected.end;
    playArchiveSegment(next, Math.max(next.start, archive.selected.end), true);
    if (gap > 1) {
      archive.notice = `中间有 ${Math.round(gap)} 秒没有录像，继续回放 ${archiveDate(next.start)}。`;
      archiveStatus(archive.notice);
    }
  } else {
    archive.waiting = Boolean(archive.catalog?.active);
    archiveStatus(archive.waiting ? "已到最新完成录像，等待新分段完成后继续。" : "已播完当前可用录像。可拖动时间轴回看其他时段。");
  }
});
archiveVideo.addEventListener("error", () => {
  if (!archiveVideo.getAttribute("src")) return;
  archive.error = true; archive.waiting = false;
  archiveStatus("此时段录像已被覆盖、目录改变或编码无法播放。请刷新时间轴后重新定位。", true);
});
new ResizeObserver(drawArchiveTimeline).observe(archiveNode("archiveTimeline"));
document.addEventListener("visibilitychange", () => { if (document.hidden) pauseReplay(); });
setInterval(() => { if (!document.hidden && archiveNode("pg-replay").classList.contains("on") && !archive.scrubbing) loadReplay(true); }, 5000);
