
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
function cycleAppearance(){
  const modes=['dark','light'],current=document.documentElement.dataset.appearance||'dark';
  const mode=modes[(modes.indexOf(current)+1)%modes.length];
  try{localStorage.setItem('catcam-theme',mode);}catch(e){}
  applyAppearance();
}
applyAppearance();
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',applyAppearance);

async function logoutAccess(){
  const response=await fetch('/api/auth/logout',{method:'POST'});
  if(response.ok)location.replace('/login');
}
fetch('/api/setup').then(r=>r.json()).then(s=>{if(s.access_enabled){$('#logoutControl').hidden=false;$('#accessControl').hidden=false;}}).catch(()=>{});

function openMobileSettings(){
  const access=!$('#accessControl').hidden;
  openModal(`<h3>设置</h3><div class="mobile-settings"><button onclick="cycleAppearance();openMobileSettings()"><span>外观</span><strong>${document.documentElement.dataset.theme==='dark'?'深色':'浅色'} · 切换</strong></button><button onclick="openMailSetup()"><span>邮件提醒</span><strong>配置 →</strong></button>${access?'<button onclick="openAccessSetup()"><span>访问密码</span><strong>修改 →</strong></button><button onclick="logoutAccess()"><span>退出登录</span><strong>→</strong></button>':''}</div><div class="modal-acts"><button class="btn ghost" onclick="closeModal()">关闭</button></div>`);
}
function openAccessSetup(){
  openModal(`<h3>修改访问密码</h3><p class="msub">为局域网访问设置新密码。保存后，其他设备需要使用新密码重新登录。</p><form id="accessPasswordForm" onsubmit="event.preventDefault();saveAccessPassword()"><div class="mail-fields"><label class="mail-wide">当前密码<input id="accessCurrent" type="password" autocomplete="current-password" maxlength="256" required></label><label class="mail-wide">新密码<input id="accessNew" type="password" autocomplete="new-password" minlength="8" maxlength="256" placeholder="至少 8 个字符" required></label><label class="mail-wide">再次输入新密码<input id="accessConfirm" type="password" autocomplete="new-password" minlength="8" maxlength="256" required></label></div><div id="accessPasswordError" class="access-password-error" role="alert"></div><div class="modal-acts"><button class="btn ghost" type="button" onclick="closeModal()">取消</button><button class="btn" type="submit" id="accessPasswordSave">保存新密码</button></div></form>`);
}
async function saveAccessPassword(){
  const error=$('#accessPasswordError'),button=$('#accessPasswordSave');
  error.textContent='';
  if($('#accessNew').value!==$('#accessConfirm').value){error.textContent='两次输入的新密码不一致';return;}
  button.disabled=true;
  try{
    const r=await fetch('/api/auth/password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({current_password:$('#accessCurrent').value,new_password:$('#accessNew').value,confirm_password:$('#accessConfirm').value})}),d=await r.json();
    if(!r.ok){error.textContent=d.detail||'修改失败';return;}
    closeModal();toast('访问密码已更新，其他设备需要重新登录');
  }catch(e){error.textContent='连接失败，请重试';}finally{button.disabled=false;}
}

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
function openModal(html){$('#modalCard').classList.toggle('teacher-modal',html.includes('id="aiBase"'));$('#modalCard').classList.toggle('fountain-modal',$('#pg-disp').classList.contains('on')||/id="(?:rfNum|waterMeasured|cycleRemaining|signalRemaining|waterCapacity|cyV)"/.test(html));$('#modalCard').innerHTML=html;$('#modal').classList.add('show');}
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
const PAGES=['home','disp','records','replay','trend','lab'];
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
  if(p==='replay')loadReplay();
  else pauseReplay();
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
let statsTodaySignature='',statsWeekSignature='';
async function loadStats(){
  try{
    const s=await (await fetch('/api/stats/today')).json();
    const todaySignature=JSON.stringify(s);
    if(todaySignature!==statsTodaySignature){
    statsTodaySignature=todaySignature;
    countUp($('#sideCount'),s.count); countUp($('#heroCount'),s.count);
    const last=s.times.length?s.times[s.times.length-1]:null;
    $('#heroLast').innerHTML=last?`<i></i>最近一次 ${esc(last.slice(0,5))}`:'<i></i>今日暂无确认记录';
    $('#heroLast').className='tag '+(last?'ok':'mute');
    renderTimeline(s.times);
    }
    const w=await (await fetch('/api/stats/range?days=7')).json();
    const weekSignature=JSON.stringify(w);if(weekSignature===statsWeekSignature)return;statsWeekSignature=weekSignature;
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
  $('#tlSummary').textContent=times?.length?`已确认 ${times.length} 次`:'暂无喝水记录';
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
function dispState(d){return d.need_water?'low':'';}
async function loadDispenser(){
  try{const d=await (await fetch('/api/dispenser')).json();
    if(d&&!d.detail){dispData=d;if(d.last_refill_ml)dispLastMl=d.last_refill_ml;
      renderFlow(d.flow);renderDispMini(d);renderDisp(d);
      const nav=$('#navDisp');
      const dot=nav.querySelector('.adot');
      if(d.need_water||d.need_filter||d.flow?.alert){if(!dot)nav.insertAdjacentHTML('beforeend','<span class="adot"></span>');}
      else if(dot)dot.remove();
    }}catch(e){}
}
/* 总览速览小水箱 */
function renderDispMini(d){
  const box=$('#dmBody'),card=$('#storageCard'),note=$('#dmNote');if(!box||!card)return;
  const pct=d.has_refill&&d.remaining_pct!=null?Math.round(Math.max(0,Math.min(1,d.remaining_pct))*100):null;
  note.textContent=d.flow?.alert?'疑似停水':d.need_water?'水量偏低':d.flow?.state==='flowing'?'出水正常':'';
  card.classList.toggle('storage-low',!!d.need_water);card.classList.toggle('storage-fault',!!d.flow?.alert);
  card.classList.toggle('storage-unknown',pct==null);
  card.style.setProperty('--water-level',(pct??0)+'%');
  if(!card.querySelector('.storage-bubbles').children.length){
    card.querySelector('.storage-bubbles').innerHTML=Array.from({length:12},(_,i)=>`<i style="--x:${8+(i*23)%84}%;--size:${5+(i*7)%13}px;--duration:${9+(i*3)%10}s;--delay:-${(i*11)%23}s;--drift:${i%2?16:-14}px"></i>`).join('');
  }
  const supply=d.days_until_empty==null?'待周期校准':`约 ${d.days_until_empty} 天`;
  const filter=!d.has_filter?'待记录更换':d.need_filter?'该更换了':`约 ${Math.max(0,Math.round(d.filter_days_left||0))} 天`;
  box.innerHTML=`<div class="storage-reading"><span class="storage-label">剩余水量</span><div class="storage-percent">${pct==null?'—':pct}<small>${pct==null?'':'%'}</small></div><div class="storage-volume">${d.remaining_ml==null?'记录第一次加水':`约 ${d.remaining_ml} ml`}</div><span class="storage-basis">${!d.has_refill?'先加少量水，开始学习':d.capacity_ml?'按设备容量估算':'按本轮记录水量估算'}</span></div><div class="storage-details"><div><span>预计可供小猫饮用</span><strong>${supply}</strong></div><div><span>滤芯寿命剩余</span><strong>${filter}</strong></div></div>`;
  $('#msMl').textContent=d.today_ml==null?'—':d.today_ml;
}
/* 饮水机页 */
function renderDisp(d){
  const box=$('#dispBody');if(!box)return;
  const current=d.additions?.filter(a=>a.cycle_start===d.last_refill_ts)||[];
  const added=current.reduce((n,a)=>n+a.ml,0);
  const completed=d.reminder?.complete_cycles||0;
  const pct=d.has_refill&&d.remaining_pct!=null?Math.max(0,Math.min(100,Math.round(d.remaining_pct*100))):null;
  const steps=[['记录加水','少量开始，也可分次追加',d.has_refill],['确认喝水','积累真实饮水记录',d.drinks_since_refill>0],['测量残水','停水后反馈实际剩余量',d.cycle_closed],['校准周期','逐步学习消耗和提醒水位',completed>0]];
  box.innerHTML=`<div class="fountain-overview">
    <div class="fountain-tank ${d.need_water?'is-low':''}" style="--tank-level:${pct??0}%">
      <div class="fountain-tank-fill" aria-hidden="true"></div><div class="fountain-tank-top"><span>${I.drop} 储水状态</span><span class="fountain-estimate">估算</span></div>
      <div class="fountain-tank-reading"><span>当前剩余水量</span><strong>${d.remaining_ml==null?'—':Number(d.remaining_ml).toLocaleString()}<small>ml</small></strong><p>${pct==null?'记录加水后开始估算':`${pct}% · ${d.capacity_ml?'按设备容量':'按本轮记录水量'}`}</p></div>
      <div class="fountain-tank-foot"><span>预计还能饮用</span><b>${d.days_until_empty==null?'待周期校准':`约 ${d.days_until_empty} 天`}</b></div>
    </div>
    <div class="fountain-cycle"><div class="fountain-section-head"><div><span class="fountain-kicker">${d.cycle_closed?'本轮已结束':'当前用水周期'}</span><h2>${d.cycle_closed?'下一次加水，开始新一轮':d.has_refill?'本轮用水概览':'从第一次加水开始'}</h2></div><span class="tag ${d.need_water?'bad':completed?'ok':''}">${d.need_water?'准备加水':completed?`已校准 ${completed} 轮`:'学习中'}</span></div>
      <p class="fountain-lead">${esc(d.need_water?d.reminder.message:d.cycle_closed?'本轮残水已保存，下一轮会继续积累校准数据。':d.has_refill?'追加加水会计入本轮，喝水记录和周期起点继续保留。':'首次不必加满，记录新增水量和加水前的实际残水。')}</p>
      <div class="fountain-stats"><div><span>累计新增</span><strong>${added||(!d.has_refill?'—':d.last_refill_ml)}<small> ml</small></strong><p>${current.length?`${current.length} 次加水`:'尚未记录加水'}</p></div><div><span>确认喝水</span><strong>${d.drinks_since_refill||0}<small> 次</small></strong><p>${d.calibrated?`约 ${d.ml_per_drink.toFixed(1)} ml / 次`:'每次水量待校准'}</p></div><div><span>停水残水位</span><strong>${d.reminder?.stop_residual_ml??'—'}<small> ml</small></strong><p>${d.reminder?.stop_residual_ml==null?'等待实测反馈':'已从反馈中学习'}</p></div></div>
      <div class="fountain-primary-action"><button class="btn" onclick="dispRefill()">${I.drop} ${d.has_refill&&!d.cycle_closed?'追加加水':'记录加水'} ${I.arrow}</button><span>${d.has_refill?`本轮开始于 ${waterDate(d.last_refill_ts)}`:'支持 950 ml + 1050 ml 分次记录'}</span></div>
      ${d.has_refill&&d.options?.baseline_known===false?'<div class="fountain-note">'+I.info+'<span>初始残水尚未测量，本轮水量为估值。下一轮可从实测残水建立完整基线。</span></div>':''}
    </div>
  </div>
  <div class="fountain-shortcuts"><button onclick="dispObserve()" ${!d.has_refill?'disabled':''}><span class="fountain-action-icon">${I.drop}</span><span><strong>校正剩余水量</strong><small>测量一次，让当前估值更准确</small></span>${I.right}</button><button onclick="finishWaterCycle()" ${!d.has_refill||d.cycle_closed?'disabled':''}><span class="fountain-action-icon amber">${I.check}</span><span><strong>停水了，反馈残水</strong><small>记录真实余量，完成本轮周期</small></span>${I.right}</button></div>
  <section class="fountain-learning"><div class="fountain-section-head"><div><span class="fountain-kicker">周期学习</span><h3>每一轮，积累一点准确度</h3></div><span class="fountain-muted">${completed} 轮有效校准</span></div><ol class="fountain-steps">${steps.map((s,i)=>`<li class="${s[2]?'complete':''}"><span>${s[2]?I.check:i+1}</span><div><strong>${s[0]}</strong><small>${s[1]}</small></div></li>`).join('')}</ol></section>`;
  renderWaterSettings(d);renderDispenserRecords(d);
}
function renderWaterSettings(d){
  const box=$('#waterSettings');if(!box)return;
  const r=d.reminder||{};
  box.innerHTML=`<div class="fountain-settings-grid">
    <section class="fountain-setting"><div class="fountain-setting-title"><span class="fountain-action-icon amber">${I.info}</span><span class="tag ${r.ready?'ok':''}">${r.ready?'可估算提醒':'待周期校准'}</span></div><h3>水少提醒</h3><p>先观察设备何时提醒，再测量剩余水量。系统会学习您这台饮水机的提醒水位。</p><div class="fountain-setting-value"><span>已记录提醒水位</span><strong>${r.low_ml==null?'尚未设置':`${r.low_ml} ml`}</strong></div><div class="fountain-setting-status"><i class="${r.enabled&&r.ready?'active':''}"></i>${r.enabled?(r.ready?'软件估算提醒已开启':'软件提醒等待完整周期'):'软件估算提醒未开启'}</div><div class="fountain-setting-buttons"><button class="btn" onclick="recordWaterSignal('low')" ${!d.has_refill||d.cycle_closed?'disabled':''}>记录水少时的水位</button><button class="btn ghost" onclick="openWaterReminder()">提醒与容量设置</button></div></section>
    <section class="fountain-setting"><div class="fountain-setting-title"><span class="fountain-action-icon">${I.check}</span><span class="tag">${r.complete_cycles||0} 轮校准</span></div><h3>停水与残水</h3><p>停止出水不代表水已耗尽。实测停水时剩下的水，帮助平台学习工作水位与加水周期。</p><div class="fountain-setting-value"><span>已学习停水残水位</span><strong>${r.stop_residual_ml==null?'等待实测':`${r.stop_residual_ml} ml`}</strong></div><div class="fountain-setting-status">出水停止与恢复由摄像头自动观察</div><div class="fountain-setting-buttons"><button class="btn ghost" onclick="finishWaterCycle()" ${!d.has_refill||d.cycle_closed?'disabled':''}>反馈停水残水</button><button class="fountain-text-button" onclick="waterTab('water');$('#flowPanel').scrollIntoView({behavior:reduceMotion?'instant':'smooth',block:'center'})">查看出水监测 ${I.right}</button></div></section>
    <section class="fountain-setting"><div class="fountain-setting-title"><span class="fountain-action-icon">${I.clock}</span><span class="tag ${d.need_filter?'bad':''}">${d.need_filter?'需要更换':'日常维护'}</span></div><h3>滤芯维护</h3><p>记录实际更换日期，按您设置的周期提醒下一次更换。</p><div class="fountain-setting-value"><span>距离建议更换</span><strong>${!d.has_filter?'尚未记录':d.need_filter?'已到更换时间':`约 ${Math.max(0,Math.round(d.filter_days_left||0))} 天`}</strong></div><div class="fountain-setting-status">当前更换周期 · ${d.filter_cycle_days} 天</div><div class="fountain-setting-buttons"><button class="btn ghost" onclick="dispFilter()">记录更换滤芯</button><button class="fountain-text-button" onclick="dispCycle()">调整周期 ${I.right}</button></div></section>
    <section class="fountain-setting"><div class="fountain-setting-title"><span class="fountain-action-icon">${I.info}</span><span class="tag">邮件通知</span></div><h3>离开页面，也能收到提醒</h3><p>确切喝水、疑似停水和已校准的水少提醒，通过邮件及时通知。</p><div class="fountain-setting-value"><span>每日汇总</span><strong>23:00</strong></div><div class="fountain-setting-status">周日 23:00 改发每周汇总 · 需配置邮箱</div><div class="fountain-setting-buttons"><button class="btn ghost" onclick="openMailSetup()">配置邮件提醒</button></div></section>
  </div>`;
}
let dispenserRecordFilter='all';
function filterDispenserRecords(kind){dispenserRecordFilter=kind;if(dispData)renderDispenserRecords(dispData);}
function renderDispenserRecords(d){
  const box=$('#dispenserRecords');if(!box)return;
  const records=[];
  for(const a of d.additions||[])records.push({ts:a.ts,title:`加水 +${a.ml} ml`,detail:'本轮新增水量，支持分次累加',kind:'water'});
  for(const m of d.readings||[])if(m.kind==='measurement')records.push({ts:m.ts,title:`实测剩余 ${m.ml} ml`,detail:'用户测量水位，校正当前估值',kind:'water'});
  for(const v of d.signals||[])if(v.kind==='low')records.push({ts:v.ts,title:`水少提醒 · 剩余 ${v.remaining_ml} ml`,detail:'用户观察设备提醒后实测，用于学习提醒水位',kind:'cycle'});
  for(const c of d.cycles||[])records.push({ts:c.end,title:`本轮结束 · 残水 ${c.remaining_ml??'未测量'} ml`,detail:`确认喝水 ${c.drinks} 次 · ${c.used?'已用于校准':c.baseline_known?'等待补全喝水复核':'本轮初始残水不确定'}`,kind:'cycle'});
  for(const i of d.flow?.incidents||[])records.push({ts:i.end||i.start,title:i.end?'出水已恢复':'检测到疑似停水',detail:`${waterDate(i.start)} — ${i.end?waterDate(i.end):'持续观察中'}${i.end?' · 持续 '+waterDuration(i.end-i.start):''}`,kind:i.end?'recovery':'fault'});
  records.sort((a,b)=>b.ts-a.ts);
  const filtered=records.filter(r=>dispenserRecordFilter==='all'||(dispenserRecordFilter==='flow'?['fault','recovery'].includes(r.kind):r.kind===dispenserRecordFilter));
  const groups=new Map();for(const r of filtered.slice(0,60)){const day=new Date(r.ts*1000).toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'});if(!groups.has(day))groups.set(day,[]);groups.get(day).push(r);}
  box.innerHTML=`<div class="fountain-records-header"><div><span class="fountain-kicker">设备日志</span><h2>每一次变化，都有迹可循</h2><p>加水、实测水位、周期反馈与出水异常</p></div><span class="fountain-record-count">${records.length}<small> 条记录</small></span></div><div class="fountain-record-filters" aria-label="筛选设备记录">${[['all','全部'],['water','加水与水位'],['cycle','周期与提醒'],['flow','停水与恢复']].map(([k,t])=>`<button class="${dispenserRecordFilter===k?'on':''}" aria-pressed="${dispenserRecordFilter===k}" onclick="filterDispenserRecords('${k}')">${t}</button>`).join('')}</div>
    ${filtered.length?[...groups].map(([day,items])=>`<section class="fountain-record-day"><h3>${day}</h3><div class="fountain-record-list">${items.map(r=>`<div class="fountain-record-row"><time>${new Date(r.ts*1000).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</time><span class="fountain-record-icon ${r.kind}">${r.kind==='water'?I.drop:r.kind==='fault'?I.info:I.check}</span><div><strong>${esc(r.title)}</strong><p>${esc(r.detail)}</p></div><span class="fountain-record-type">${({water:'水量记录',cycle:'周期学习',fault:'出水异常',recovery:'恢复记录'})[r.kind]}</span></div>`).join('')}</div></section>`).join(''):`<div class="fountain-record-empty">${I.clock}<h3>${records.length?'暂无这类记录':'从第一条加水记录开始'}</h3><p>${records.length?'其他类型的记录可以在上方切换查看。':'加水、测量残水和出水变化会自动汇集在这里。'}</p>${!records.length?'<button class="btn ghost" onclick="dispRefill()">记录第一次加水</button>':''}</div>`}`;
}
function waterDate(ts){return new Date(ts*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});}
function waterDuration(seconds){return seconds<60?Math.round(seconds)+' 秒':seconds<3600?Math.round(seconds/60)+' 分钟':(seconds/3600).toFixed(1)+' 小时';}
function waterTab(name){for(const t of ['water','settings','records']){$('#water-'+t).hidden=t!==name;}$$('.water-tab').forEach(b=>b.classList.toggle('on',b.dataset.tab===name));}
/* 加水模态：预设芯片 + 滑杆 + 数字输入 */
function dispRefill(){
  const first=!dispData?.has_refill||dispData?.cycle_closed;
  openModal(`<h3>${first?'记录本次加水':'追加加水'}</h3><p class="msub">填写这次新倒进去的水量。${first?'首次不必加满；记录初始残水后，停水时再测残水，就能完成一轮校准。':'不会重置本轮喝水记录；新增量会累加到同一轮。'}</p>
    <label class="water-input-label">本次新增水量<input type="number" id="rfNum" min="1" max="20000" step="1" placeholder="例如 950" autofocus> ml</label>
    <div class="presets">${[250,500,950,1050].map(v=>`<button onclick="$('#rfNum').value=${v}">+${v} ml</button>`).join('')}</div>
    <label class="water-input-label">${first?'加水前的实测残水':'加水前残水（可选，用于校正估值）'}<input type="number" id="rfInitial" min="0" max="20000" step="1" placeholder="${first?'已倒空填 0；没测量可留空':'未测量留空，按当前估值累计'}"> ml</label>
    <p class="msub">残水未测量时不当作实测值。比如先加 950 ml，再加 1050 ml，本轮累计新增 2000 ml；实际剩余还会扣除期间消耗。</p>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="rfGo()">保存本次加水</button></div>`);
}
async function rfGo(){
  const ml=Number($('#rfNum').value),initial=$('#rfInitial').value;
  if(!(ml>0)){toast('请输入本次新增水量','err');return;}
  const body={added_ml:ml};if(initial!=='')body.initial_remaining_ml=Number(initial);
  try{const r=await fetch('/api/dispenser/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
    if(!r.ok){toast(d.detail,'err');return;}closeModal();loadDispenser();toast(`已追加 ${ml} ml，保留本轮喝水记录`);
  }catch(e){toast('加水记录失败','err');}
}
function dispObserve(){
  openModal(`<h3>记录实测剩余水量</h3><p class="msub">读取刻度或称重后填写；总消耗包含蒸发和泼洒，系统不会自动认作猫喝掉的量。</p>
    <label class="water-input-label">实测剩余水量（ml）<input id="waterMeasured" type="number" min="0" max="20000" placeholder="请填写测量结果" aria-label="实测剩余毫升"></label>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveWaterObservation()">保存水位</button></div>`);
}
async function saveWaterObservation(){
  if($('#waterMeasured').value===''){toast('请填写实际测量的水量','err');return;}
  const ml=Number($('#waterMeasured').value);
  if(!Number.isFinite(ml)||ml<0){toast('请输入有效水量','err');return;}
  try{const response=await fetch('/api/dispenser/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ml})});
    const d=await response.json();if(!response.ok){toast(d.detail||'记录失败','err');return;}
    closeModal();dispData=d;renderDisp(d);renderDispMini(d);toast('实测水位已更新');
  }catch(e){toast('记录水位失败','err');}
}
function finishWaterCycle(){
  openModal(`<h3>反馈本轮实际用水</h3><p class="msub">本轮水量记录 ${dispData.last_refill_ml} ml，已确认 ${dispData.drinks_since_refill} 次。饮水机不出水时仍可能有残水，请倒出后用量杯或称重测量。保存后下次加水将开启新一轮。</p>
    <div class="mail-fields"><label class="mail-wide">停水时的实测残水（ml）<input id="cycleRemaining" type="number" min="0" placeholder="请填写测量结果"></label><label class="mail-wide">已知蒸发、泼洒等损耗（ml）<input id="cycleLoss" type="number" min="0" value="0"></label></div>
    <p class="lab-note">若未填写其他损耗，水量下降会全部归入饮水估值。多只宠物共用时也是合计估值。</p>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveWaterCycle()">确认周期反馈</button></div>`);
}
async function saveWaterCycle(){
  if($('#cycleRemaining').value===''){toast('请先测量残水并填写，不能默认按零计算','err');return;}
  try{const r=await fetch('/api/dispenser/cycle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({remaining_ml:Number($('#cycleRemaining').value),other_loss_ml:Number($('#cycleLoss').value),stopped_confirmed:true})});
    const d=await r.json();if(!r.ok){toast(d.detail,'err');return;}closeModal();dispData=d;renderDisp(d);renderDispMini(d);loadDispenser();loadHealth();
    toast(d.calibrated?`校准更新：约 ${d.ml_per_drink.toFixed(1)} ml/次`:'周期已保存，请先复核遗漏的喝水候选');
  }catch(e){toast('周期反馈失败','err');}
}
function dispFilter(){
  openModal(`<h3>记录更换滤芯</h3><p class="msub">确认后将从今天重新倒计时（周期 ${dispData?dispData.filter_cycle_days:30} 天）。</p>
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
let clipsRequest=null;
async function fetchClips(){
  if(clipsRequest)return clipsRequest;
  clipsRequest=(async()=>{
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    try{
      const response=await fetch('/api/clips',{signal:controller.signal});
      if(!response.ok)throw new Error('视频列表加载失败');
      clipsData=await response.json();
      const pending=clipsData.clips.filter(needsReview).length;
      $('#navReviewCount').textContent=pending;
      $('#navReviewCount').hidden=!pending;
      $('#modelPending').textContent=`待复核 ${pending}`;
      $('#reviewRefresh').textContent='刷新';
      return clipsData;
    }finally{clearTimeout(timer);}
  })();
  try{return await clipsRequest;}finally{clipsRequest=null;}
}
function clipLoadError(selector,retry){
  $(selector).innerHTML=`<div class="empty">视频列表暂时加载失败 <button class="btn ghost" onclick="${retry}()">重试</button></div>`;
}
// 只为即将进入视口的录像请求封面；不预加载视频内容。
const posterObserver='IntersectionObserver' in window?new IntersectionObserver(entries=>{
  entries.forEach(({target,isIntersecting})=>{if(isIntersecting){target.poster=target.dataset.poster;delete target.dataset.poster;posterObserver.unobserve(target);}});
},{rootMargin:'200px'}):null;
function observePosters(root){
  if(root.nodeType!==1&&root!==document)return;
  const videos=[...(root.matches?.('video[data-poster]')?[root]:[]),...root.querySelectorAll('video[data-poster]')];
  videos.forEach(video=>{if(posterObserver)posterObserver.observe(video);else{video.poster=video.dataset.poster;delete video.dataset.poster;}});
}
new MutationObserver(records=>records.forEach(record=>{
  record.removedNodes.forEach(node=>{if(node.nodeType===1){posterObserver?.unobserve(node);node.querySelectorAll('video').forEach(v=>posterObserver?.unobserve(v));}});
  record.addedNodes.forEach(observePosters);
})).observe(document.body,{childList:true,subtree:true});
observePosters(document);
function isReviewCandidate(n){return clipsData.review_candidates?.[n]!==false;}
async function checkClipUpdates(){
  const response=await fetch('/api/clips/summary'+(labState.snapshot?'?since='+encodeURIComponent(labState.snapshot):''));
  if(!response.ok)return;
  const summary=await response.json(),pending=summary.pending_count??summary.pending_names.length;
  $('#navReviewCount').textContent=pending;$('#navReviewCount').hidden=!pending;
  $('#modelPending').textContent=`待复核 ${pending}`;
  const known=new Set(clipsData?.clips.filter(isReviewCandidate)||[]);
  const count=summary.new_count??summary.review_names.filter(n=>!known.has(n)).length;
  $('#reviewRefresh').textContent=count?`刷新 · ${count} 个新视频`:'刷新';
}
async function promoteReplay(clip,button){
  const response=await fetch(`/api/clips/${encodeURIComponent(clip)}/review`,{method:'POST'});
  if(!response.ok){toast('操作失败，请重试','err');return;}
  button.disabled=true;button.textContent='已加入待复核';checkClipUpdates();
}
function clipTime(name){const m=String(name).match(/clip_(\d+)/);return m?new Date(parseInt(m[1],10)):null;}
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
  try{await fetchClips();renderRecords();}catch(e){clipLoadError('#recGrid','loadRecords');}
}
function recItems(){
  const lab=clipsData.labels||{};
  const names=new Set(clipsData.clips);
  return (clipsData.drinking_records||clipsData.clips).filter(n=>names.has(n)&&lab[n]===true);
}
function recordTime(name){
  const seconds=clipsData.sessions?.[name]?.presence_start;
  return Number.isFinite(seconds)&&seconds>0?new Date(seconds*1000):clipTime(name);
}
function focusDrinkRecord(name){
  const card=[...$('#recGrid').querySelectorAll('.clip')].find(el=>el.dataset.name===name);
  if(card){card.scrollIntoView({behavior:reduceMotion?'instant':'smooth',block:'center'});card.querySelector('video')?.focus({preventScroll:true});}
}
function recCard(n,i){
  const dur=(clipsData.durations||{})[n],t=recordTime(n),en=esc(n),jn=JSON.stringify(n);
  return `<div class="clip record-card" data-name="${en}" style="--i:${Math.min(i,10)}">
    <div class="record-time-head"><div><span>${clipsData.sessions?.[n]?.presence_start?'到碗时间':'记录时间'}</span><time datetime="${t?t.toISOString():''}">${t?hhmmss(t):'时间未知'}</time></div><span class="record-date">${t?t.toLocaleDateString('zh-CN'):''}</span></div>
    <div class="thumb">
      <video class="inline-clip" src="/clips/${encodeURIComponent(n)}" data-poster="/clips/${encodeURIComponent(n)}/thumb.jpg" controls preload="none" playsinline></video>
      ${dur?`<span class="dur">录像 ${dur.toFixed(1)} 秒</span>`:''}</div>
    <div class="cmeta"><div class="row">
      <span class="record-confirmed">已确认喝水</span>
      <a class="dlbtn" href="/clips/${encodeURIComponent(n)}" download onclick='this.href=dlUrl(${jn})'>${I.dl}下载</a>
    </div><div class="review-duration"><div><span>${clipsData.meta?.[n]?.drinking_seconds!=null?'喝水时长 · 人工记录':'估算喝水时长'}</span><strong>${clipsData.meta?.[n]?.effective_drinking_seconds??'—'}<small>秒</small></strong></div><button class="review-adjust" onclick='labelDuration(${jn})'>调整时长</button></div><p class="lab-note">${({human:'人工确认',ai:'外部模型确认',local:'模型自动确认'})[clipsData.meta?.[n]?.source]||'已确认'} · ${clipsData.meta?.[n]?.drinking_seconds!=null?'使用人工时长':'按录像时长的 80% 估算'}</p><button class="record-correction" onclick='correctDrinkRecord(${jn})'>修改这条记录</button></div></div>`;
}
function renderRecords(){
  if(!clipsData)return;
  const box=$('#recGrid'),items=recItems();
  const cnt=$('#recCount');
  cnt.style.display='inline-flex';cnt.innerHTML=`<i></i>共 ${items.length} 次确认喝水 · 相邻片段不足 1 分钟合并计数`;
  if(!items.length){
    box.innerHTML=`<div class="empty" style="grid-column:1/-1"><span class="eico">🐟</span>
      还没有确认喝水的回放～<br>喵喵来水碗喝水会自动录像，AI 判定「真喝水」后就会出现在这里。</div>`;
    return;
  }
  const dayCounts={}; items.forEach(n=>{const k=dayKey(recordTime(n));dayCounts[k]=(dayCounts[k]||0)+1;});
  const shown=items.slice(0,recPage*REC_PAGE);
  let html='',lastDay=null;
  shown.forEach((n,i)=>{const dt=recordTime(n),k=dayKey(dt);
    if(k!==lastDay){lastDay=k;
      html+=`<div class="grp-head"><span class="grp-dot"></span>${dayLabel(dt)}<span class="grp-n">${dayCounts[k]} 次</span><span class="record-day-date">${dt?dt.toLocaleDateString('zh-CN'):''} · 最新在前</span><div class="record-day-times" aria-label="当天喝水时间点">${shown.filter(name=>dayKey(recordTime(name))===k).map(name=>`<button onclick='focusDrinkRecord(${JSON.stringify(name)})'>${hhmmss(recordTime(name))}</button>`).join('')}</div></div>`;}
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
  const r=await (await fetch('/api/stats/trend?'+(typeof trendDays==='string'?'period='+trendDays:'days='+trendDays))).json();
  const pts=r.days||[],vals=pts.map(p=>p.count);
  const total=r.total??vals.reduce((a,b)=>a+b,0),days=pts.length;
  const avg=days?total/days:0,today=days?vals[vals.length-1]:0;
  const hero=$('#trendHero');
  hero.className='trend-hero';
  countUp($('#trToday'),today);
  $('#trStatus').textContent=today===0?'今天尚无确认记录':'已确认的饮水访问';
  const prev=r.prev_total||0;
  const deltaTxt=prev?`　·　环比 ${total>=prev?'▲':'▼'} ${Math.round(Math.abs(total-prev)/prev*100)}%`:'';
  $('#trSub').innerHTML=`日均 <b>${avg.toFixed(1)}</b> 次　·　${trendDays==='week'?'本周':trendDays==='month'?'本月':`近 ${days} 天`}共 <b>${total}</b> 次　·　活跃 <b>${r.active_days??0}/${days}</b> 天${deltaTxt}`;
  const hourly=r.hourly||[],hmax=Math.max(1,...hourly);
  $('#hourHeat').innerHTML=hourly.map((v,h)=>
    `<div class="hcell" style="background:${heatColor(v,hmax)};--i:${h}" title="${h}:00–${(h+1)%24}:00 · ${v} 次"></div>`).join('');
  const pk=Math.max(0,...hourly);
  $('#hourPeak').textContent=pk>0?`最爱 ${hourly.indexOf(pk)}:00 前后`:'';
  renderDailyTrend(pts);
  const cmax=Math.max(1,...vals);
  $('#calHeat').innerHTML=pts.map((p,i)=>
    `<div class="ccell" style="background:${heatColor(p.count,cmax)};--i:${i}" title="${p.date} · ${p.count} 次"><span>${p.count||''}</span></div>`).join('');
  $('#calNote').textContent=`单日最多 ${cmax} 次`;
  $$('#pg-trend .callegend .cl').forEach(el=>{el.style.background=heatColor(+el.dataset.l,4);});
}
function renderDailyTrend(points){
  const w=600,h=190,left=34,bottom=157,top=16,n=points.length;
  if(!n)return;
  const values=points.map(p=>p.count),peak=Math.max(1,...values),step=(w-left-16)/n;
  const average=values.map((v,i)=>{const a=values.slice(Math.max(0,i-6),i+1);return a.reduce((x,y)=>x+y,0)/a.length;});
  const axis=`<line x1="${left}" y1="${bottom}" x2="${w-16}" y2="${bottom}" stroke="currentColor" opacity=".3"/>`;
  const labels=points.map((p,i)=>i%Math.max(1,Math.ceil(n/10))===0||i===n-1?`<text x="${left+step*(i+.5)}" y="181" text-anchor="middle" fill="currentColor" font-size="12">${esc(p.date)}</text>`:'').join('');
  const ticks=[0,peak].map(v=>`<text x="32" y="${bottom-(bottom-top)*v/peak+4}" text-anchor="end" fill="currentColor" font-size="12">${v}</text>`).join('');
  const svg=content=>`<svg viewBox="0 0 ${w} ${h}" role="img" class="trend-chart-svg" aria-label="${esc(points[0].date)}至${esc(points[n-1].date)}饮水次数">${axis}${ticks}${labels}${content}</svg>`;
  $('#trendBars').innerHTML=svg(points.map((p,i)=>`<rect x="${left+step*i+step*.15}" y="${bottom-(bottom-top)*p.count/peak}" width="${step*.7}" height="${(bottom-top)*p.count/peak}" rx="3" fill="var(--aqua,#28c9dd)"><title>${esc(p.date)}：${p.count} 次</title></rect>`).join(''));
  const coords=average.map((v,i)=>`${left+step*(i+.5)},${bottom-(bottom-top)*v/peak}`);
  $('#trendLine').innerHTML=svg(`<polyline points="${coords.join(' ')}" fill="none" stroke="#2ad4b5" stroke-width="3"/>`+average.map((v,i)=>`<circle cx="${left+step*(i+.5)}" cy="${bottom-(bottom-top)*v/peak}" r="4" fill="#2ad4b5"><title>${esc(points[i].date)}：近 ${Math.min(i+1,7)} 天日均 ${v.toFixed(1)} 次</title></circle>`).join(''));
}
function setRange(d,btn){trendDays=d;for(const b of $('#rangeCtl').children)b.classList.toggle('on',b===btn);renderTrend();}

/* ---------- 标注工作台 ---------- */
let labelFilter='review', labelPage=1;
const LAB_PAGE=12;
const labState={names:[],cursor:null,snapshot:null,counts:null,generation:0,controller:null,busy:false};
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
    confidence:m.source==='local'?m.confidence:(m.is_drinking?m.confidence:1-m.confidence),
    by:m.source
  };
  return null;
}
function needsReview(n){
  if(!isReviewCandidate(n))return false;
  const m=(clipsData.meta||{})[n];
  if(m&&m.source==='human')return false;
  // 与分页、侧栏一致：尚未经人工确认的候选均可复核，包括机器高置信标签。
  return true;
}
function modelTag(n){
  if(clipsData.candidate_only?.[n])return `<span class="mpred n">弱光轮廓候选 · 请确认喝水动作</span>`;
  const p=modelDetail(n);
  if(!p)return `<span class="tag mute"><i></i>模型未判断</span>`;
  const uncertain=needsReview(n)?' · 待复核':'';
  return `<span class="mpred ${p.drinking?'y':'n'}" title="${esc(p.by||'模型')} · 模型参考，需人工确认">模型参考：${p.drinking?'疑似喝水':'未发现喝水'}${uncertain}</span>`;
}
async function fetchLabelPage(cursor,generation){
  const controller=new AbortController();labState.controller=controller;
  const timer=setTimeout(()=>controller.abort(),15000);
  const params=new URLSearchParams({filter:labelFilter,limit:String(LAB_PAGE)});
  if(cursor)params.set('cursor',cursor);
  if(labState.snapshot)params.set('snapshot',labState.snapshot);
  try{
    const response=await fetch('/api/clips/page?'+params,{signal:controller.signal});
    if(!response.ok)throw Error('视频列表加载失败');
    const data=await response.json();
    return generation===labState.generation?data:null;
  }finally{clearTimeout(timer);if(labState.controller===controller)labState.controller=null;}
}
function mergeLabelPage(data){
  if(!clipsData)clipsData={clips:[]};
  for(const key of ['labels','durations','meta','notes','review_candidates','candidate_only','water_review','predictions','prediction_details']){
    clipsData[key]={...clipsData[key],...data[key]};
  }
  clipsData.clips=[...new Set([...clipsData.clips,...data.clips])];
  for(const key of ['max_clips','retention_days','recording_fps'])clipsData[key]=data[key];
  labState.names=[...new Set([...labState.names,...data.clips])];
  labState.cursor=data.next_cursor;labState.snapshot=data.snapshot;labState.counts=data.counts;
}
async function loadLabelClips(resetSnapshot=true){
  labState.controller?.abort();
  const generation=++labState.generation;
  labState.busy=true;labState.names=[];labState.cursor=null;
  if(resetSnapshot)labState.snapshot=null;
  $('#labelClips').innerHTML=skeletons(8);
  try{
    const data=await fetchLabelPage(null,generation);if(!data)return;
    mergeLabelPage(data);
    $('#labCap').textContent=`画面变化即录制 · ${data.recording_fps} fps · 滚动保留 ${data.retention_days||3} 天`;
    $('#reviewRefresh').textContent='刷新';
    updateLabelChipCounts();renderLabelClips();
  }catch(e){if(generation===labState.generation)clipLoadError('#labelClips','loadLabelClips');}
  finally{if(generation===labState.generation)labState.busy=false;}
}
async function loadMoreLabelClips(button){
  if(labState.busy||!labState.cursor)return;
  labState.busy=true;button.disabled=true;button.textContent='加载中…';
  const generation=labState.generation;
  try{
    const data=await fetchLabelPage(labState.cursor,generation);if(!data)return;
    const known=new Set(labState.names);mergeLabelPage(data);
    // 只追加新卡片，已播放或正在复核的卡片不重建。
    button.insertAdjacentHTML('beforebegin',data.clips.filter(n=>!known.has(n)).map(labelCard).join(''));
    button.remove();appendLabelMore();updateLabelChipCounts();
  }catch(e){if(generation===labState.generation){button.disabled=false;button.textContent='加载失败，点击重试';}}
  finally{if(generation===labState.generation)labState.busy=false;}
}
function appendLabelMore(){
  if(labState.cursor)$('#labelClips').insertAdjacentHTML('beforeend','<button class="more" onclick="loadMoreLabelClips(this)">加载更多</button>');
}
async function refreshLabelCounts(){
  const generation=labState.generation,revision=reviewMutation;
  const params=new URLSearchParams({filter:labelFilter,limit:'1'});
  if(labState.snapshot)params.set('snapshot',labState.snapshot);
  try{const response=await fetch('/api/clips/page?'+params);if(!response.ok)return;
    const data=await response.json();if(generation!==labState.generation||revision!==reviewMutation)return;
    labState.counts=data.counts;updateLabelChipCounts();
  }catch(e){}
}
function updateLabelChipCounts(){
  const counts=labState.counts||{};
  const map={all:`全部 ${counts.all??0}`,review:`待复核 ${counts.review??0}`,yes:`模型：喝水 ${counts.yes??0}`,no:`模型：没喝 ${counts.no??0}`};
  $$('#labelFilters .fchip').forEach(b=>{if(map[b.dataset.f])b.textContent=map[b.dataset.f];});
}
function labelItems(){
  return labState.names.filter(isReviewCandidate).filter(n=>{const p=modelDetail(n);
    return labelFilter==='all'||(labelFilter==='review'&&needsReview(n))||
      (labelFilter==='mistakes'&&p&&(clipsData.meta[n]?.source==='human')&&p.drinking!==(clipsData.meta[n]?.action_drinking??clipsData.labels[n]))||(labelFilter==='yes'&&p&&p.drinking)||(labelFilter==='no'&&p&&!p.drinking);});
}
function labelCard(n,i){
  const lab=clipsData.labels||{},dur=clipsData.durations||{};
  const v=lab[n],d=dur[n],en=esc(n),jn=JSON.stringify(n),t=clipTime(n);
  const meta=clipsData.meta?.[n]||{},seconds=meta.effective_drinking_seconds??(d?Math.round(d*.8*10)/10:null);
  return `<div class="clip review-card" data-name="${en}">
    <div class="thumb">
      <video class="inline-clip" src="/clips/${encodeURIComponent(n)}" data-poster="/clips/${encodeURIComponent(n)}/thumb.jpg" controls preload="none" playsinline></video>
      ${d?`<span class="dur">录像 ${d.toFixed(1)} 秒</span>`:''}</div>
    <div class="review-side">
      <div class="review-head"><div class="review-title">${t?'发生于 '+hhmmss(t):'候选视频'}</div>${meta.outcome==='dry_lick'?'<span class="tag warn">舔水未喝到</span>':clipsData.meta?.[n]?.source==='local'?'<span class="tag mute"><i></i>模型自动确认</span>':statusTag(v)}</div>
      <div class="row">${modelTag(n)}</div>
      <div class="seg2">
        <button class="yes ${v===true?'on':''}" onclick='fb(${jn},true)'>${I.check}喝了</button>
        <button class="no ${v===false?'on':''}" onclick='fb(${jn},false)'>${I.x}没喝</button>
      </div>
      <button class="review-reason" onclick='explainClip(${jn})'>说明原因（可选）</button>
      ${v===true?`<div class="review-duration"><div><span>${meta.drinking_seconds!=null?'喝水时长 · 人工记录':'估算喝水时长'}</span><strong>${seconds??'—'}<small>秒</small></strong></div><button class="review-adjust" onclick='labelDuration(${jn})'>调整时长</button></div>`:''}
      <p class="water-data-note review-note">${esc((clipsData.notes?.[n]||(meta.source==='human'?meta.reason:'')||'').slice(0,120))}</p>
      <div class="review-foot"><span class="review-foot-note">${v===true?(meta.drinking_seconds!=null?'按人工记录估算用水':'按录像时长的 80% 估算'):v===false?(meta.outcome==='dry_lick'?'有舔水动作，未喝到水 · 动作正样本':'无喝水动作 · 动作负样本'):'确认后计入饮水统计'}</span><a class="dlbtn" href="/clips/${encodeURIComponent(n)}" download onclick='this.href=dlUrl(${jn})'>${I.dl}<span>下载</span></a></div>
    </div></div>`;
}
function renderLabelClips(){
  const box=$('#labelClips'); if(!box||!clipsData)return;
  const items=labelItems();
  if(!items.length){
    const message=labState.counts?.all===0?'暂无候选录像，录制后会出现在这里。':labelFilter==='review'?'当前没有待复核视频，可切换「全部」查看已复核录像。':'当前筛选下没有视频，可切换其他筛选查看。';
    box.innerHTML=`<div class="empty" style="grid-column:1/-1">${message}</div>`;appendLabelMore();return;
  }
  box.innerHTML=items.map(labelCard).join('');
  appendLabelMore();
}
function setLabelFilter(f,btn){labelFilter=f;
  for(const b of $('#labelFilters').querySelectorAll('.fchip'))b.classList.toggle('on',b===btn);
  loadLabelClips(false);
}
function labelDuration(clip){
  const max=clipsData.durations[clip]||0;
  openModal(`<h3>记录真正舔水的时长</h3><video src="/clips/${encodeURIComponent(clip)}" controls style="width:100%"></video>
    <p class="msub">只填实际舔水动作的总秒数，排除靠近、嗅闻和离开；录像共 ${max} 秒。将用于按时长校准估值。</p>
    <input id="drinkDuration" type="number" min="0.1" step="0.1" max="${max}" value="${clipsData.meta?.[clip]?.effective_drinking_seconds??Math.round(max*.8*10)/10}">
    <div class="modal-acts"><button class="btn" onclick='saveDrinkDuration(${JSON.stringify(clip)})'>保存时长</button></div>`);
}
async function saveDrinkDuration(clip){
  const seconds=Number($('#drinkDuration').value);
  const r=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,is_drinking:true,drinking_seconds:seconds})});
  const d=await r.json();if(!r.ok){toast(d.detail,'err');return;}closeModal();await loadLabelClips(false);loadDispenser();toast('喝水动作时长已记录');
}
// 连续复核只合并刷新计数，不为每次点击重拉总览、周趋势等隐藏页面数据。
let reviewMutation=0,reviewRefreshTimer=null,reviewRefreshRunning=false,reviewRefreshDirty=false;
function scheduleReviewRefresh(){
  reviewRefreshDirty=true;
  if(reviewRefreshTimer||reviewRefreshRunning)return;
  reviewRefreshTimer=setTimeout(async()=>{
    reviewRefreshTimer=null;reviewRefreshRunning=true;reviewRefreshDirty=false;
    try{await Promise.allSettled([refreshLabelCounts(),checkClipUpdates()]);}
    finally{reviewRefreshRunning=false;if(reviewRefreshDirty)scheduleReviewRefresh();}
  },400);
}
async function fb(clip,is){
  const card=$(`#labelClips .clip[data-name="${CSS.escape(clip)}"]`);
  card?.querySelectorAll('.seg2 button').forEach(b=>b.disabled=true);
  try{
    const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,is_drinking:is})});
    const result=await response.json();
    if(!response.ok)throw Error(result.detail||'保存失败');
    const wasPending=needsReview(clip);
    reviewMutation++;
    clipsData.labels[clip]=is;clipsData.meta[clip]=result.meta;
    if(wasPending&&labState.counts){labState.counts.review=Math.max(0,labState.counts.review-1);updateLabelChipCounts();}
    if(card){
      if(labelItems().includes(clip))card.outerHTML=labelCard(clip,0);
      else card.remove();
    }
    scheduleReviewRefresh();
    // 快复核完当前页时自动补下一页，保留其他视频的播放状态。
    const more=$('#labelClips .more');
    if(labelItems().length<=4&&more&&!labState.busy)loadMoreLabelClips(more);
    toast(is?'已确认：喝了':'已确认：没喝');
  }catch(error){toast(error.message||'保存失败，请重试','err');}
  finally{card?.querySelectorAll('.seg2 button').forEach(b=>b.disabled=false);}
}
function explainClip(clip){
  const reason=clipsData.notes?.[clip]||clipsData.meta?.[clip]?.reason||'';
  openModal(`<h3>说明原因（可选）</h3><p class="msub">补充说明不会改变“喝了／没喝”的判断，也可以留空。</p>
    <textarea id="clipReason" maxlength="500" rows="4" style="width:100%" placeholder="可选，例如路过、嗅闻或正在换水">${esc(reason)}</textarea>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick='saveClipReason(${JSON.stringify(clip)})'>保存说明</button></div>`);
}
async function saveClipReason(clip){
  const reason=$('#clipReason').value;
  const response=await fetch(`/api/clips/${encodeURIComponent(clip)}/note`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason})});
  if(!response.ok){toast('说明保存失败，请重试','err');return;}
  clipsData.notes??={};clipsData.notes[clip]=reason;
  const note=$(`#labelClips .clip[data-name="${CSS.escape(clip)}"] .review-note`);if(note)note.textContent=reason;
  closeModal();toast('说明已保存');
}

/* ---------- 模型训练 / 版本 ---------- */
function fmtAcc(a){return (typeof a==='number')?(a*100).toFixed(1)+'%':'—';}
function fmtPct(a){return (typeof a==='number')?(a*100).toFixed(0)+'%':'—';}
function fmtTime(ts){if(!ts)return '';const d=new Date(ts*1000);
  const p=n=>String(n).padStart(2,'0');return `${d.getMonth()+1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;}
async function pollTrain(){
  const active=$('#activeBox'),models=$('#modelList');
  if(active&&!active.innerHTML.trim())active.innerHTML='<div class="empty">正在读取模型状态…</div>';
  if(models&&!models.innerHTML.trim())models.innerHTML='<div class="empty">正在读取模型版本…</div>';
  try{
    const response=await fetch('/api/train_video/status',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const s=await response.json();
    const ls=s.label_states||{labeled:0,drinking:0,not_drinking:0,untrained:0,trained:0};
    $('#dsUn').textContent=(s.unlabeled??'–');
    $('#dsNew').textContent=ls.untrained; $('#dsTr').textContent=ls.trained;
    $('#dsBal').textContent=`${ls.drinking} / ${ls.not_drinking}`;
    const ready=ls.drinking>=4&&ls.not_drinking>=4;
    $('#studioReadiness').textContent=s.state==='running'?'正在训练':ready?'已有基础数据':'继续积累标注';
    $('#studioReadiness').classList.toggle('ok',ready);
    $('#studioTrainHint').textContent=ready?'训练耗时取决于数据量与电脑性能':'喝水与没喝动作，每类至少标注 4 段';
    if(s.dataset)$('#datasetStatus').textContent=`有效标注 ${s.dataset.labels} 段 · 原视频可用 ${s.dataset.available_clips} 段 · 缺失 ${s.dataset.missing_video} 段。${s.dataset.advice}`;
    renderTrainingAdvice(s);renderActive(s); renderModels(s);renderModelProgress(s);loadAIStatus();pollRecheck();
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
    headers:{'Content-Type':'application/json'},body:JSON.stringify({rebuild,backend:$("#trainBackend").value,auto_activate:$("#trainAutoActivate").checked})})).json();
  if(!r.started&&r.error){toast(r.error,'err');$('#trainVideoStatus').textContent=r.error;return;}
  toast('训练已启动','info');
  pollTrainVideo();
}
function renderTrainProg(s){
  const p=$('#trainVideoProg'); if(!p)return;
  const phase=s.phase,done=s.done||0,total=s.total||0,prog=s.progress||0;
  const lab=phase==='finetuning'?`微调主干 ${s.epoch}/${s.epochs} 轮 · 本轮 ${s.sample||0}/${s.samples||0} 段 · 验证损失 ${Number(s.val_loss||0).toFixed(3)}`:phase==='extracting'?`抽取特征 ${done}/${total} 段`
    :phase==='training'?'训练分类器…'
    :'加载 s3d 模型…（首次需下载约 30MB，请稍候）';
  const det=phase==='extracting'||phase==='finetuning';
  const pct=Math.round(prog*100);
  p.style.display='block';
  p.innerHTML=`<div class="pbar-track"><div class="pbar-fill ${det?'det':'indet'}" `+
    `style="${det?'width:'+pct+'%':''}"></div></div>`+
    `<div class="pbar-lab"><span>${lab}</span><span>${det?pct+'%':''}</span></div>`;
}
async function pollTrainVideo(){
  try{
    const s=await (await fetch('/api/train_video/status')).json();
    const btn=$('#trainVideoBtn'),st=$('#trainVideoStatus'),prog=$('#trainVideoProg');
    if(s.state==='disabled'){btn.disabled=true;st.textContent='演示模式未接训练器——用 python -m catcam 跑真实采集后这里可用';return;}
    if(s.state==='running'){
      btn.disabled=true;btn.textContent='训练进行中';st.textContent=s.detail||'训练中…';
      renderTrainProg(s);

    }else{
      if(trainVideoTimer){clearInterval(trainVideoTimer);trainVideoTimer=null;}
      if(prog)prog.style.display='none';
      btn.disabled=false;btn.textContent='开始训练';
      const r=s.result;
      if(s.state==='done'&&r){
        st.innerHTML=`完成 ${r.version} · <b>喝水召回 ${fmtPct(r.drinking_recall)}</b> `+
          `精确 ${fmtPct(r.drinking_precision)} <span style="color:var(--faint)">`+
          `(独立测试准确率 ${fmtPct(r.top1)}，多数类基线 ${fmtPct(r.naive_baseline)}；`+
          `样本 👍${r.counts.drinking}/👎${r.counts.not_drinking})</span>`+
          `<p>${esc(r.activation_note||'版本已保存，可在模型版本中启用。')}</p><p>${(r.warnings||[]).map(esc).join(' ')}</p><p>漏判喝水 ${r.confusion?.missed_drinking??'—'} 段 · 误判喝水 ${r.confusion?.false_drinking??'—'} 段 · ${r.auto_label_eligible?'满足自动判定评估条件':'保持测试模式，继续收集纠错数据'}</p>`;
      }else{st.textContent=s.detail||'';}
      renderTrainingAdvice(s);if(s.models){renderModels(s);renderActive(s);renderModelProgress(s);}
      if(s.state==='done')pollTrain();
    }
  }catch(e){}
}
function renderTrainingAdvice(s){
  const data=s.dataset||{},counts=data.video_counts||{drinking:0,not_drinking:0},fresh=data.new_video_counts||{drinking:0,not_drinking:0};
  const minimum=data.minimum_per_class||4,total=counts.drinking+counts.not_drinking,newTotal=fresh.drinking+fresh.not_drinking;
  const trained=(s.models||[]).some(m=>m.report),ready=Math.min(counts.drinking,counts.not_drinking)>=minimum;
  const missingYes=Math.max(0,minimum-counts.drinking),missingNo=Math.max(0,minimum-counts.not_drinking);
  const hr=s.hitrate,errors=hr?.total?hr.total-hr.correct:0;
  let title,message,action='去复核素材',target='label',state='collecting';
  if(s.state==='running'){title='正在训练下一版，等待完成即可';message='训练结束后会检查独立测试表现；满足启用条件才替换当前模型。';action='查看训练进度';target='train';state='running';}
  else if(!ready){title='还需要补一些素材，暂不建议训练';message=`可用喝水动作 ${counts.drinking} 段、没喝动作 ${counts.not_drinking} 段。${[missingYes?`再确认 ${missingYes} 段喝水动作`:'',missingNo?`再确认 ${missingNo} 段没喝动作`:''].filter(Boolean).join('，')}，即可启动第一次训练。`;}
  else if(!trained){title='现在可以训练你的第一个专属版本';message=`已准备 ${total} 段可用标注：喝水动作 ${counts.drinking} 段、没喝动作 ${counts.not_drinking} 段。当前预置模型还没用这些素材训练，建议现在开始轻量训练。`;action='去训练新版本';target='train';state='ready';}
  else if(!newTotal){title='暂时不用重训，先积累新的纠错素材';message=`已有 ${total} 段可用标注，目前没有未训练的新标注或改判。同一批数据重复训练通常不会更聪明。${errors?`当前复核中有 ${errors} 段不一致，可继续补充同类场景。`:''}`;}
  else if(newTotal>=10){title='素材已积累一批，建议训练下一版';message=`新增或改判 ${newTotal} 段：喝水动作 ${fresh.drinking} 段、没喝动作 ${fresh.not_drinking} 段。可以把这批变化交给下一版，再比较独立测试效果。`;action='去训练下一版';target='train';state='ready';}
  else{title=`已积累 ${newTotal} 段新素材，可再收集 ${10-newTotal} 段`;message=`建议每新增或改判约 10 段训练一版，这只是操作建议。若刚纠正了关键漏判、误判，或改变了猫咪、饮水机、机位，也可以现在训练。`;action='继续复核素材';}
  const rules=`<details><summary>什么时候需要再训练？</summary><ul><li>初次训练：原视频可用的喝水与没喝动作，每类至少 ${minimum} 段；只代表可以启动，准确度仍需验证。</li><li>后续训练：建议每新增或改判约 10 段训练一次；不同日期和靠近、嗅闻、真喝水等场景更有帮助。</li><li>刚纠正了重要漏判、误判，或改变了机位、饮水机、猫咪：补充并标注对应场景后，可提前训练。</li><li>没有新数据或改判：不必反复训练。“重新评估已有录像”只重新打分，不会训练新版本。</li><li>自动确认的片段不直接当训练素材；舔水但未喝到水仍可作为动作正样本。</li></ul></details>`;
  for(const box of $$('[data-training-advice]')){
    const opened=box.querySelector('details')?.open;
    box.className='studio-training-advice '+state;
    box.innerHTML=`<div class="studio-advice-main"><span class="studio-advice-icon">${state==='ready'?I.check:I.info}</span><div><span class="fountain-kicker">下一步建议</span><h3>${title}</h3><p>${esc(message)}</p></div><button class="btn ${state==='ready'?'':'ghost'}" onclick="setLab('${target}',$('#labTabs').children[${target==='train'?1:0}]);${target==='train'?"$('#trainVideoBtn').scrollIntoView({block:'center',behavior:reduceMotion?'instant':'smooth'})":''}">${action}</button></div><div class="studio-advice-foot"><span>可用标注 ${total} 段 · 新增或改判 ${newTotal} 段${data.missing_video?` · ${data.missing_video} 段缺少原视频` :''}</span>${rules}${ready&&trained&&newTotal>0&&newTotal<10?`<button class="studio-link" onclick="setLab('train',$('#labTabs').children[1]);$('#trainVideoBtn').scrollIntoView({block:'center'})">有关键纠错，现在去训练 →</button>`:''}</div>`;
    if(opened)box.querySelector('details').open=true;
  }
}
function renderActive(s){
  const box=$('#activeBox');if(!box)return;
  const m=(s.models||[]).find(x=>x.id===s.active),mode=s.active_mode||'shadow',hr=s.hitrate;
  if(!m){box.innerHTML=`<div class="studio-active-empty"><span class="studio-model-mark">${I.info}</span><div><h3>尚未启用专属模型</h3><p>当前使用基础检测录制候选，完成训练后可在这里启用新版本。</p></div><button class="btn ghost" onclick="setLab('train',$('#labTabs').children[1])">准备训练</button></div>`;return;}
  box.innerHTML=`<div class="studio-active-summary"><div class="studio-model-identity"><span class="studio-model-mark">${I.check}</span><div><h3>${esc(m.id)} <span>生效中</span></h3><p>${esc(m.report?.backend||'预置动作模型')} · ${m.created_ts?fmtTime(m.created_ts):'预置版本'}</p></div></div><div class="studio-active-metric"><span>${m.report?'独立测试准确率':'发布评估准确率'}</span><strong>${fmtAcc(m.top1)}</strong></div><div class="studio-active-metric"><span>人工复核命中率</span><strong>${hr?.total?fmtAcc(hr.rate):'—'}</strong><small>${hr?.total?`${hr.correct} / ${hr.total} 段`:'等待新片段人工复核'}</small></div></div>
    <div class="studio-mode-row"><div class="seg-ctl"><button class="${mode==='shadow'?'on':''}" onclick="activate('${esc(m.id)}','shadow')">测试模式</button><button class="${mode==='gate'?'on':''}" onclick="activate('${esc(m.id)}','gate')">自动判定模式</button></div><p>${mode==='gate'?'录像结束后，模型确认喝水即自动加入喝水记录并发送邮件（需开启邮件），不确定项保留复核。':'先预测打分，继续录制候选；通过人工复核观察实际效果。'}</p></div>`;
}
function renderModels(s){
  const box=$('#modelList');if(!box)return;
  const models=s.models||[];
  box.innerHTML=(models.length?models.map(m=>{const cur=m.id===s.active,r=m.report,c=r?.counts||m.image_counts||{};
    return `<article class="studio-version ${cur?'current':''}"><div class="studio-version-title"><div><h3>${esc(m.id)} ${cur?'<span class="tag ok">生效中</span>':''}</h3><p>${fmtTime(m.created_ts)} · ${esc(r?.backend||'预置 / 旧版模型')}</p></div><button class="mbtn ${cur?'cur':''}" ${cur?'disabled':`onclick="activate('${esc(m.id)}')"`}>${cur?'当前版本':'启用此版本'}</button></div><div class="studio-version-metrics"><div><span>${r?'独立测试准确率':'发布评估准确率'}</span><strong>${fmtAcc(m.top1)}</strong></div><div><span>喝水召回率</span><strong>${fmtAcc(r?.drinking_recall)}</strong></div><div><span>喝水精确率</span><strong>${fmtAcc(r?.drinking_precision)}</strong></div><div><span>漏判 / 误判</span><strong>${r?.confusion?.missed_drinking??'—'} <small>/</small> ${r?.confusion?.false_drinking??'—'}</strong></div></div><div class="studio-version-footer"><span>${r?`训练数据 · 喝水 ${c.drinking||0} / 没喝动作 ${c.not_drinking||0} 段`:'尚未在当前场景独立评估'}</span><span>${r?(r.auto_label_eligible?'满足自动判定评估条件':'继续测试与纠错'):'通过人工复核检验实际效果'}</span><a class="btn ghost" href="/api/model/export/${encodeURIComponent(m.id)}" download>导出分享模型</a></div>${r?.warnings?.length?`<details class="studio-explainer"><summary>评估说明</summary><p>${r.warnings.map(esc).join(' ')}</p></details>`:''}</article>`;
  }).join(''):`<div class="studio-panel studio-empty"><h3>还没有专属模型版本</h3><p>标注喝水和没喝动作，再开始第一次训练。</p><button class="btn ghost" onclick="setLab('train',$('#labTabs').children[1])">开始准备</button></div>`)+`<div class="studio-disable-model"><span>${s.active?'需要暂停使用专属模型时，可退回基础检测。':'当前使用基础检测录制候选。'}</span><button class="mbtn off" ${s.active?'onclick="activate(null)"':'disabled'}>${s.active?'停用专属模型':'基础检测运行中'}</button></div>`;
}
async function activate(id,mode){
  $$('.mbtn,#activeBox .seg-ctl button').forEach(b=>b.disabled=true);
  try{const r=await (await fetch('/api/model/activate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,mode:mode||'shadow'})})).json();
    if(r.detail){toast(r.detail,'err');return;} if(r&&r.note)toast(r.note,'info');else toast(id?`${id} 已生效`:'已停用模型');}
  catch(e){toast('操作失败','err');}
  pollTrain();
}

async function importSharedModel(input){
  const file=input.files?.[0];if(!file)return;
  try{
    if(file.size>128*1024*1024)throw Error('模型包超过 128 MB');
    const response=await fetch('/api/model/import',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});
    const result=await response.json();if(!response.ok)throw Error(result.detail||'导入失败');
    toast(result.note);pollTrain();
  }catch(error){toast(error.message||'导入失败','err');}
  finally{input.value='';}
}
let recheckTimer=null;
async function recheckClips(){
  const r=await (await fetch('/api/model/review',{method:'POST'})).json();
  if(!r.started){toast('尚无可用模型或评估正在进行','err');return;}pollRecheck();
}
async function pollRecheck(){
  const s=await (await fetch('/api/model/review/status')).json();const box=$('#recheckStatus');
  if(box)box.textContent=s.state==='running'?`${s.model} 正在重新评估 ${s.done}/${s.total} 段`:(s.state==='done'?`${s.model} 已重新评估 ${s.done} 段；人工结论保留`:(s.detail||''));

  if(s.state!=='running'&&recheckTimer){recheckTimer=null;await checkClipUpdates();}
}
function renderModelProgress(s){
  const box=$('#modelProgress');if(!box)return;
  const trained=(s.models||[]).filter(m=>m.report);
  if(!trained.length){box.innerHTML=`<div class="studio-growth-empty"><span>${I.clock}</span><div><h3>第一次训练，就是成长的起点</h3><p>完成后展示独立测试的漏判、误判；后续版本会在相同测试片段上比较，避免样本变化造成虚假进步。</p></div></div>`;return;}
  const opened=new Set([...box.querySelectorAll('details[open]')].map(e=>e.dataset.version));
  box.innerHTML=trained.map((m,i)=>{const r=m.report,c=r.comparison,conf=r.confusion||{},counts=r.counts||{};
    const before=c?.before,after=c?.after;
    const comparable=before&&after&&c.test_names?.length;
    const change=comparable?`相同 ${c.test_names.length} 段独立片段 · 漏判 ${before.confusion?.missed_drinking??'—'} → ${after.confusion?.missed_drinking??'—'} · 误判 ${before.confusion?.false_drinking??'—'} → ${after.confusion?.false_drinking??'—'}`:(c?.message||'首次基线，后续版本会用相同独立片段比较。');
    const graph=comparable?`<div class="studio-comparison"><div><span>${esc(c.version)}</span><i><b style="width:${Math.max(0,Math.min(100,before.balanced_accuracy*100))}%"></b></i><strong>${fmtAcc(before.balanced_accuracy)}</strong></div><div class="latest"><span>${esc(m.id)}</span><i><b style="width:${Math.max(0,Math.min(100,after.balanced_accuracy*100))}%"></b></i><strong>${fmtAcc(after.balanced_accuracy)}</strong></div><small>同一批测试片段上的平衡准确率</small></div>`:'';
    return `<details class="studio-growth-version" data-version="${esc(m.id)}" ${i===0||opened.has(m.id)?'open':''}><summary><span><b>${esc(m.id)}</b><small>人工标注 ${(counts.drinking||0)+(counts.not_drinking||0)} 段</small></span><span>召回 ${fmtAcc(r.drinking_recall)} · 精确 ${fmtAcc(r.drinking_precision)}</span></summary><div class="studio-growth-detail"><div class="studio-growth-numbers"><span>漏判喝水 <b>${conf.missed_drinking??'—'}</b> 段</span><span>误判喝水 <b>${conf.false_drinking??'—'}</b> 段</span></div>${graph}<p>${esc(change)}</p>${r.warnings?.length?`<p class="lab-note">${r.warnings.map(esc).join(' ')}</p>`:''}</div></details>`;
  }).join('');
}
async function loadAIStatus(){
  try{const s=await (await fetch('/api/ai/status')).json();const box=$('#aiBudgetStatus');if(!box)return;
    box.textContent=s.enabled?`外部老师今日已用 ${s.budget?.used??0}/${s.budget?.limit??40} 次（含失败），待处理 ${s.queue.waiting||0} 段，辅助标注 ${s.queue.labeled||0} 段。${s.last_detail||''}`:'外部老师未启用；可完全使用人工标注和本地训练。';
  }catch(e){}
}
const OPENROUTER_BASE='https://openrouter.ai/api/v1';
async function openAISetup(){
  const s=await (await fetch('/api/setup')).json();if(!s.enabled){toast('当前入口没有场景设置','err');return;}
  const base=s.ai_base_url||OPENROUTER_BASE;
  openModal(`<span class="fountain-kicker">辅助标注 · 可选</span><h3>视觉老师</h3><p class="msub">本地筛选有效片段，再请视觉模型辅助判断。训练和监控仍在本机运行。</p>
    <div class="teacher-form">
      <label class="teacher-switch"><input type="checkbox" id="aiEnable" ${s.ai_enabled?'checked':''}><span>启用辅助标注<small>启用后向所选服务发送视频抽帧</small></span></label>
      <label>服务渠道<select id="aiChannel" onchange="changeAIChannel()"><option value="openrouter" ${base===OPENROUTER_BASE?'selected':''}>OpenRouter · 官方</option><option value="custom" ${base!==OPENROUTER_BASE?'selected':''}>自定义 · OpenAI 兼容接口</option></select></label>
      <label>Base URL<input type="url" id="aiBase" value="${esc(base)}" placeholder="${OPENROUTER_BASE}" oninput="$('#aiChannel').value=this.value===OPENROUTER_BASE?'openrouter':'custom';clearAITestResult()"><small>默认 OpenRouter，可改为代理的 /v1 地址，不含 /chat/completions。</small></label>
      <label>API Key<input type="password" id="aiKey" autocomplete="new-password" placeholder="${s.has_ai_key?'已保存，留空保留当前密钥':'输入服务密钥'}" oninput="clearAITestResult()"></label>
      <div class="teacher-fields"><label>视觉模型<input type="text" id="aiTeacher" list="teacherModels" value="${esc(s.ai_model||'')}" placeholder="填写模型 ID" oninput="clearAITestResult()"><datalist id="teacherModels"></datalist></label><label>每日调用上限<input type="number" id="aiLimit" min="1" step="1" value="${s.ai_daily_budget||40}" oninput="clearAITestResult()"></label></div>
      <small>上限可自由填写。OpenRouter 免费账户通常每日 50 次，默认 40 次留出余量；按您的账户额度调整，失败尝试和视觉测试也计入。</small>
    </div>
    <div class="teacher-test"><div><strong>先确认接口可用</strong><small>测试使用当前填写内容，不保存设置或生成训练标签。</small></div><div class="teacher-test-buttons"><button class="btn ghost" data-ai-test onclick="testAISetup('connection')">测试连接 / 获取模型</button><button class="btn ghost" data-ai-test onclick="testAISetup('vision')">测试视觉 · 发送当前画面</button></div><div id="aiTestResult" class="teacher-test-result" role="status" aria-live="polite" hidden></div></div>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveAISetup()">保存设置</button></div>`);
}
function changeAIChannel(){if($('#aiChannel').value==='openrouter')$('#aiBase').value=OPENROUTER_BASE;clearAITestResult();}
function clearAITestResult(){const box=$('#aiTestResult');if(box&&!box.dataset.running){box.hidden=true;box.textContent='';}}
function aiSetupBody(){
  const limit=Number($('#aiLimit').value);
  if(!Number.isSafeInteger(limit)||limit<1){toast('每日调用上限请填写正整数','err');return null;}
  const body={ai_enabled:$('#aiEnable').checked,ai_base_url:$('#aiBase').value,ai_model:$('#aiTeacher').value,ai_daily_budget:limit};
  if($('#aiKey').value)body.ai_api_key=$('#aiKey').value;
  return body;
}
async function testAISetup(kind){
  const body=aiSetupBody();if(!body)return;
  const box=$('#aiTestResult');box.hidden=false;box.className='teacher-test-result';box.dataset.running='true';
  box.textContent=kind==='vision'?'正在发送当前画面，测试视觉推理…':'正在连接并读取模型列表…';
  const buttons=$$('[data-ai-test]');buttons.forEach(b=>b.disabled=true);
  const form=JSON.stringify(body);
  try{
    const r=await fetch('/api/ai/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,kind})}),d=await r.json();
    if($('#aiTestResult')!==box)return;
    const changed=JSON.stringify(aiSetupBody())!==form;
    box.classList.add(d.ok?'success':'failure');
    box.innerHTML=`<strong>${esc(d.message||d.detail||'测试失败')}${d.http_status?` · HTTP ${d.http_status}`:''}</strong>${d.latency_ms!=null?`<small>响应 ${(d.latency_ms/1000).toFixed(1)} 秒</small>`:''}${d.reply?`<p>${esc(d.reply)}</p>`:''}${kind==='connection'&&d.ok?`<p>${d.model_available?'所填模型已在列表中。':'所填模型未在列表中，请选择服务支持的模型。'}连接成功不代表视觉推理已可用。</p>`:''}${changed?'<p>测试期间配置已修改，请按当前内容重新测试。</p>':''}`;
    if(d.models&&!changed)$('#teacherModels').innerHTML=d.models.map(id=>`<option value="${esc(id)}"></option>`).join('');
    loadAIStatus();
  }catch(e){if($('#aiTestResult')===box){box.classList.add('failure');box.textContent='测试请求失败，请检查网络和服务状态。';}}
  finally{delete box.dataset.running;buttons.forEach(b=>b.disabled=false);}
}
async function openAutoConfirmSetup(){
  const s=await (await fetch('/api/setup')).json();
  openModal(`<h3>高分自动确认</h3><p class="msub">喝水得分 ≥99.5% 记为喝水，≤0.5% 记为没喝，中间得分保留待复核。得分是模型判断，不是正确率；可以在「全部」中抽查和改判。</p><div class="teacher-form"><label class="teacher-switch"><input type="checkbox" id="autoConfirm" ${s.auto_confirm_extremes?'checked':''}> 启用高分自动确认</label><small>自动结果参与次数及饮水估值统计；人工确认或纠错后才作为训练数据。开启时也会重新评估已有录像。</small></div><div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveAutoConfirmSetup()">保存设置</button></div>`);
}
async function saveAutoConfirmSetup(){
  const r=await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({auto_confirm_extremes:$('#autoConfirm').checked})}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();await loadLabelClips(false);loadStats();loadHealth();toast(d.auto_confirm_extremes?'已开启，已有录像正在重新评估':'已关闭高分自动确认');
}
async function saveAISetup(){
  const body=aiSetupBody();if(!body)return;
  const r=await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();loadAIStatus();toast('视觉老师设置已生效');
}
let sceneROI=null,sceneTarget="bowl";
async function openSceneSetup(target="bowl"){
  sceneTarget=target;
  if(target==='flow')dispData=await (await fetch('/api/dispenser')).json();
  const setup=await (await fetch('/api/setup')).json();if(target!=='flow'&&!setup.enabled){toast('当前入口没有场景设置','err');return;}
  sceneROI=target==="flow"?dispData?.flow?.roi:setup.bowl_roi;
  if(!sceneROI){toast("请先加载饮水机状态","err");return;}
  openModal(`<h3>${target==="flow"?"框选出水口":"框选饮水盆和猫头动作区域"}</h3><p class="msub">${target==="flow"?"紧贴出水口和水柱框选（如喷泉冒水处），不要框整个水盆，尽量排除背景和猫咪经过的区域。调整区域会清除旧参考，请重新确认停止和正常出水。":"包含猫嘴和水盆，尽量排除背景。更换相机角度后请重新框选并重训。"}</p>
    <div style="position:relative"><img id="sceneImage" src="/snapshot.jpg?t=${Date.now()}" style="width:100%;display:block"><canvas id="sceneCanvas" style="position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair"></canvas></div>
    <div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveSceneSetup()">保存区域</button></div>`);
  const img=$('#sceneImage'),canvas=$('#sceneCanvas');
  const draw=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;const ctx=canvas.getContext('2d'),r=sceneROI;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#00ffff';ctx.lineWidth=3;ctx.strokeRect(r[0]*canvas.width,r[1]*canvas.height,(r[2]-r[0])*canvas.width,(r[3]-r[1])*canvas.height);};
  img.onload=draw;if(img.complete)draw();
  let start=null;const point=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];};
  canvas.onpointerdown=e=>{start=point(e);canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!start)return;const end=point(e);sceneROI=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];draw();};
  canvas.onpointerup=()=>{start=null;};
}
async function saveSceneSetup(){
  const r=await fetch(sceneTarget==='flow'?'/api/dispenser/flow/roi':'/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sceneTarget!=='bowl'?{roi:sceneROI}:{bowl_roi:sceneROI})}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();loadDispenser();toast(sceneTarget==='flow'?'出水口区域已保存，请重新确认参考':'饮水区域已保存，新训练会使用此区域');
}
async function loadHealth(){
  try{const r=await fetch('/api/health');if(!r.ok)throw new Error('health');const h=await r.json();
    if(h.camera)renderLiveStatus(h.camera);
    else {const liveQuality=$('#liveQuality');if(liveQuality){
      liveQuality.hidden=!!h.monitoring.usable;
      const liveTag=$('.live-tag');if(liveTag){liveTag.classList.toggle('offline',h.monitoring.status==='offline');liveTag.innerHTML=h.monitoring.status==='offline'?'<span class="live-dot"></span>重连中':'<span class="live-dot"></span>LIVE';}
      liveQuality.textContent=h.monitoring.status==='offline'?'摄像头连接中断，正在重连…':'画面细节不足，暂停记录';
    }}
    const building=h.status==='insufficient_data';
    const days=Math.min(3,h.baseline_days||0);
    const content=`<div class="health-summary"><div class="health-intro"><div class="health-eyebrow">饮水健康<span class="health-stage">${building?'积累数据中':'习惯变化'}</span></div><h3>${building?'了解小猫，从每一天开始':'关注小猫的饮水变化'}</h3><p>${esc(building?'积累 3 天有效监控后，建立专属饮水基线。当前记录仅供观察，尚不能判断健康变化。':h.message)}</p>
      ${building?`<div class="baseline-progress" aria-label="基线进度 ${days} / 3 天">${[1,2,3].map(i=>`<i class="${days>=i?'complete':''}"></i>`).join('')}<span>${days} / 3 天 · 每天有效监控至少 18 小时</span></div>`:''}</div>
      <div class="health-metrics"><div><span>有效监控</span><b>${Number(h.coverage_hours).toFixed(1)}<small>小时</small></b><em>今日可用画面</em></div><div><span>监控覆盖</span><b>${fmtPct(h.coverage_ratio)}</b><em>占今日已过时间</em></div><div><span>待复核片段</span><b>${h.pending_review}<small>段</small></b><em>${h.pending_review?'等待确认喝水动作':'暂无待处理片段'}</em></div></div></div>
      <details class="health-details"><summary><span class="health-camera"><i class="health-dot ${h.monitoring.usable?'':'unavailable'}"></i>${esc(h.monitoring.message||'暂无摄像头状态')}</span><span class="health-more">监控与估值说明</span></summary><div class="health-detail-body"><p>${esc(h.message)}</p><p>今日确认 ${h.today_count} 次，摄水估值 ${h.today_estimated_ml??'—'} ml · ${h.estimate_calibrated?'已做周期校准':'尚待周期校准'}${h.baseline_daily_count==null?'':` · 历史日均 ${h.baseline_daily_count.toFixed(1)} 次`}</p><p>${(h.actions||[]).map(esc).join(' ')}</p><p class="lab-note">${esc(h.note)}</p></div></details>`;
    for(const id of ['#healthHome','#healthTrend'])if($(id))$(id).innerHTML=content;
  }catch(e){for(const id of ['#healthHome','#healthTrend'])if($(id))$(id).textContent='监控状态加载失败，请检查服务。';}
}

/* ---------- 启动 ---------- */
// 合并同时到达的页面加载、操作回调与变更通知请求。
for(const name of ['loadStats','loadDispenser','loadHealth','pollTrain','pollTrainVideo','pollRecheck']){
  const original=window[name];let running=null;
  window[name]=function(...args){if(running)return running;running=Promise.resolve().then(()=>original(...args)).finally(()=>{running=null;});return running;};
}
bindCardFx($('#recGrid')); bindCardFx($('#labelClips'));
show((location.hash||'#/home').replace('#/','').replace('#',''));
loadStats();loadHealth();checkClipUpdates().catch(()=>{});
let liveEvents=null,liveRefreshTimer=null,liveRefreshRunning=false,livePendingTopics=new Set();
let liveRetryTimer=null,liveRetryFailures=0,liveStatusBusy=false,liveWasOffline=false;
function renderLiveStatus(camera){
  if(typeof camera.connected!=='boolean')return;
  const tag=$('.live-tag'),quality=$('#liveQuality');
  if(tag){tag.classList.toggle('offline',!camera.connected);tag.innerHTML='<span class="live-dot"></span>'+(camera.connected?(camera.detection_enabled===false?'预览中':'LIVE'):'重连中');}
  if(quality){
    quality.hidden=camera.connected&&camera.detection_enabled!==false&&camera.quality?.usable!==false;
    quality.textContent=!camera.connected?'摄像头暂时没有新画面，正在等待恢复…':camera.detection_enabled===false?'视频已连接 · 当前仅预览，识别未启动':'画面细节不足，暂停记录';
  }
  if(camera.connected){
    liveRetryFailures=0;
    if(liveWasOffline&&!document.hidden&&currentPage()==='home')restartLivePreview();
  }
  liveWasOffline=!camera.connected;
}
function restartLivePreview(){
  clearTimeout(liveRetryTimer);liveRetryTimer=null;
  if(document.hidden||currentPage()!=='home')return;
  const live=$('#live');if(live)live.src='/stream.mjpg?t='+Date.now();
}
async function pollLiveStatus(){
  if(document.hidden||currentPage()!=='home'||liveStatusBusy)return;
  liveStatusBusy=true;
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),4000);
  try{const response=await fetch('/api/live/status',{signal:controller.signal,cache:'no-store'});if(response.ok)renderLiveStatus(await response.json());}
  catch(error){const quality=$('#liveQuality'),tag=$('.live-tag');if(quality){quality.hidden=false;quality.textContent='本机服务暂不可用，正在重新连接…';}if(tag){tag.classList.add('offline');tag.innerHTML='<span class="live-dot"></span>服务重连中';}}
  finally{clearTimeout(timeout);liveStatusBusy=false;}
}
$('#live').addEventListener('error',()=>{
  if(document.hidden||currentPage()!=='home'||!$('#live').getAttribute('src')||liveRetryTimer)return;
  liveRetryTimer=setTimeout(restartLivePreview,Math.min(8000,500*2**Math.min(liveRetryFailures++,4)));
});
setInterval(pollLiveStatus,2500);
function currentPage(){return (location.hash||'#/home').replace('#/','');}
function updateLiveVisibility(){
  const live=$('#live');if(!live)return;
  const active=!document.hidden&&currentPage()==='home';
  if(active&&!live.getAttribute('src'))live.src='/stream.mjpg';
  else if(!active){clearTimeout(liveRetryTimer);liveRetryTimer=null;live.removeAttribute('src');}
  if(active)pollLiveStatus();
}
async function refreshLivePage(topics){
  const page=currentPage(),jobs=[];
  if(topics.has('data')){if(page==='lab')scheduleReviewRefresh();else jobs.push(loadStats(),checkClipUpdates());}
  if((page==='home'||page==='disp')&&['data','flow'].some(t=>topics.has(t)))jobs.push(loadDispenser());
  if((page==='home'||page==='trend')&&['data','monitoring','clock'].some(t=>topics.has(t)))jobs.push(loadHealth());
  if(page==='lab'&&topics.has('training'))jobs.push(pollTrain(),pollTrainVideo());
  if(page==='records'&&topics.has('data'))jobs.push(loadRecords());
  if(page==='replay'&&topics.has('data'))jobs.push(loadReplay(true));
  if(page==='trend'&&topics.has('data'))jobs.push(renderTrend());
  if(topics.has('clock')){const now=new Date(),label=$('#tlNow'),line=$('.tl-now');if(label)label.textContent='现在 '+now.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});if(line)line.style.left=((now.getHours()+now.getMinutes()/60)/24*100)+'%';}
  await Promise.allSettled(jobs);
}
function queueLiveRefresh(topics){
  topics.forEach(t=>livePendingTopics.add(t));
  if(liveRefreshTimer||liveRefreshRunning)return;
  liveRefreshTimer=setTimeout(async()=>{
    liveRefreshTimer=null;if(document.hidden)return;
    liveRefreshRunning=true;const next=new Set(livePendingTopics);livePendingTopics.clear();
    try{await refreshLivePage(next);}finally{liveRefreshRunning=false;if(livePendingTopics.size)queueLiveRefresh([]);}
  },250);
}
function connectLiveUpdates(){
  if(document.hidden||liveEvents)return;
  const stream=new EventSource('/api/events');liveEvents=stream;
  stream.addEventListener('auth-expired',()=>{stream.close();location.replace('/login');});
  stream.addEventListener('ready',()=>queueLiveRefresh(['data','flow','monitoring','training','clock']));
  stream.addEventListener('change',e=>{try{queueLiveRefresh(JSON.parse(e.data).topics||[]);}catch(error){}});
}
document.addEventListener('visibilitychange',()=>{updateLiveVisibility();if(document.hidden){liveEvents?.close();liveEvents=null;clearTimeout(liveRefreshTimer);liveRefreshTimer=null;}else connectLiveUpdates();});
window.addEventListener('hashchange',()=>{updateLiveVisibility();if(['home','trend'].includes(currentPage()))loadHealth();});
updateLiveVisibility();connectLiveUpdates();

function renderFlow(f){
  const alert=$('#flowAlert'),panel=$('#flowPanel');
  if(!f){alert.hidden=true;panel.textContent='出水监测未启用';return;}
  const recovery=f.recovery;
  const needsCalibration=!f.has_stopped_video||!f.has_flowing_video;
  alert.hidden=!f.alert&&!recovery&&!needsCalibration;
  alert.classList.toggle('recovered',!f.alert&&!needsCalibration);
  if(f.alert)alert.innerHTML=`<div><strong>疑似停止出水</strong><p>${esc(f.state==='suspected_stopped'?(f.cause_message||f.message):'上次检测到停水，正在观察恢复情况。')}</p></div><button class="btn ghost" onclick="go('disp')">检查饮水机</button>`;
  else if(recovery)alert.innerHTML=`<div class="recovery-scroll"><strong>出水已恢复</strong><span>${waterDate(recovery.start)} 至 ${waterDate(recovery.end)} 检测到停水 · ${waterDate(recovery.end)} 已恢复 · 今日累计停水 ${waterDuration(f.today_downtime_seconds||0)}</span></div><button class="recovery-dismiss" aria-label="关闭恢复提醒" onclick='dismissFlowRecovery(${JSON.stringify(recovery.id)})'>×</button>`;
  else if(needsCalibration)alert.innerHTML=`<div><strong>出水监测尚未完成校准</strong><p>还需保存${[!f.has_stopped_video?'停止出水':'',!f.has_flowing_video?'正常出水':''].filter(Boolean).join('和')}参考视频，才能完整判断停止与恢复。</p></div><button class="btn ghost" onclick="go('disp')">设置出水监测</button>`;
  const names={suspected_stopped:'疑似停止出水',flowing:'正在正常出水',unknown:'出水状态待确认',learning:'正在持续观察',unconfigured:'尚未设置出水监测'};
  const calibrationOpen=needsCalibration||panel.querySelector('.flow-calibration')?.open;
  const observation=f.observation;
  const referenceHint=!observation||Date.now()/1000-observation.observed_at>3?'正在等待出水口的最新画面。':!observation.usable?'框选区域光线或细节不足，请检查区域或补光。':observation.occluded?'检测到猫咪遮挡框选出水口，请等猫咪离开后保存。':(f.video_buffer_frames||0)<50?`正在积累连续视频（${f.video_buffer_frames||0}/50 帧），请保持当前状态约 6 秒。`:'出水口视频已就绪，当前未检测到猫咪遮挡，可保存参考。';
  panel.innerHTML=`<div class="flow-heading"><div><span class="flow-eyebrow">实时出水监测</span><h3>${esc(names[f.state]||names.unknown)}</h3></div><span class="tag ${f.alert?'bad':f.state==='flowing'?'ok':''}">${f.alert?'请检查':f.state==='flowing'?'出水正常':'观察中'}</span></div><p class="flow-description">${esc(f.alert?(f.cause_message||f.message):f.message)}</p>
    <details class="flow-calibration" ${calibrationOpen?'open':''}><summary>设置与校准出水画面 <span>停止视频${f.has_stopped_video?'已保存':'未保存'} · 正常视频${f.has_flowing_video?'已保存':'未保存'}</span></summary><p>框选出水口，关机后保持约 6 秒，保存停止参考视频；再开机保持约 6 秒，保存正常出水参考视频。系统比较连续视频里的水流变化，再持续确认停止与恢复。疑似停水仅提醒检查设备，不影响喝水确认和记录。</p><p>${esc(referenceHint)}</p><div class="flow-actions"><button class="btn ghost" onclick="openSceneSetup('flow')">框选出水口</button><button class="btn ghost" onclick="captureFlowReference('stopped')">保存停止参考视频</button><button class="btn" onclick="captureFlowReference('flowing')">保存正常出水参考视频</button>${f.has_stopped_video?'<a class="btn ghost" href="/api/dispenser/flow/reference/stopped.mp4" target="_blank" rel="noopener">查看停止视频</a>':''}${f.has_flowing_video?'<a class="btn ghost" href="/api/dispenser/flow/reference/flowing.mp4" target="_blank" rel="noopener">查看出水视频</a>':''}</div></details>`;
  renderDispenserRecords(dispData);
}
async function dismissFlowRecovery(id){
  const r=await fetch('/api/dispenser/flow/dismiss',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}loadDispenser();
}
async function captureFlowReference(kind){
  const r=await fetch('/api/dispenser/flow/reference',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind})});const d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}loadDispenser();toast('最近约 6 秒已保存为'+(kind==='stopped'?'停止出水':'正常出水')+'参考视频');
}

async function openMailSetup(){
  const r=await fetch('/api/mail'),d=await r.json();if(!r.ok||d.available===false){toast('邮件设置不可用','err');return;}
  openModal(`<h3>邮件提醒</h3><p class="msub">每次确认喝水都提醒；每天 23:00 发送今日汇总，周日改发本周汇总。电脑需要保持项目运行。疑似停止出水也会提醒。</p>
    <div class="mail-status">${esc(d.last_error||(d.enabled?(d.configured?'提醒已开启':'配置不完整'):'提醒未开启'))}${d.pending?` · ${d.pending} 封待发送`:''}${d.last_sent_at?`<br>最近发送：${esc(new Date(d.last_sent_at*1000).toLocaleString())}`:''}</div>
    <div class="mail-fields"><label class="mail-wide">收件邮箱<input id="mailTo" type="email" value="${esc(d.mail_to||'')}" autocomplete="email" placeholder="接收提醒的邮箱"></label>
    <label class="mail-wide">发件邮箱<input id="mailUser" type="email" value="${esc(d.smtp_user||'')}" placeholder="例如 xxx@qq.com"></label>
    <label class="mail-wide">SMTP 授权码<input id="mailPassword" type="password" autocomplete="new-password" placeholder="${d.has_password?'已保存，留空保留':'邮箱开启 SMTP 后获取的授权码'}"></label>
    <label class="mail-wide">SMTP 服务器<input id="mailHost" value="${esc(d.smtp_host||'smtp.qq.com')}"></label>
    <label>端口<input id="mailPort" type="number" value="${d.smtp_port||465}" min="1" max="65535"></label>
    <label>连接安全<select id="mailSecurity"><option value="ssl" ${d.smtp_security==='ssl'?'selected':''}>SSL（常用 465）</option><option value="starttls" ${d.smtp_security==='starttls'?'selected':''}>STARTTLS（常用 587）</option></select></label></div>
    <label class="mail-enable"><input type="checkbox" id="mailEnabled" ${d.enabled?'checked':''}>启用邮件提醒</label>
    <p class="msub">QQ 邮箱使用 SMTP 授权码。密码只在本机保存，不会在设置接口中返回。时间以电脑本地时区为准，汇总统计截至 23:00。</p>
    <div class="modal-acts"><button class="btn ghost" onclick="testMailSettings()">保存并发测试邮件</button><button class="btn" onclick="saveMailSettings()">保存设置</button></div>`);
}
async function saveMailSettings(close=true){
  const body={enabled:$('#mailEnabled').checked,smtp_host:$('#mailHost').value,smtp_port:Number($('#mailPort').value),smtp_security:$('#mailSecurity').value,smtp_user:$('#mailUser').value,mail_to:$('#mailTo').value};
  if($('#mailPassword').value)body.smtp_password=$('#mailPassword').value;
  const r=await fetch('/api/mail',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return false;}if(close){closeModal();toast('邮件设置已保存');}return true;
}
async function testMailSettings(){
  if(!await saveMailSettings(false))return;
  const r=await fetch('/api/mail/test',{method:'POST'}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();toast('测试邮件已加入发送队列，可在邮件设置查看结果');
}

function correctDrinkRecord(clip){
  openModal(`<h3>修改喝水记录</h3><p class="msub">把喝水动作和是否实际喝到水分别记录。可先填写原因，再选择实际结果。</p>
    <div class="outcome-options"><button onclick='saveDrinkOutcome(${JSON.stringify(clip)},"drinking")'><strong>确实喝到了水</strong><span>计入饮水统计，动作模型使用正样本</span></button><button onclick='saveDrinkOutcome(${JSON.stringify(clip)},"dry_lick")'><strong>有舔水动作，但没喝到水</strong><span>例如换水、缺水；不计饮水量，动作模型仍使用正样本</span></button><button onclick='saveDrinkOutcome(${JSON.stringify(clip)},"no_action")'><strong>没有喝水动作，是识别错了</strong><span>不计饮水量，作为动作模型的负样本</span></button></div>
    <label class="water-input-label">原因补充（可选）<input id="drinkOutcomeReason" maxlength="500" placeholder="例如正在换水，舔了几下就离开"></label>`);
}
async function saveDrinkOutcome(clip,outcome){
  const reason=$('#drinkOutcomeReason')?.value||({dry_lick:'有舔水动作但未实际喝到水',no_action:'没有喝水动作，模型误识别',drinking:'人工确认实际喝到水'})[outcome];
  const r=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip,is_drinking:outcome==='drinking',outcome,reason})}),d=await r.json();
  if(!r.ok){toast(d.detail||'纠正失败','err');return;}closeModal();if(page==='lab')await loadLabelClips(false);else{await fetchClips();renderRecords();}loadStats();loadDispenser();loadHealth();toast(outcome==='dry_lick'?'已记为舔水未喝到：不计饮水量，保留动作正样本':'人工结果已更新');
}
async function openWaterReminder(){
  const d=dispData||{},r=d.reminder||{},o=d.options||{};
  openModal(`<h3>提醒与设备设置</h3><p class="msub">没有预设提醒水位。饮水机提醒水少时，由您测量并确认；也可以在完整周期后自行指定。</p>
    <div class="mail-fields"><label class="mail-wide">饮水机容量（可选）<input id="waterCapacity" type="number" min="1" max="20000" value="${o.capacity_ml??''}" placeholder="标称最大容量 ml"></label><label class="mail-wide">水少提醒时的实测剩余量（ml）<input id="waterLow" type="number" min="1" max="20000" value="${r.low_ml??''}" placeholder="未测量留空"></label></div>
    <label class="mail-enable"><input type="checkbox" id="waterRemindEnabled" ${r.enabled?'checked':''}>开启软件估算水少提醒</label><p class="msub">${r.ready?'已有完整周期，可按您设置的水位提醒。':'至少完成一轮确认喝水和实测残水后才会触发软件估算提醒。'}</p>
    <p class="msub">设备指示灯由您观察，无需让摄像头看到。出水停止和恢复通过出水口画面判断。</p><div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveWaterReminder()">保存设置</button></div>`);
}
async function saveWaterReminder(){
  const capacity=$('#waterCapacity').value,low=$('#waterLow').value;
  const body={capacity_ml:capacity===''?null:Number(capacity),reminder_low_ml:low===''?null:Number(low),reminder_enabled:$('#waterRemindEnabled').checked,indicator_enabled:false};
  const r=await fetch('/api/dispenser/reminder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();loadDispenser();toast('已保存您自己的提醒设置');
}
function recordWaterSignal(kind){
  openModal(`<h3>${kind==='low'?'确认水少提醒 · 学习水位':'确认停止出水 · 测量残水'}</h3><p class="msub">请测量现在实际剩下多少水。平台会记住这台设备的提醒水位，停水残水需在本轮结束时再测一次。</p><label class="water-input-label">当前实测剩余量<input id="signalRemaining" type="number" min="0" max="20000" placeholder="ml，不预填零"></label><div class="modal-acts"><button class="btn ghost" onclick="closeModal()">取消</button><button class="btn" onclick="saveWaterSignal('${kind}')">保存实测并学习</button></div>`);
}
async function saveWaterSignal(kind){
  const value=$('#signalRemaining').value;if(value===''){toast('请先测量当前剩余量','err');return;}
  const r=await fetch('/api/dispenser/signal',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,remaining_ml:Number(value)})}),d=await r.json();
  if(!r.ok){toast(d.detail,'err');return;}closeModal();loadDispenser();toast('已记住实际提醒水位，软件提醒会等待完整周期数据');
}
