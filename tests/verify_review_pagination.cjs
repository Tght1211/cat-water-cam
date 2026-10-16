const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2]||'/tmp/catcam-push-main-20261017/catcam/static/app.js','utf8');
const nodes=new Map(),requests=[];
const get=key=>{if(!nodes.has(key))nodes.set(key,{innerHTML:'',textContent:'',hidden:false,querySelectorAll:()=>[],insertAdjacentHTML(where,html){this.innerHTML+=html;}});return nodes.get(key);};
let handler;
const context=vm.createContext({AbortController,URLSearchParams,setTimeout,clearTimeout,console,
 $:get,$$:()=>[],clipsData:null,skeletons:()=>'<skeleton>',
 clipLoadError:(selector)=>get(selector).innerHTML='retry',
 fetch:async(url,opts)=>{requests.push(url);assert(url.startsWith('/api/clips/page?'));return handler(url,opts);},
 isReviewCandidate:n=>context.clipsData.review_candidates?.[n]!==false,
});
vm.runInContext(source.slice(source.indexOf("let labelFilter='review'"),source.indexOf('function labelDuration(')),context);
vm.runInContext("labelCard=(n)=>'<card>'+n+'</card>';",context);
const run=code=>vm.runInContext(code,context);
function data(names,cursor,snapshot='clip_3000.mp4'){
 const d={clips:names,next_cursor:cursor,snapshot,counts:{all:3,review:3,yes:0,no:0},recording_fps:30,retention_days:3};
 for(const key of ['labels','durations','meta','notes','review_candidates','candidate_only','water_review','predictions','prediction_details'])d[key]={};
 for(const n of names){d.labels[n]=null;d.meta[n]=null;d.review_candidates[n]=true;}
 return {ok:true,json:async()=>d};
}
(async()=>{
 handler=async()=>data(['clip_3000.mp4','clip_2000.mp4'],'clip_2000.mp4');
 await run('loadLabelClips()');
 assert(requests[0].includes('limit=12'));
 assert.equal(get('#labelClips').innerHTML.includes('clip_3000.mp4'),true);
 get('#labelClips').innerHTML+='PLAYING_CARD_MARKER';
 const button={disabled:false,removed:false,insertAdjacentHTML(where,html){get('#labelClips').innerHTML+=html;},remove(){this.removed=true;}};
 context.button=button;
 handler=async()=>data(['clip_1000.mp4'],null);
 await run('loadMoreLabelClips(button)');
 assert(requests[1].includes('cursor=clip_2000.mp4'));
 assert(requests[1].includes('snapshot=clip_3000.mp4'));
 assert(get('#labelClips').innerHTML.includes('PLAYING_CARD_MARKER'));
 assert.equal(button.removed,true);
 assert.equal(run('labState.names.length'),3);
 // 切换筛选后旧请求晚到，也不能覆盖新筛选结果。
 const pending=[];handler=()=>new Promise(resolve=>pending.push(resolve));
 const old=run('loadLabelClips(false)');
 run("labelFilter='no'");const latest=run('loadLabelClips(false)');
 pending[1](data(['clip_1000.mp4'],null));await latest;
 pending[0](data(['clip_3000.mp4'],null));await old;
 assert.equal(run('labState.names[0]'),'clip_1000.mp4');
 assert.equal(get('#labelClips').innerHTML.includes('clip_3000.mp4'),false);
 // 请求失败可重试，不一直显示骨架屏。
 handler=async()=>{throw Error('offline');};await run('loadLabelClips()');
 assert.equal(get('#labelClips').innerHTML,'retry');
 console.log('Passed: paged requests, cursor/snapshot, appended cards, stale response isolation, retry.');
})().catch(e=>{console.error(e);process.exitCode=1;});
