// 连续复核应独立保存、合并统计刷新，并在失败时保留原卡片。
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2]||'catcam/static/app.js','utf8');
const cards=new Map(),saved=[],refreshes=[];
let fail=false;
const names=Array.from({length:10},(_,i)=>`clip_${i+1000}.mp4`);
for(const name of names)cards.set(name,{removed:false,querySelectorAll:()=>[],remove(){this.removed=true;}});
const data={labels:Object.fromEntries(names.map(n=>[n,null])),meta:{}};
const context=vm.createContext({setTimeout,clearTimeout,Promise,console,CSS:{escape:x=>x},
 clipsData:data,labState:{counts:{review:10},busy:false},
 $:selector=>[...cards].find(([name])=>selector.includes('"'+name+'"'))?.[1]||null,
 needsReview:name=>!data.meta[name],labelItems:()=>[],updateLabelChipCounts:()=>{},
 refreshLabelCounts:async()=>refreshes.push('counts'),checkClipUpdates:async()=>refreshes.push('summary'),
 toast:()=>{},loadMoreLabelClips:()=>{},
 fetch:async(url,options)=>{
   assert.equal(url,'/api/feedback');const body=JSON.parse(options.body);saved.push(body.clip);
   return {ok:!fail,json:async()=>fail?{detail:'保存失败'}:{meta:{source:'human',is_drinking:body.is_drinking}}};
 }});
vm.runInContext(source.slice(source.indexOf('let reviewMutation='),source.indexOf('function explainClip(')),context);
(async()=>{
 await Promise.all(names.map(name=>vm.runInContext(`fb(${JSON.stringify(name)},true)`,context)));
 assert.equal(saved.length,10);
 assert.equal(context.labState.counts.review,0);
 assert(names.every(n=>cards.get(n).removed&&data.labels[n]===true));
 await new Promise(resolve=>setTimeout(resolve,500));
 assert.deepEqual(refreshes,['counts','summary']);
 fail=true;const card=cards.get(names[0]);card.removed=false;
 await vm.runInContext(`fb(${JSON.stringify(names[0])},false)`,context);
 assert.equal(card.removed,false);assert.equal(data.labels[names[0]],true);
 console.log('Passed: 10 rapid reviews saved independently, one statistics refresh, failed save preserves card.');
})().catch(e=>{console.error(e);process.exitCode=1;});
