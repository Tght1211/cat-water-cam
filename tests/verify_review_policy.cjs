// 浏览器卡片使用服务端的统一复核结果，不能再次把确定结果标为待复核。
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2]||'catcam/static/app.js','utf8');
const data={meta:{certain:{source:'local',is_drinking:false,confidence:0},uncertain:null,failed:null},
  review_required:{certain:false,uncertain:true,failed:true},
  prediction_details:{certain:{drinking:false,confidence:0},uncertain:{drinking:false,confidence:.3}}};
const context=vm.createContext({clipsData:data,isReviewCandidate:()=>true,esc:x=>x});
vm.runInContext(source.slice(source.indexOf('function modelDetail('),source.indexOf('async function fetchLabelPage(')),context);
const run=code=>vm.runInContext(code,context);
assert.equal(run("needsReview('certain')"),false);
assert(!run("modelTag('certain')").includes('待复核'));
assert(run("modelTag('certain')").includes('模型判断：未发现喝水'));
assert.equal(run("needsReview('uncertain')"),true);
assert(run("modelTag('uncertain')").includes('待复核'));
assert.equal(run("needsReview('failed')"),true);
data.meta.uncertain={source:'human'};
assert.equal(run("needsReview('uncertain')"),false);
console.log('通过：确定没喝不提示复核，不确定与判断失败保留，人工确认立即生效。');
